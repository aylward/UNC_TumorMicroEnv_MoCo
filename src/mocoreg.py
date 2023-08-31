import numpy as np

import math

import itk
from itk import TubeTK as tube

import csv


class mocoreg:
    def __init__(self, smooth_registrations=True, debug=False):
        self.debug = debug

        self.smooth_registrations = smooth_registrations
        if smooth_registrations:
            self.register_to_prior_frame=True
            self.register_to_zero_frame=True
            self.multi_frame_smoothing_window_size = 2
            self.keyframe_transform_smoothing_window_size = 3
            self.frames_per_second = 200
            self.seconds_per_change = 0.333
            self.max_keyframe_interval = int(
                self.frames_per_second * self.seconds_per_change / 2
            )
            self.keyframe_search_stepsize = max(2, int(self.max_keyframe_interval * 0.05))
            self.registration_metric = "MEAN_SQUARED_ERROR_METRIC"
            self.feature_size = 0.75
            self.expected_offset = 2
            self.expected_rotation = 0.02
        else:
            self.register_to_prior_frame=False
            self.register_to_zero_frame=True
            self.multi_frame_smoothing_window_size = 0
            self.keyframe_transform_smoothing_window_size = 0
            self.frames_per_second = 1
            self.seconds_per_change = 1
            self.max_keyframe_interval = 1
            self.keyframe_search_stepsize = 1
            #self.registration_metric = "MATTES_MI_METRIC"
            self.registration_metric = "MEAN_SQUARED_ERROR_METRIC"
            self.feature_size = 0.75
            self.expected_offset = 2
            self.expected_rotation = 0.02

        print("smooth_registrations", self.smooth_registrations)
        print("feature_size", self.feature_size)
        print("multi_frame_smoothing_window_size", self.multi_frame_smoothing_window_size)
        print("keyframe_transform_smoothing_window_size", self.keyframe_transform_smoothing_window_size)
        print("max_keyframe_interval", self.max_keyframe_interval)
        print("keyframe_search_stepsize", self.keyframe_search_stepsize)

        
        self.data_array = []
        self.keyframes = []
        self.keyframe_transforms = []

        self.data_array_reg = []

    def read_4d_bmode_matlab_file(
        self, filename, nlateral=92, nframes=200, ndepth=153, nelevation=102, bits=32, permute=False
    ):
        with open(filename, mode="rb") as file:
            file_content = file.read()
        pixel_type = None
        if bits == 32:
            pixel_type = np.dtype(np.float32)
        elif bits == 64:
            pixel_type = np.dtype(np.float64)
        elif bits == 16:
            pixel_type = np.dtype(np.float16)
        else:
            print("ERROR: Only 16, 32, and 64 bit floats supported")
            self.data_array = []
            return

        if permute:
            pixel_type.newbyteorder("<")
            data_1d = np.frombuffer(file_content, dtype=pixel_type)
            data_raw = np.reshape(
                data_1d, [nlateral, nframes, ndepth, nelevation], order="C"
            )
            self.data_array = np.transpose(data_raw, (1, 3, 0, 2))
        else:
            data_1d = np.frombuffer(file_content)
            data_raw = np.reshape(
                data_1d, [nframes, nelevation, nlateral, ndepth], order="C"
            )
            self.data_array = data_raw

    def import_4d_bmode_matlab_data(self, data):
        # data_raw = np.reshape(data_1d,
        # [nlateral, nframes, ndepth, nelevation],
        # order='C')
        # self.data_array = np.transpose(data_raw, (1,3,0,2))

        self.data_array = np.asarray(data)

    def compute_keyframes(self, data_array=[], tolerance=1.0):
        if len(data_array) == 0:
            data_array = self.data_array
            if len(data_array) == 0:
                print("ERROR: No data found for computing key frames")
                return

        if self.max_keyframe_interval > 1:
            diff = np.zeros(
                [
                    data_array.shape[0] // self.keyframe_search_stepsize,
                    self.max_keyframe_interval // self.keyframe_search_stepsize,
                ]
            )
            for frame_num in range(diff.shape[0]):
                for win_num in range(diff.shape[1]):
                    win_min = frame_num * self.keyframe_search_stepsize
                    win_max = min(
                        win_min + win_num * self.keyframe_search_stepsize,
                        data_array.shape[0] - 1,
                    )
                    tmp_diff = np.sum(
                        np.abs(data_array[win_min, :, :, :] - data_array[win_max, :, :, :])
                    )
                    diff[frame_num, win_num] = tmp_diff
    
            diff_img = itk.GetImageFromArray(diff.astype(np.float32))
            diff_img_blur = itk.SmoothingRecursiveGaussianImageFilter(
                diff_img,
                sigma=(
                    self.frames_per_second
                    * self.seconds_per_change
                    / 2
                    / self.keyframe_search_stepsize
                ),
            )
            diff_blur = itk.GetArrayFromImage(diff_img_blur)
    
            diff_avg = np.average(diff_blur) * tolerance
            diff_thresh = np.where(diff_blur <= diff_avg, 1, 0)
            win_limit = np.argmin(diff_thresh, axis=1) * self.keyframe_search_stepsize
            win_limit = np.where(win_limit == 0, self.max_keyframe_interval - 1, win_limit)
    
            self.keyframes = []
            keyframe = 0
            self.keyframes = [0]
            while keyframe < data_array.shape[0] - 1:
                keyframe += win_limit[keyframe // self.keyframe_search_stepsize]
                keyframe = min(keyframe, data_array.shape[0] - 1)
                self.keyframes.append(keyframe)
        else:
            self.keyframes = [x for x in range(data_array.shape[0])]
    
    def smooth_frame(self, data, frame_num):
        tmp_data = []
        if self.multi_frame_smoothing_window_size>0:
            t_min = int(max(frame_num - self.multi_frame_smoothing_window_size, 0))
            t_max = int(min(frame_num + self.multi_frame_smoothing_window_size + 1, data.shape[0]))

            tmp_data = np.average(data[t_min:t_max], axis=0)
        else:
            tmp_data = data[frame_num]

        img_data = itk.GetImageFromArray(tmp_data.astype(np.float32))
        if self.feature_size > 0:
            img_data_median = itk.MedianImageFilter(
                img_data, radius=max(1, int(self.feature_size))
            )
            img_data_blur = itk.SmoothingRecursiveGaussianImageFilter(
                img_data_median, sigma=self.feature_size
            )
            tmp_data_blur = itk.GetArrayFromImage(img_data_blur)
            return tmp_data_blur, img_data_blur
        else:
            tmp_data = itk.GetArrayFromImage(img_data)
            return tmp_data, img_data


    def compute_inter_keyframe_diffs(self, data_array=[], keyframes=[]):
        if len(data_array) == 0:
            data_array = self.data_array
            if len(data_array) == 0:
                print("ERROR: No data found for computing keyframe differences.")
                return
        if len(keyframes) == 0:
            keyframes = self.keyframes
            if len(keyframes) == 0:
                print("Info: computing keyframes")
                self.compute_keyframes(data_array)
                if len(self.keyframes) == 0:
                    print("ERROR: keyframes not defined")
                    return
                keyframes = self.keyframes
        ref_frame = data_array[keyframes[0]]
        diffs = []
        diff_rmse = 0
        diff_count = len(keyframes)
        for i in range(1, diff_count):
            frame = data_array[keyframes[i]]
            diff_frame = np.where(frame > 0, ref_frame-frame, 0)
            diff = np.average(np.square(diff_frame))
            diffs.append(math.sqrt(diff))
            diff_rmse += diff
            # Always compare to prior frame
            #ref_frame = frame

        diff_rmse /= diff_count - 1
        diff_rmse = math.sqrt(diff_rmse)

        return diff_rmse, diffs

    def register_keyframes(self, keyframes=[]):
        if len(keyframes) == 0:
            keyframes = self.keyframes
        if len(keyframes) == 0:
            self.compute_keyframes()
            if len(self.keyframes) == 0:
                print("ERROR: keyframes not defined.")
                return
            keyframes = self.keyframes
        else:
            self.keyframes = keyframes

        img_fixed_blur = self.smooth_frame(self.data_array, keyframes[0])[1]
        img_fixed_blur_0 = img_fixed_blur

        self.keyframe_transforms = []
        self.keyframe_data_reg = [img_fixed_blur]


        if self.debug:
            itk.imwrite(itk.GetImageFromArray(self.data_array[keyframes[0]]),
                        str(keyframes[0]) + "_fxd.mha")
            itk.imwrite(img_fixed_blur,
                        str(keyframes[0]) + "_fxd_blur.mha")

        for i in range(len(keyframes) - 1):
            print(
                f"Registering set {i+1} of {len(keyframes)-1}: Frame = {keyframes[i+1]}"
            )

            img_moving_blur = self.smooth_frame(self.data_array,
                                                keyframes[i + 1])[1]

            if self.debug:
                itk.imwrite(img_moving_blur,
                            str(keyframes[i + 1]) + "_org_blur.mha")

            Reg = tube.RegisterImages[itk.Image[itk.F, 3]].New()
            Reg.SetReportProgress(self.debug)
            Reg.SetMetric(self.registration_metric)
            Reg.SetExpectedOffsetMagnitude(self.expected_offset)
            Reg.SetExpectedRotationMagnitude(self.expected_rotation)
            Reg.SetRigidSamplingRatio(0.2)
            Reg.SetRigidMaxIterations(2000)
            Reg.SetExpectedScaleMagnitude(0.02)
            Reg.SetExpectedSkewMagnitude(0.01)
            Reg.SetAffineSamplingRatio(0.2)
            Reg.SetAffineMaxIterations(500)
            Reg.SetAffineTargetError(0.0000001)

            Reg.SetFixedImage(img_fixed_blur)
            Reg.SetMovingImage(img_moving_blur)

            Reg.SetRegistration("PIPELINE_AFFINE")
            
            if i > 0 and not self.register_to_prior_frame and self.smooth_registrations:
                Reg.SetLoadedMatrixTransform(self.keyframe_transforms[-1])
                Reg.SetEnableLoadedRegistration(False)
                Reg.SetInitialMethodEnum("INIT_WITH_LOADED_TRANSFORM")
            else:
                Reg.SetInitialMethodEnum("INIT_WITH_NONE")
                
            Reg.SetEnableInitialRegistration(True)
                
            Reg.SetRigidMaxIterations(250)
            Reg.SetAffineMaxIterations(250)

            Reg.Update()

            if np.isnan(Reg.GetCurrentMatrixTransform().GetParameters()[0]):
                print("ERROR: Registration failed!!")
                print("   Attempting re-registration")
                Reg.SetInitialMethodEnum("INIT_WITH_NONE")
                Reg.SetEnableInitialRegistration(True)
                Reg.SetRigidMaxIterations(250)
                Reg.SetAffineMaxIterations(250)
                Reg.Update()
            if np.isnan(Reg.GetCurrentMatrixTransform().GetParameters()[0]):
                print("ERROR: Registration failed twice!!")
                print("    Reverting to identity transform")
                Reg.SetEnableInitialRegistration(True)
                Reg.SetRigidMaxIterations(0)
                Reg.SetAffineMaxIterations(0)
                Reg.Update()
                Reg.SetRigidMaxIterations(250)
                Reg.SetAffineMaxIterations(250)

            if self.register_to_prior_frame:
                img_fixed_blur = Reg.GetFinalMovingImage()
            
            if self.register_to_zero_frame and self.register_to_prior_frame and i>0:
                trns = Reg.GetCurrentMatrixTransform()
                trns.Compose(self.keyframe_transforms[-1], True)
                
                RegZero = tube.RegisterImages[itk.Image[itk.F, 3]].New()
                RegZero.SetReportProgress(self.debug)
                RegZero.SetMetric(self.registration_metric)
                RegZero.SetExpectedOffsetMagnitude(self.expected_offset)
                RegZero.SetExpectedRotationMagnitude(self.expected_rotation)
                RegZero.SetRigidSamplingRatio(0.2)
                RegZero.SetRigidMaxIterations(2000)
                RegZero.SetExpectedScaleMagnitude(0.02)
                RegZero.SetExpectedSkewMagnitude(0.01)
                RegZero.SetAffineSamplingRatio(0.2)
                RegZero.SetAffineMaxIterations(500)
                RegZero.SetAffineTargetError(0.0000001)
    
                RegZero.SetFixedImage(img_fixed_blur_0)
                RegZero.SetMovingImage(img_moving_blur)
            
                RegZero.SetLoadedMatrixTransform(trns)
                RegZero.SetEnableLoadedRegistration(False)
                RegZero.SetInitialMethodEnum("INIT_WITH_LOADED_TRANSFORM")
                RegZero.SetEnableInitialRegistration(True)
                RegZero.Update()
                if np.isnan(Reg.ZeroGetCurrentMatrixTransform().GetParameters()[0]):
                    print("ERROR: Registration failed!!")
                    print("   Attempting re-registration")
                    RegZero.SetInitialMethodEnum("INIT_WITH_NONE")
                    RegZero.SetEnableInitialRegistration(True)
                    RegZero.SetRigidMaxIterations(250)
                    RegZero.SetAffineMaxIterations(250)
                    RegZero.Update()
                if np.isnan(RegZero.GetCurrentMatrixTransform().GetParameters()[0]):
                    print("ERROR: Registration failed twice!!")
                    print("    Reverting to identity transform")
                    RegZero.SetEnableInitialRegistration(True)
                    RegZero.SetRigidMaxIterations(0)
                    RegZero.SetAffineMaxIterations(0)
                    RegZero.Update()
                    RegZero.SetRigidMaxIterations(250)
                    RegZero.SetAffineMaxIterations(250)
                self.keyframe_transforms.append(RegZero.GetCurrentMatrixTransform())
                if self.debug:
                    itk.imwrite(Reg.GetFinalMovingImage(), str(keyframes[i + 1]) + "_reg_blur.mha")
            else:
                self.keyframe_transforms.append(Reg.GetCurrentMatrixTransform())
                if self.debug:
                    itk.imwrite(Reg.GetFinalMovingImage(), str(keyframes[i + 1]) + "_reg_blur.mha")

        print("Done!")

    def smooth_keyframe_transform_parameters(self, i):
        transform = itk.ComposeScaleSkewVersor3DTransform[itk.D].New()
        num_params = transform.GetNumberOfParameters()
        
        params = np.zeros(num_params)
        if self.smooth_registrations and self.keyframe_transform_smoothing_window_size>0:
            window_min = max(-1, i-self.keyframe_transform_smoothing_window_size)
            window_max = min(len(self.keyframe_transforms), i+self.keyframe_transform_smoothing_window_size+1)
            params_weight = 0
            denom = 2 * ((self.keyframe_transform_smoothing_window_size+1)/2)**2
            for w in range(window_min,window_max):
                transform = itk.ComposeScaleSkewVersor3DTransform[itk.D].New()
                if w == -1:
                    transform.SetIdentity()
                else:
                    transform.SetCenter(self.keyframe_transforms[w].GetCenter())
                    transform.SetMatrix(self.keyframe_transforms[w].GetMatrix())
                    transform.SetOffset(self.keyframe_transforms[w].GetOffset())
                p = transform.GetParameters()
                weight = np.exp(-(w-i)**2/denom)
                print(i, weight)
                params += [p[x]*weight for x in range(num_params)]
                params_weight += weight
            params /= params_weight
        else:
            transform = itk.ComposeScaleSkewVersor3DTransform[itk.D].New()
            transform.SetCenter(self.keyframe_transforms[i].GetCenter())
            transform.SetMatrix(self.keyframe_transforms[i].GetMatrix())
            transform.SetOffset(self.keyframe_transforms[i].GetOffset())
            p = transform.GetParameters()
            params = [p[x] for x in range(num_params)]
        
        return params
        
    def interpolate_keyframe_transforms(self):  
        transform = itk.ComposeScaleSkewVersor3DTransform[itk.D].New()
        num_params = transform.GetNumberOfParameters()
        transform.SetIdentity()
        start_params = np.array(transform.GetParameters())
        self.transforms = []
        num_keyframes = len(self.keyframe_transforms)
        for i in range(num_keyframes):
            start_frame = self.keyframes[i]
            end_frame = self.keyframes[i + 1]
            step_frame = 1.0 / (end_frame - start_frame)
            end_params = self.smooth_keyframe_transform_parameters(i)
            for interp_frame in range(start_frame, end_frame):
                portion = (interp_frame - start_frame) / (end_frame - start_frame)
                new_transform = itk.ComposeScaleSkewVersor3DTransform[itk.D].New()
                new_transform.SetFixedParameters( self.keyframe_transforms[i].GetFixedParameters() )
                new_params = new_transform.GetParameters()
                for x in range(num_params):
                    new_params[x] = start_params[x] + portion * (end_params[x] - start_params[x])
                new_transform.SetParameters(new_params)
                self.transforms.append(new_transform)
            start_params = end_params
        new_transform = itk.ComposeScaleSkewVersor3DTransform[itk.D].New()
        new_transform.SetFixedParameters( self.keyframe_transforms[num_keyframes-1].GetFixedParameters() )
        new_params = new_transform.GetParameters()
        end_params = self.smooth_keyframe_transform_parameters(num_keyframes-1)
        for x in range(num_params):
            new_params[x] = end_params[x]
        new_transform.SetParameters(new_params)
        self.transforms.append(new_transform)

    def apply_transforms(self):
        self.data_array_reg = np.zeros(self.data_array.shape)
        self.data_array_reg[0] = self.data_array[0].astype(np.float32)
        match_image = itk.GetImageFromArray(self.data_array[0].astype(np.float32))
        kfc = 1
        for i in range(1,len(self.data_array)):
            img = itk.GetImageFromArray(self.data_array[i].astype(np.float32))
            trns = itk.AffineTransform[itk.D, 3].New()
            trns.SetIdentity()
            trns.SetCenter(self.transforms[i].GetCenter())
            trns.SetMatrix(self.transforms[i].GetMatrix())
            trns.SetOffset(self.transforms[i].GetOffset())
            Res = tube.ResampleImage[itk.Image[itk.F, 3]].New()
            Res.SetInput(img)
            Res.SetMatchImage(match_image)
            Res.SetTransform(trns)
            Res.SetLoadTransform(True)
            Res.Update()
            self.data_array_reg[i] = itk.GetArrayFromImage(Res.GetOutput())

            if self.debug and kfc<len(self.keyframes) and i == self.keyframes[kfc]:
                itk.imwrite(Res.GetOutput(), str(i+1)+"_reg_resample.mha")
                kfc = kfc + 1

    def save_transforms(self, filename):
        transform = itk.ComposeScaleSkewVersor3DTransform[itk.D].New()
        num_params = transform.GetNumberOfParameters()
        if num_params != 12:
            print("ERROR: Expecting 12 parameters per transform")
            return
        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "VersorX",
                    "VersorY",
                    "VersorZ",
                    "OffsetX",
                    "OffsetY",
                    "OffsetZ",
                    "ScaleX",
                    "ScaleY",
                    "ScaleZ",
                    "SkewXY",
                    "SkewXZ",
                    "SkewYZ",
                    "CenterX",
                    "CenterY",
                    "CenterZ",
                ]
            )
            for trns in self.transforms:
                p = trns.GetParameters()
                params = [p[x] for x in range(num_params)]
                params.append(trns.GetCenter()[0])
                params.append(trns.GetCenter()[1])
                params.append(trns.GetCenter()[2])
                writer.writerow(params)

    def save_matrix_transforms(self, filename):
        transform = itk.ComposeScaleSkewVersor3DTransform[itk.D].New()
        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Matrix00",
                    "Matrix01",
                    "Matrix02",
                    "Matrix10",
                    "Matrix11",
                    "Matrix12",
                    "Matrix20",
                    "Matrix21",
                    "Matrix22",
                    "OffsetX",
                    "OffsetY",
                    "OffsetZ",
                ]
            )
            for trns in self.transforms:
                m = trns.GetMatrix()
                o = trns.GetOffset()
                params = [m(x, y) for x in range(3) for y in range(3)]
                params.append(o[0])
                params.append(o[1])
                params.append(o[2])
                writer.writerow(params)

    def get_transforms(self):
        transform = itk.ComposeScaleSkewVersor3DTransform[itk.D].New()
        transform_list = np.zeros([len(self.transforms), 12])
        for i, trns in enumerate(self.transforms):
            m = trns.GetMatrix()
            o = trns.GetOffset()
            params = [m(x, y) for x in range(3) for y in range(3)]
            params.append(o[0])
            params.append(o[1])
            params.append(o[2])
            transform_list[i] = params
        return transform_list
