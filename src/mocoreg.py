import numpy as np

import math

import itk
from itk import TubeTK as tube

import csv


class mocoreg:
    def __init__(self, register_to_frame_zero=True, debug=False):
        self.debug = debug

        # Pixel units
        self.feature_size = 3.0

        self.register_to_frame_zero = register_to_frame_zero
        # self.registration_metric = "MATTES_MI_METRIC"
        self.registration_metric = "MEAN_SQUARED_ERROR_METRIC"

        self.frames_per_second = 200
        self.seconds_per_change = 0.333
        
        # Derived scales
        self.max_keyframe_interval = int( self.frames_per_second * self.seconds_per_change / 2 )
        self.keyframe_search_stepsize = max(5, int(self.max_keyframe_interval * 0.05))

        self.data_array = []
        self.keyframes = []
        self.keyframe_transforms = []

        self.data_array_reg = []

    def read_4d_bmode_matlab_file(
        self, filename, nlateral=92, nframes=200, ndepth=153, nelevation=102, bits=32
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

        pixel_type.newbyteorder("<")
        data_1d = np.frombuffer(file_content, dtype=pixel_type)
        data_raw = np.reshape(
            data_1d, [nlateral, nframes, ndepth, nelevation], order="C"
        )
        self.data_array = np.transpose(data_raw, (1, 3, 0, 2))

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
            diff_img, sigma=(self.frames_per_second * self.seconds_per_change / 2 / self.keyframe_search_stepsize)
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

    def smooth_frame(self, data, frame_num, half_window=1):
        t_min = int(max(frame_num - half_window, 0))
        t_max = int(min(frame_num + half_window+1, data.shape[0]))

        tmp_data = np.average(data[t_min:t_max], axis=0)

        img_data = itk.GetImageFromArray(tmp_data.astype(np.float32))
        img_data_median = itk.MedianImageFilter(
            img_data, radius=max(1, int(self.feature_size))
        )
        img_data_blur = itk.SmoothingRecursiveGaussianImageFilter(
            img_data_median, sigma=self.feature_size
        )
        tmp_data_blur = itk.GetArrayFromImage(img_data_blur)

        return tmp_data_blur, img_data_blur

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
        ref_frame =  data_array[keyframes[0]]
        diffs = []
        diff_rmse = 0
        diff_count = len(keyframes)
        for i in range(1,diff_count):
            frame =  data_array[keyframes[i]]
            diff = np.average( np.square( ref_frame - frame ) )
            diffs.append(math.sqrt(diff))
            diff_rmse += diff
            if self.register_to_frame_zero != False:
                # Do the opposite of registration: 
                #   * compare prior frame, if all registered to zero
                #   * compare to zero frame, if all registered to prior
                ref_frame = frame

        diff_rmse /= diff_count-1
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

        img_fixed_blur = self.smooth_frame(self.data_array, keyframes[0])[1]

        self.keyframe_transforms = []
        self.keyframe_data_reg = [img_fixed_blur]

        Reg = tube.RegisterImages[itk.Image[itk.F, 3]].New()
        
        Reg.SetReportProgress(self.debug)
        Reg.SetMetric(self.registration_metric)

        Reg.SetExpectedOffsetMagnitude(2)
        Reg.SetExpectedRotationMagnitude(0.01)
        Reg.SetRigidSamplingRatio(0.2)
        Reg.SetRigidMaxIterations(2000)

        Reg.SetExpectedScaleMagnitude(0.01)
        Reg.SetExpectedSkewMagnitude(0.001)
        Reg.SetAffineSamplingRatio(0.2)
        Reg.SetAffineTargetError(0.0000001)

        for i in range(len(keyframes) - 1):
            print(f"Registering set {i+1} of {len(keyframes)-1}: Frame = {keyframes[i+1]}")

            img_moving_blur = self.smooth_frame(self.data_array, keyframes[i + 1])[1]
            
            #itk.imwrite(img_moving_blur, str(keyframes[i+1])+".mha")

            Reg.SetFixedImage(img_fixed_blur)
            Reg.SetMovingImage(img_moving_blur)
            
            if i == 0:
                Reg.SetRegistration("PIPELINE_AFFINE")
                Reg.SetInitialMethodEnum("INIT_WITH_IMAGE_CENTERS")
                Reg.SetAffineMaxIterations(2000)
            else:
                Reg.SetRegistration("AFFINE")
                Reg.SetLoadedMatrixTransform(self.keyframe_transforms[-1])
                Reg.SetEnableLoadedRegistration(False)
                Reg.SetInitialMethodEnum("INIT_WITH_LOADED_TRANSFORM")
                Reg.SetEnableInitialRegistration(True)
                Reg.SetAffineMaxIterations(250)
            Reg.Update()
            
            if self.register_to_frame_zero == False:
                img_fixed_blur = img_moving_blur

            self.keyframe_transforms.append(Reg.GetCurrentMatrixTransform())
        print("Done!")

    def interpolate_keyframe_transforms(self, keyframes=[]):
        if len(keyframes) == 0:
            keyframes = self.keyframes
        if len(keyframes) == 0:
            self.compute_keyframes()
            if len(self.keyframes) == 0:
                print("ERROR: keyframes not defined.")
                return
            keyframes = self.keyframes

        transform = itk.ComposeScaleSkewVersor3DTransform[itk.D].New()
        num_params = transform.GetNumberOfParameters()
        p = transform.GetParameters()
        params = [p(x) for x in range(num_params)]
        self.transforms = []
        for i in range(len(keyframes) - 1):
            start_frame = keyframes[i]
            end_frame = keyframes[i + 1]
            step_frame = 1.0 / (end_frame - start_frame)
            transform = itk.ComposeScaleSkewVersor3DTransform[itk.D].New()
            transform.SetCenter(self.keyframe_transforms[i].GetCenter())
            transform.SetMatrix(self.keyframe_transforms[i].GetMatrix())
            transform.SetOffset(self.keyframe_transforms[i].GetOffset())
            p = transform.GetParameters()
            end_params = [p[x] for x in range(num_params)]
            for interp_frame in range(start_frame + 1, end_frame + 1):
                portion = (interp_frame - start_frame) / (end_frame - start_frame)
                for x in range(num_params):
                    p[x] = params[x] + portion * (end_params[x] - params[x])
                new_transform = itk.ComposeScaleSkewVersor3DTransform[itk.D].New()
                new_transform.SetFixedParameters(self.keyframe_transforms[i].GetFixedParameters())
                new_transform.SetParameters(p)
                self.transforms.append(new_transform)

            params = end_params

    def apply_transforms(self):
        self.data_array_reg = np.zeros(self.data_array.shape)
        self.data_array_reg[0] = self.data_array[0].astype(np.float32)
        match_image = itk.GetImageFromArray(self.data_array[0].astype(np.float32))
        #kfc = 0
        for i in range(len(self.data_array) - 1):
            img = itk.GetImageFromArray(self.data_array[i + 1].astype(np.float32))
            trns = itk.AffineTransform[itk.D, 3].New()
            trns.SetCenter(self.transforms[i].GetCenter())
            trns.SetMatrix(self.transforms[i].GetMatrix())
            trns.SetOffset(self.transforms[i].GetOffset())
            Res = tube.ResampleImage[itk.Image[itk.F, 3]].New()
            Res.SetInput(img)
            Res.SetMatchImage(match_image)
            Res.SetTransform(trns)
            Res.Update()
            self.data_array_reg[i + 1] = itk.GetArrayFromImage(Res.GetOutput())
            
            #if i == self.keyframes[kfc]:
                #itk.imwrite(Res.GetOutput(), str(i)+"_reg.mha")
                #kfc = kfc + 1


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
                params = [m(x,y) for x in range(3) for y in range(3)]
                params.append(o[0])
                params.append(o[1])
                params.append(o[2])
                writer.writerow(params)

    def get_transforms(self):
        transform = itk.ComposeScaleSkewVersor3DTransform[itk.D].New()
        transform_list = np.zeros([len(self.transforms), 12])
        for i,trns in enumerate(self.transforms):
            m = trns.GetMatrix()
            o = trns.GetOffset()
            params = [m(x,y) for x in range(3) for y in range(3)]
            params.append(o[0])
            params.append(o[1])
            params.append(o[2])
            transform_list[i] = params
        return transform_list
