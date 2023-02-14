import numpy as np

import math

import itk
from itk import TubeTK as tube

import csv

class mocoreg():
    
    def __init__(self):
        self.debug = False
        
        self.feature_size = 2.0
        
        self.registration_metric = "MATTES_MI_METRIC"
        #self.registration_metric = "MEAN_SQUARED_ERROR_METRIC"
        
        self.keyframe_portion = 1.0
        
        self.frames_per_second = 10
        self.max_keyframe_interval = 45
        self.keyframe_search_stepsize = 3
        
        self.data_array = []
        self.keyframes = []
        self.keyframe_transforms = []
        
        self.data_array_reg = []
        
    def read_4d_bmode_matlab_file(self, filename, 
                                  nlateral=92, nframes=200, ndepth=153, nelevation=102):
        with open(filename, mode='rb') as file:
            file_content = file.read()
        pixel_type = np.dtype(np.float64)
        pixel_type.newbyteorder('<')
        data_1d = np.frombuffer(file_content, dtype=pixel_type)
        data_raw = np.reshape(data_1d, [nlateral, nframes, ndepth, nelevation], order='C')
        self.data_array = np.transpose(data_raw, (1,3,0,2))
        
    def compute_keyframes(self, data_array=[]):
        if len(data_array) == 0:
            data_array = self.data_array
            if len(data_array) == 0:
                print("ERROR: No data found for computing key frames")
                return
        
        diff = np.zeros([data_array.shape[0]//self.keyframe_search_stepsize,
                        self.max_keyframe_interval//self.keyframe_search_stepsize])
        for frame_num in range(diff.shape[0]):
            for win_num in range(diff.shape[1]):
                win_min = frame_num * self.keyframe_search_stepsize
                win_max = min(win_min + win_num*self.keyframe_search_stepsize, data_array.shape[0]-1)
                tmp_diff = np.sum(np.abs(data_array[win_min,:,:,:] -
                                         data_array[win_max,:,:,:]))
                diff[frame_num, win_num] = tmp_diff
                    
        diff_img = itk.GetImageFromArray(diff)
        diff_img_blur = itk.SmoothingRecursiveGaussianImageFilter(diff_img,
                            sigma=(self.frames_per_second/self.keyframe_search_stepsize)/2)
        diff_blur = itk.GetArrayFromImage(diff_img_blur)
        
        diff_avg = np.average(diff_blur) * self.keyframe_portion
        diff_thresh = np.where(diff_blur<=diff_avg, 1, 0)
        win_limit = np.argmin(diff_thresh, axis=1) * self.keyframe_search_stepsize
        win_limit = np.where(win_limit==0, self.max_keyframe_interval-1, win_limit)
        
        self.keyframes = []
        keyframe = 0
        self.keyframes = [0]
        while keyframe<data_array.shape[0]-1:
            keyframe += win_limit[keyframe//self.keyframe_search_stepsize]
            keyframe = min(keyframe, data_array.shape[0]-1)
            self.keyframes.append(keyframe)
    
    def smooth_frame(self, data, frame_num):
        window = int(self.frames_per_second / 2)
        t_min = int(max(frame_num-window,0))
        t_max = int(min(frame_num+window,data.shape[0]))

        tmp_data = np.average(data[t_min:t_max],axis=0)

        img_data = itk.GetImageFromArray(tmp_data.astype(np.float32))
        img_data_blur = itk.SmoothingRecursiveGaussianImageFilter(img_data,sigma=self.feature_size)
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
                self.compute_keyframes()
                if len(self.keyframes) == 0:
                    print("ERROR: keyframes not defined")
                    return
                keyframes = self.keyframes
        diffs = []
        diff_meansq = 0
        diff_count = len(keyframes)-1
        for i in range(diff_count):
            diff = math.sqrt(np.sum(np.square(self.smooth_frame(data_array, keyframes[i])[0] -
                                              self.smooth_frame(data_array, keyframes[i+1])[0])))
            diffs.append(diff)
            diff_meansq += diff

        # Include the diff between the first and last frames
        diff = math.sqrt(np.sum(np.square(self.smooth_frame(data_array, keyframes[-1])[0] -
                                          self.smooth_frame(data_array, keyframes[0])[0])))
        diffs.append(diff)
        diff_meansq += diff
        
        diff_meansq /= diff_count
   
        return diff_meansq, diffs
            
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
        
        for i in range(len(keyframes)-1):
            print(f"Registering set {i} of {len(keyframes)-1}")
            
            img_moving_blur = self.smooth_frame(self.data_array, keyframes[i+1])[1]
            
            Reg = tube.RegisterImages[itk.Image[itk.F,3]].New()
            Reg.SetFixedImage(img_fixed_blur)
            Reg.SetMovingImage(img_moving_blur)
            Reg.SetReportProgress(self.debug)
            Reg.SetRegistration("PIPELINE_AFFINE")
            Reg.SetMetric(self.registration_metric)
            
            Reg.SetInitialMethodEnum("INIT_WITH_IMAGE_CENTERS")

            Reg.SetExpectedOffsetMagnitude(2)
            Reg.SetExpectedRotationMagnitude(0.01)
            Reg.SetRigidSamplingRatio(0.2)
            Reg.SetRigidMaxIterations(4000)

            Reg.SetExpectedScaleMagnitude(0.01)
            Reg.SetExpectedSkewMagnitude(0.001)
            Reg.SetAffineSamplingRatio(0.2)
            Reg.SetAffineMaxIterations(2000)

            Reg.Update()
            
            img_fixed_blur = img_moving_blur
            
            self.keyframe_transforms.append(Reg.GetCurrentMatrixTransform())
            
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
        params = [ p(x) for x in range(num_params) ]
        self.transforms = []
        for i in range(len(keyframes)-1):
            start_frame = keyframes[i]
            end_frame = keyframes[i+1]
            print(f"Interpolating frames {start_frame} to {end_frame}")
            step_frame = 1.0 / (end_frame - start_frame)
            transform = itk.ComposeScaleSkewVersor3DTransform[itk.D].New()
            transform.SetCenter( self.keyframe_transforms[i].GetCenter() )
            transform.SetMatrix( self.keyframe_transforms[i].GetMatrix() )
            transform.SetOffset( self.keyframe_transforms[i].GetOffset() )
            p = transform.GetParameters()
            end_params = [ p[x] for x in range(num_params) ]
            for interp_frame in range(start_frame+1, end_frame+1):
                portion = (interp_frame - start_frame) / (end_frame - start_frame)
                for x in range(num_params):
                    p[x] = params[x] + portion * ( end_params[x] - params[x] )
                new_transform = itk.ComposeScaleSkewVersor3DTransform[itk.D].New()
                new_transform.SetCenter( self.keyframe_transforms[i].GetCenter() )
                new_transform.SetParameters( p )
                self.transforms.append(new_transform)
            params = end_params
            
    def apply_interpolations(self):
        self.data_array_reg = np.zeros(self.data_array.shape)
        self.data_array_reg[0] = self.data_array[0].astype(np.float32)
        match_image = itk.GetImageFromArray(self.data_array[0].astype(np.float32))
        for i in range(len(self.data_array)-1):
            img = itk.GetImageFromArray(self.data_array[i+1].astype(np.float32))
            trns = itk.AffineTransform[itk.D, 3].New()
            trns.SetMatrix(self.transforms[i].GetMatrix())
            trns.SetOffset(self.transforms[i].GetOffset())
            Res = tube.ResampleImage[itk.Image[itk.F,3]].New()
            Res.SetInput(img)
            Res.SetMatchImage(match_image)
            Res.SetTransform(self.transforms[i])
            Res.Update()
            self.data_array_reg[i+1] = itk.GetArrayFromImage(Res.GetOutput())
            
    def save_transforms(self, filename):
        transform = itk.ComposeScaleSkewVersor3DTransform[itk.D].New()
        num_params = transform.GetNumberOfParameters()
        if num_params != 12:
            print("ERROR: Expecting 12 parameters per transform")
            return
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["VersorX", "VersorY", "VersorZ",
                             "OffsetX", "OffsetY", "OffsetZ",
                             "ScaleX", "ScaleY", "ScaleZ",
                             "SkewXY", "SkewXZ", "SkewYZ",
                             "CenterX", "CenterY", "CenterZ"])
            for trns in self.transforms:
                p = trns.GetParameters()
                params = [ p[x] for x in range(num_params) ]
                params.append(trns.GetCenter()[0])
                params.append(trns.GetCenter()[1])
                params.append(trns.GetCenter()[2])
                writer.writerow(params)