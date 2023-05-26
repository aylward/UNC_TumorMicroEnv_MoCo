import sys
import time
import argparse
import traceback

from mocoreg import mocoreg

def prepare_argparser():
    parser = argparse.ArgumentParser(description='Motion Correction via Registration')
    parser.add_argument('input_filename',
                        help='Name of the 4D BMode Matlab file to be analyzed.')
    parser.add_argument('transforms_filename',
                        help='Name of the file to store registration transforms.')
    parser.add_argument('-b', '--bits', default=32, type=int,
                        help='Input file bits per pixel [32|64] (default=32)')
    parser.add_argument('-f', '--frames', default=200, type=int,
                        help='Number of frames (default=200)')
    parser.add_argument('-l', '--laterals', default=92, type=int,
                        help='Number of laterals (default=92)')
    parser.add_argument('-d', '--depths', default=153, type=int,
                        help='Number of depths (default=152)')
    parser.add_argument('-e', '--elevations', default=102, type=int,
                        help='Number of elevations (default=102)')
    parser.add_argument('-s', '--scale', default=3.0, type=float,
                        help='Scale applied to data (default=3.0)')
    parser.add_argument('-t', '--tolerance', default=1.0, type=float,
                        help='How selective is keyframe selection.'
                             'Larger = less selective = more keyframes'
                              '(default=1.0)')
    parser.add_argument('-z', '--zero', action='store_true',
                        help='Register all to the zero-th frame')
    parser.add_argument('-r', '--results_filename',
                        help='Save registered bmode data to this file')
    parser.add_argument('-D', '--Debug', action='store_true',
                        help='Enable debugging.')
    return parser

def main(args):
    try:
        app = mocoreg(register_to_frame_zero=args.zero, debug=args.Debug)
        
        app.read_4d_bmode_matlab_file(args.input_filename,
                                      nlateral=args.laterals,
                                      nframes=args.frames,
                                      ndepth=args.depths,
                                      nelevation=args.elevations,
                                      bits=args.bits)
        app.compute_keyframes(tolerance=args.tolerance)
        if args.Debug:
            print("Keyframes =", app.keyframes)
            
        start_time = time.perf_counter()
        app.register_keyframes()
        end_time = time.perf_counter()
        if args.Debug:
            print("Registration time =", end_time-start_time)
            
        app.interpolate_keyframe_transforms()
        
        if args.Debug:
            meansq,diff = app.compute_inter_keyframe_diffs()
            meansq_reg,diff_reg = app.compute_inter_keyframe_diffs(data_array=app.data_array_reg)
            print("Before registration MSE =", meansq)
            print("After registration MSE =", meansq_reg)
        
        app.save_matrix_transforms(args.transforms_filename)
        
        if args.results_filename != None:
            app.apply_transforms()
            itk.imwrite(itk.GetImageFromArray(app.data_array_reg.astype(np.float32)),
                        args.results_filename, compression=True)
            
        sys.exit()
    
    except Exception as e:
        traceback.print_exc()
        sys.exit("ERROR")

if __name__ == '__main__':
    parser = prepare_argparser()
    args = parser.parse_args()
    
    main(args)
