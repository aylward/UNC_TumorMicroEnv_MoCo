import sys
import time
import argparse
import traceback

from mocoreg import mocoreg


def mocoreg_func(
    input_data,
    scale=3.0,
    tolerance=1.0,
    frames_per_second=200,
    seconds_per_change=0.333,
    reg_to_zero=True,
    debug=False,
):
    transform_list = []

    app = mocoreg(register_to_frame_zero=reg_to_zero, debug=debug)
    app.feature_size = scale
    app.frames_per_second = frames_per_second
    app.seconds_per_change = seconds_per_change

    app.import_4d_bmode_matlab_data(input_data)

    app.compute_keyframes(tolerance=tolerance)
    if debug:
        print("Keyframes =", app.keyframes)

    start_time = time.perf_counter()
    app.register_keyframes()
    end_time = time.perf_counter()
    if debug:
        print("Registration time =", end_time - start_time)

    app.interpolate_keyframe_transforms()

    if debug:
        meansq, diff = app.compute_inter_keyframe_diffs()
        app.apply_transforms()
        meansq_reg, diff_reg = app.compute_inter_keyframe_diffs(
            data_array=app.data_array_reg
        )
        print("Before registration MSE =", meansq)
        print("After registration MSE =", meansq_reg)

    transform_list = app.get_transforms()

    return transform_list
