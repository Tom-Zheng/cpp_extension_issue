import os
import paddle
import numpy as np
from custom_add_layer import custom_add_static

input_shape = [8, 16]

feed_dict = {
    "input1": np.random.random(input_shape).astype(np.float32),
    "input2": np.random.random(input_shape).astype(np.float32),
}

def build_program(main_program, startup_program):
    with paddle.static.program_guard(main_program, startup_program):
        with paddle.utils.unique_name.guard():
            x1 = paddle.static.data(
                name="input1",
                shape=input_shape,
                dtype='float32',
            )
            x2 = paddle.static.data(
                name="input2",
                shape=input_shape,
                dtype='float32',
            )
            y = custom_add_static(x1, x2)
    return [y]

def cal_output(exe):
    main_prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    fetch_list = build_program(main_prog, startup_prog)
    scope = paddle.static.Scope()
    with paddle.static.scope_guard(scope):
        exe.run(startup_prog)
        results = exe.run(
            main_prog, feed=feed_dict, fetch_list=fetch_list
        )
    return results

def test_main():
    place = paddle.CUDAPlace(0)
    exe = paddle.static.Executor(place)
    results = cal_output(exe)
    print(results)

if __name__ == '__main__':
    paddle.enable_static()
    test_main()
