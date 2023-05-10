import paddle
from paddle.fluid.layer_helper import LayerHelper
import custom_cpp_extension as ext

def _custom_add_forward(x1, x2):
    return ext.custom_add(x1, x2)

def custom_add_static(x1, x2):
    helper = LayerHelper("custom_add_op", **locals())
    y_var = helper.create_variable(dtype='float32', shape=x1.shape)
    paddle.static.nn.py_func(func=_custom_add_forward,
        x=[x1, x2],
        out=y_var,
        backward_func=None)
    return y_var