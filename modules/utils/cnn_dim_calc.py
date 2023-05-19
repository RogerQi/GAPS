from math import floor

def conv_output_shape(h, w, kernel_size = 1, stride = 1, pad = 0, dilation = 1):
    h = floor( ((h + (2 * pad) - ( dilation * (kernel_size - 1) ) - 1 )/ stride) + 1)
    w = floor( ((w + (2 * pad) - ( dilation * (kernel_size - 1) ) - 1 )/ stride) + 1)
    return h, w