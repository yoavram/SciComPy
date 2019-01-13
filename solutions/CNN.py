def myconvolve(x, y):
    assert x.ndim == y.ndim == 1
    m = y.size // 2
    x = np.pad(x, m, mode='constant', constant_values=0)
    return [ 
        (x[i-m:i+m+1] * y).sum() 
        for i in range(m, x.size - m) 
    ]


def horizontal_gradient(im):
    kernel = np.zeros((3, 3))
    kernel[:, 0] = -1
    kernel[:, 2] = 1
    return convolve(im, kernel)    