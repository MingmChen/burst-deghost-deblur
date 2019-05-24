import time


def time_log(func):
    def warp_func(*args, **kwargs):
        old = time.time()
        output = func(*args, **kwargs)
        print('%s execute in in %.4fs' % (func.__name__, time.time()-old))
        return output
    return warp_func
