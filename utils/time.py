def func_time(f, *args):
    import time
    tic = time.time()
    ret = f(*args)
    toc = time.time()
    return toc - tic, ret
