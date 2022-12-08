import psutil
import os
from functools import wraps
import time

def tic_toc(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = time.time()
        result = func(*args, **kwargs)
        toc = time.time()
        print("Function {} costs : {:.2f} sec".format(func.__name__, toc - tic))
        return result
    return wrapper


def get_memory(verbose=False):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    if verbose:
        print("Memory usage: {:.2f} MB".format(info.uss / 1024. / 1024.))
    return info.uss / 1024. / 1024. # in MB



if __name__ == "__main__":
    @tic_toc
    def test():
        a = []
        for i in range(1000000):
            a.append(i)
        time.sleep(1)
        get_memory(verbose=True)
    test()