import psutil
import os

def get_memory(verbose=False):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    if verbose:
        print("Memory usage: {:.2f} MB".format(info.uss / 1024. / 1024.))
    return info.uss / 1024. / 1024. # in MB