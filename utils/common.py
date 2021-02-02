from functools import wraps
import time
import os


def set_num_threads(nt):
    nt = str(nt)
    os.environ["OMP_NUM_THREADS"] = nt         # export OMP_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"] = nt    # export OPENBLAS_NUM_THREADS=1
    os.environ["MKL_NUM_THREADS"] = nt         # export MKL_NUM_THREADS=1
    os.environ["VECLIB_MAXIMUM_THREADS"] = nt  # export VECLIB_MAXIMUM_THREADS=1
    os.environ["NUMEXPR_NUM_THREADS"] = nt     # export NUMEXPR_NUM_THREADS=1

def convert(seconds): 
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print (f'@timefn: {fn.__name__} took {convert(t2-t1)} (hh:mm:ss)')
        return result
    return measure_time
