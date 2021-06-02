import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable


def x_log2x_builtin(x: float) -> float:
    return x * math.log2(x) if x else x


def error_check(test_func: Callable[[float], float],
                exact_func: Callable[[float], float], start: float,
                stop: float, samples: int, plot: bool) -> None:
    test_floats = []
    exact_results = []
    approx_results = []
    approx_errors = []

    print(f"Test Func: {test_func.__name__}")
    print(f"Exact Func: {exact_func.__name__}")

    for i in range(samples):
        print(f'samples: {i}', end='\r')
        test_float = random.uniform(start, stop)
        test_floats.append(test_float)

        exact = exact_func(test_float)
        exact_results.append(exact)

        approx = test_func(test_float)
        if plot:
            approx_results.append(approx)
        approx_errors.append(abs(exact - approx))
    print(f'samples: {samples}')

    print(f"""
    Error Stats:
    avg: {sum(approx_errors)/samples}
    min: {min(approx_errors)}
    max: {max(approx_errors)}
    median: {approx_errors[int(len(approx_errors)/2)]}
    """)

    if plot:
        df = pd.DataFrame({
            'floats': test_floats,
            'exact': exact_results,
            'approx': approx_results,
        })

        df.sort_values(by=['floats'], inplace=True)
        df.reset_index(inplace=True, drop=True)
        df[['exact', 'approx']].plot()
        plt.show()

    return None


# docker run --init lolremez --double --progress -d 12 -r "0.5:1" "log2(x)"
def log_approx2(x):
    u = -5.5151853441976255
    u = u * x + 5.3909424236676342e+1
    u = u * x + -2.423916095038938e+2
    u = u * x + 6.6402993163788989e+2
    u = u * x + -1.2372809889413113e+3
    u = u * x + 1.6573540652948472e+3
    u = u * x + -1.6444215388143739e+3
    u = u * x + 1.2268839247022905e+3
    u = u * x + -6.9180853572782214e+2
    u = u * x + 2.9441425117956526e+2
    u = u * x + -9.4667945689958979e+1
    u = u * x + 2.4474307443141791e+1
    return u * x + -4.9801004728787323


def x_log2x_approx(x):
    mantissa, exponent = math.frexp(x)
    return x * (exponent * math.log2(2) + log_approx2(mantissa))


def taylor_coef_log2(order: int) -> float:
    if order % 2 == 0:
        return -1 / (order * math.log(2))
    return 1 / (order * math.log(2))


if __name__ == '__main__':
    error_check(log_approx2, math.log2, 0.5, 1.0, 100_000, False)
    error_check(x_log2x_approx, x_log2x_builtin, 0.0, 1.0, 100_000, False)

    # import numpy as np
    # import baryrat

    # def f(x): return np.log2(x)
    # r = baryrat.brasil(f, [0.5, 1.0], 6)
    # r.__name__ = 'log_approx_brasil'
    #
    # print(r.nodes)
    # print(r.values)
    # print(r.weights)
    # print()

    # error_check(r, math.log2, 0.5, 1.0, 100_000, False)

    # Z = np.linspace(0.5, 1.0, 10000)
    # # noinspection PyTypeChecker
    # r2 = baryrat.aaa(Z, np.log2(Z))
    # r2.__name__ = 'log_approx_aaa'
    #
    # print(r2.nodes)
    # print(r2.values)
    # print(r2.weights)
    # print()
    #
    # error_check(r2, math.log2, 0.5, 1.0, 100_000, False)

    # zj, fj, wj = r.nodes, r.values, r.weights
    # xv = np.asanyarray(0.75).ravel()
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     print(xv[:, None])
    #     print()
    #     print(zj[None, :])
    #     print()
    #     print(xv[:, None] - zj[None, :])
    #     print()
    #     C = 1.0 / (xv[:, None] - zj[None, :])
    #     print(C)
    #     print()
    #     print(wj*fj)
    #     print()
    #     print(C.dot(wj*fj))
    #     print()
    #     print(C.dot(wj))
    #     print()
    #     print(C.dot(wj*fj) / C.dot(wj))
    #     print(math.log2(0.75))

    # from scipy.interpolate import pade
    # co_effs = [taylor_coef_log2(i) for i in range(1, 20)]
    # print(co_effs)
    # p, q = pade(co_effs, 10)
    #
    # print(p(0.75)/q(0.75))
    # print(math.log2(0.75))

    # from mpmath import taylor, pade, cos, log, polyval
    # a = taylor(cos, 1, 8)
    # print(a)
    # p, q = pade(a, 3, 3)
    #
    # print(p)
    # print(q)
    #
    # zz = 0.75
    # print(polyval(p[::-1], zz)/polyval(q[::-1], zz))
    # print(cos(zz))
