from timeit import repeat
from collections import defaultdict
from pygraphblas import *
from pygraphblas.descriptor import T0, ST0, ST1
from math import isclose


def PR(A, damping=0.85, itermax=100, tol=1e-4):
    d = A.reduce_vector()
    n = A.nrows
    t = Vector.sparse(FP64, n)
    r = Vector.dense(FP64, n, fill=1.0 / n)
    d.assign_scalar(damping, accum=FP64.div)
    teleport = (1 - damping) / n
    for i in range(itermax):
        temp = t
        t = r
        r = temp
        w = t / d
        r[:] = teleport
        A.plus_second(w, out=r, accum=FP64.plus, desc=T0)
        t -= r
        t.abs(out=t)
        if t.reduce_float() <= tol:
            break
    return r


def TC1(A):
    T = A.mxm(A, mask=A, desc=ST1)
    y = T.reduce_vector()
    k = y.reduce_float()
    return (3 * (A @ y) - 2 * (T.one() @ y) + y) / k


def TC2(A):
    T = A.plus_pair(A, mask=A, desc=ST1)
    y = Vector.dense(FP64, A.nrows)
    T.reduce_vector(out=y, accum=FP64.plus)
    k = y.reduce_float()
    return (3 * A.plus_second(y) - 2 * T.plus_second(y) + y) / k


def TC3(A):
    M = A.tril(-1)
    T = A.plus_pair(A, mask=M, desc=ST1)
    y = T.reduce() + T.reduce(desc=ST0)
    k = y.reduce_float()
    return (
        3 * A.plus_second(y) - (2 * (T.plus_second(y) + T.plus_second(y, desc=ST0))) + y
    ) / k


def tcount(A):
    L = A.tril()
    return L.plus_pair(L, mask=L).reduce_int()


graphs = [
     "Newman/karate",
     "SNAP/com-Youtube",
     "SNAP/as-Skitter",
     "SNAP/com-LiveJournal",
     'SNAP/com-Orkut',
     'SNAP/com-Friendster'
]


@binary_op(FP64)
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


R = 3
for name in graphs:
    print(f"Loading {name}")
    G = dict(Matrix.ssget(name, binary_cache_dir="~/.ssgetpy"))[
        name.split("/")[1] + ".mtx"
    ]
    G = G.cast(FP64)
    G.wait()
    results = defaultdict(dict)
    print(f"{name} | {G.shape} | {G.nvals} edges | {tcount(G)} triangles")
    # for centrality in TC1, TC2, TC3:
    for centrality in TC1, TC3:
        fname = centrality.__name__
        print(f"Running {fname} on {name} {R} times")
        result = repeat(
            "results[name][fname] = centrality(G)", repeat=R, number=1, globals=locals()
        )
        print(
            f"{fname} on {name} took ",
            sum(result) / len(result),
            f"average for {R} runs",
        )
    tc1 = results[name]["TC1"].nonzero()
    # tc2 = results[name]["TC2"].nonzero()
    tc3 = results[name]["TC3"].nonzero()
    # print(f"TC1 equal to TC2? {tc1.iseq(tc2, isclose)}")
    # print(f"TC2 equal to TC3? {tc2.iseq(tc3, isclose)}")
    print(f"TC1 equal to TC3? {tc1.iseq(tc3, isclose)}")
    # breakpoint()
