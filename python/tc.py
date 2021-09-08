import timeit
from collections import defaultdict
from pygraphblas import *
from pygraphblas.descriptor import T0, ST0, ST1
from math import isclose


def PR(A, d_out, damping=0.85, itermax=100, tol=1e-4):
    n = A.nrows
    d = d_out / damping
    t = Vector.sparse(FP64, n)
    r = Vector.dense(FP64, n, fill=1.0 / n)
    dmin = Vector.dense(FP64, n, fill=1.0 / damping)
    d.max_monoid(dmin, out=d)
    teleport = (1 - damping) / n
    for i in range(1, itermax):
        t, r = r, t
        w = t / d
        r[:] = teleport
        A.plus_second(w, out=r, accum=FP64.plus)
        t -= r
        t.abs(out=t)
        if t.reduce_float() <= tol:
            break
    return r


def TC1(A, d):
    T = A.mxm(A, mask=A, desc=ST1)
    y = T.reduce_vector()
    k = y.reduce_float()
    return (3 * (A @ y) - 2 * (T.one() @ y) + y) / k


def TC3(A, d):
    M = A.tril(-1)
    T = A.plus_pair(A, mask=M, desc=ST1)
    y = T.reduce_vector() + T.reduce_vector(desc=T0)
    k = y.reduce_float()
    T2 = T.plus_second(y) + T.plus_second(y, desc=T0)
    r = (3 * A.plus_second(y)) + ((-2) * T2) + y
    return r / k


def tcount(A):
    L = A.tril()
    return L.plus_pair(L, mask=L).reduce_int()


@binary_op(FP64)
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def main(graphs, repeat=3):
    for name in graphs:
        print(f"Loading {name}")
        G = dict(Matrix.ssget(name, binary_cache_dir="~/.ssgetpy"))[
            name.split("/")[1] + ".mtx"
        ]
        G = G.cast(FP64)
        G.wait()
        d = G.reduce_vector()
        print(f"missing entries in d: {(d.size-d.nvals)} ")
        results = defaultdict(dict)
        print(f"{name} | {G.shape} | {G.nvals} edges | {tcount(G)} triangles")
        # options_set(burble=True)
        for centrality in PR, TC1, TC3:
            fname = centrality.__name__
            print(f"Running {fname} on {name} {repeat} times")
            result = timeit.repeat(
                "results[name][fname] = centrality(G,d)",
                repeat=repeat,
                number=1,
                globals=locals(),
            )
            print(
                f"{fname} on {name} took ",
                sum(result) / len(result),
                f"average for {repeat} runs",
            )
        # options_set(burble=False)
        tc1 = results[name]["TC1"].nonzero()
        tc3 = results[name]["TC3"].nonzero()
        print(f"TC1 equal to TC3? {tc1.iseq(tc3, isclose)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        graphs = [
            "Newman/karate",
            "SNAP/com-Youtube",
            "SNAP/as-Skitter",
            "SNAP/com-LiveJournal",
            "SNAP/com-Orkut",
            "SNAP/com-Friendster",
        ]
    else:
        graphs = sys.argv[1:]

    main(graphs)
