"""Exact family membership tests; runnable with Python or importable in Sage.

French coordinates: cell (c,r) has upper-right corner (c,r).
A triangular partition consists of the positive lattice points below a
line c + s*r = d, with s > 0. Strict separation is equivalent to allowing
the boundary on either side, since there are finitely many relevant points.
"""
from fractions import Fraction
from functools import lru_cache
import json


def conjugate(lam):
    return tuple(sum(x >= c for x in lam) for c in range(1, max(lam, default=0)+1))


@lru_cache(None)
def triangular_interval(mu):
    """Return the open interval of separating slopes, or None.

    For each occupied row endpoint (x,r) and first missing cell (y,j),
    impose x+s*r < y+s*j. These inequalities are necessary and sufficient.
    """
    lower, upper = Fraction(0), None
    outside = [(x+1, j+1) for j, x in enumerate(mu)] + [(1, len(mu)+1)]
    for r, x in enumerate(mu, 1):
        for y, j in outside:
            a, b = r-j, y-x
            if a == 0:
                if b <= 0:
                    return None
            elif a > 0:
                bound = Fraction(b, a)
                upper = bound if upper is None else min(upper, bound)
            else:
                lower = max(lower, Fraction(b, a))
    return (lower, upper) if upper is None or lower < upper else None


def subpartitions(lam, prefix=()):
    if not lam:
        yield tuple(x for x in prefix if x)
        return
    for x in range(min(lam[0], prefix[-1] if prefix else lam[0]), -1, -1):
        yield from subpartitions(lam[1:], prefix+(x,))


def addition_support(lam, mu):
    rows, cols = set(), set()
    for r, x in enumerate(lam, 1):
        y = mu[r-1] if r <= len(mu) else 0
        if x > y:
            rows.add(r)
            cols.update(range(y+1, x+1))
    return rows, cols


def family_witnesses(lam):
    """Search ALL contained triangular partitions, including the empty one.

    Also test the larger family allowing additions in the UNION of rows
    1,2 and columns 1,2, to disambiguate the interpretation of the question.
    """
    witnesses = {}
    for mu in subpartitions(tuple(lam)):
        if triangular_interval(mu) is None:
            continue
        rows, cols = addition_support(lam, mu)
        conditions = {
            'first_two_rows': rows <= {1, 2},
            'first_two_columns': cols <= {1, 2},
            'one_row': len(rows) <= 1,
            'one_column': len(cols) <= 1,
            'union_first_two_rows_columns': all(
                r <= 2 or c <= 2
                for r, x in enumerate(lam, 1)
                for c in range((mu[r-1] if r <= len(mu) else 0)+1, x+1)),
        }
        for name, condition in conditions.items():
            if condition:
                witnesses.setdefault(name, mu)
    return witnesses


def fast_family_witness(lam):
    """Equivalent to the four usual families, without enumerating all subdiagrams."""
    for transpose, shape in [(False, tuple(lam)), (True, conjugate(lam))]:
        shape = shape + (0,)*(max(0, 2-len(shape)))
        for x in range(shape[0], shape[2] - 1 if len(shape)>2 else -1, -1):
            for y in range(min(x, shape[1]), shape[2]-1 if len(shape)>2 else -1, -1):
                mu = tuple(v for v in (x,y)+shape[2:] if v)
                if triangular_interval(mu) is not None:
                    return ('first_two_columns' if transpose else 'first_two_rows', mu)
        for r, row in enumerate(shape):
            for x in range(row-1, (shape[r+1] if r+1<len(shape) else 0)-1, -1):
                mu = tuple(v for v in shape[:r]+(x,)+shape[r+1:] if v)
                if triangular_interval(mu) is not None:
                    return ('one_column' if transpose else 'one_row', mu)
    return None


def lower_hull(lam):
    points = sorted([(x+1,r) for r,x in enumerate(lam,1)]+[(1,len(lam)+1)])
    hull = []
    for p in points:
        if hull and hull[-1][0] == p[0]:
            continue
        while len(hull)>=2:
            a,b = hull[-2:]
            cross = (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])
            if cross > 0:
                break
            hull.pop()
        hull.append(p)
    return hull


def is_concave(lam):
    hull = lower_hull(lam)
    return all(any((b[0]-a[0])*(r-a[1])-(b[1]-a[1])*(x-a[0]) < 0
                   for a,b in zip(hull,hull[1:]))
               for r,x in enumerate(lam,1))


def partitions(n, ceiling=None):
    if n == 0:
        yield ()
    else:
        for first in range(min(n, ceiling if ceiling is not None else n),1-1,-1):
            for tail in partitions(n-first,first):
                yield (first,)+tail


if __name__ == '__main__':
    for filename in ['partition_results.json', 'partition_results_old.json']:
        results = json.load(open(filename))['results']
        print(filename, flush=True)
        for key, solved in results.items():
            lam = tuple(map(int, key.split(',')))
            if lam < conjugate(lam):
                continue
            witnesses = family_witnesses(lam)
            usual = any(k != 'union_first_two_rows_columns' for k in witnesses)
            if not usual:
                print(lam, 'saved_solved=', solved, 'witnesses=', witnesses, flush=True)
