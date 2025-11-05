from __future__ import annotations

from typing import SupportsFloat

import numpy as np
import numpy.typing as npt

COLUMNS = 0
ROWS = 1


def _check_ndim(x: npt.NDArray[np.float64], *, ndim: int = 2):
    if x.ndim != ndim:
        raise ValueError(f"Expected a {ndim}-dimensional array, got '{x.ndim}' dimension(s)")


def binned_distribution_2d(
    x: npt.ArrayLike, weights: npt.ArrayLike, lo: npt.ArrayLike | SupportsFloat, up: npt.ArrayLike | SupportsFloat, n: int
) -> npt.NDArray[np.float64]:
    """Bin weighted distances."""
    x: npt.NDArray[np.float64] = np.asarray(x, dtype=np.float64)
    weights: npt.NDArray[np.float64] = np.asarray(weights, dtype=np.float64)
    lo: npt.NDArray[np.float64] = np.asarray(lo, dtype=np.float64)
    up: npt.NDArray[np.float64] = np.asarray(up, dtype=np.float64)

    _check_ndim(x)
    _check_ndim(weights)
    if lo.ndim > 1 or up.ndim > 1:
        raise ValueError("lo and up must be scalars or 1D (per-column values)")
    if lo.ndim == 0:
        lo: npt.NDArray[np.float64] = np.broadcast_to(lo, (x.shape[1],))
    if up.ndim == 0:
        up: npt.NDArray[np.float64] = np.broadcast_to(up, (x.shape[1],))
    if lo.shape != up.shape or lo.shape != (x.shape[1],):
        raise ValueError("lo/up must be of shape (M,), where input `x` is of shape (N, M)")

    N: int = x.shape[0]  # noqa: N806
    M: int = x.shape[1]  # noqa: N806
    if n <= 0:
        raise ValueError(f"n must be positive, got: {n}")

    ixmin: int = 0
    ixmax: int = n - 2
    delta: npt.NDArray[np.float64] = (up - lo) / (n - 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        xpos = (x - lo) / delta
    overflow_mask = (xpos > np.iinfo(np.int32).max) | (xpos < np.iinfo(np.int32).min)
    xpos = np.where(overflow_mask, np.nan, xpos)

    ix: npt.NDArray[np.int64] = np.floor(xpos, dtype=np.int64)
    fx: npt.NDArray[np.float64] = xpos - ix

    # scatter-add into y for three masks:
    # Case #1) ixmin <= ix <= imax    -> contribute to bins ix and ix+1
    # Case #2) ix == -1               -> contribute to bin 0 with weight fx
    # Case #3) ix == ixmax + 1        -> contribute to bin ix with (1 - fx)
    y: npt.NDArray[np.float64] = np.zeros((2 * n, M), dtype=np.float64)

    rows = np.arange(N)[:, np.newaxis]  # shape: (N, 1)
    cols = np.arange(M)[np.newaxis, :]  # shape: (1, M)

    # Case #1) interior bins
    interior: npt.NDArray[bool] = (ix >= ixmin) & (ix <= ixmax)
    if np.any(interior):
        r = rows[interior]
        c = cols[interior]
        i0 = ix[interior]
        f = fx[interior]
        w = weights[interior]
        np.add.at(y, (i0, c), (1.0 - f) * w)  # y[ix] += (1 - fx) * wi
        np.add.at(y, (i0 + 1, c), f * w)  # y[ix+1] += fx * wi

    # Case #2) left edge (ix == -1) -> bin 0 gets fx*wi
    left: npt.NDArray[bool] = ix == -1
    if np.any(left):
        r = rows[left]
        c = cols[left]
        f = fx[left]
        w = weights[left]
        np.add.at(y, (np.zeros_like(f, dtype=np.int64), c), f * w)

    # Case #3) right edge (ix == ixmax + 1) -> bin ix gets (1 -fx) * wi
    right: npt.NDArray[bool] = ix == ixmax + 1
    if np.any(right):
        r = rows[right]
        c = cols[right]
        i0 = ix[right]
        f = fx[right]
        w = weights[right]
        np.add.at(y, (i0, c), (1.0 - f) * w)

    return y


def dnorm_2d(
    x: npt.ArrayLike,
    mean: SupportsFloat | npt.ArrayLike = 0.0,
    sd: SupportsFloat | npt.ArrayLike = 1.0,
    log: bool = False,
    fast_dnorm: bool = False,
) -> npt.NDArray[np.float64]:
    # Constants
    m_ln2: float = 0.693147180559945309417232121458  # ln(2)
    m_1_sqrt_2pi: float = 0.398942280401432677939946059934  # 1/sqrt(2pi)
    m_ln_sqrt_2pi: float = 0.918938533204672741780329736406  # log(sqrt(2*pi))
    r_d__0 = np.float64(-np.inf) if log else np.float64(0.0)  # R's R_D__0: (log_p ? ML_NEGINF : 0.)

    # 11 bits exponent, where one bit is used to indicate special numbers (e.g. NaN and Infinity),
    #   so the maximum exponent is 10 bits wide (2^10 == 1024).
    # dbl_min_exp = -1022
    dbl_min_exp = -1074

    # The mantissa is 52 bits wide, but because numbers are normalized the initial binary 1 is represented
    #   implicitly (the so-called "hidden bit"), which leaves us with the ability to represent 53 bits wide mantissa.
    dbl_mant_dig = 53

    x: npt.NDArray[np.float64] = np.asarray(x, dtype=np.float64)
    mean: npt.NDArray[np.float64] = np.asarray(mean, dtype=np.float64)
    sd: npt.NDArray[np.float64] = np.asarray(sd, dtype=np.float64)
    x, mean, sd = np.broadcast_arrays(x, mean, sd)

    _check_ndim(x)
    _check_ndim(mean)
    _check_ndim(sd)

    out: npt.NDArray[np.float64] = np.full(x.shape, r_d__0, dtype=np.float64)

    # set output to nan where x, mean, or sd is nan
    nan_mask = np.isnan(x) | np.isnan(mean) | np.isnan(sd)
    out[nan_mask] = np.nan

    # set output to nan where standard deviation is less than 0
    sd_neg_mask = sd < 0.0
    out[sd_neg_mask] = np.nan

    # set output to infinity where the standard deviation is 0 OR x is equal to the mean at that (a, b) coordinate
    sd0_mask = sd == 0.0
    mean_eq_x_mask = x == mean
    out[sd0_mask & mean_eq_x_mask] = np.inf

    # set output to nan where the standard deviation is not finite, x does not equal the mean, and x is not finite
    sd_non_finite = ~np.isfinite(sd)
    x_not_finite_and_eq_mean = (~np.isfinite(x)) & mean_eq_x_mask
    out[x_not_finite_and_eq_mean] = np.nan

    done = nan_mask | sd_neg_mask | sd0_mask | sd_non_finite | x_not_finite_and_eq_mean

    # z = (x - mean) / sd where still active
    # From this point on, dividing by `sd` should be because we know it is not 0
    z = np.divide(x - mean, sd, out=np.full_like(x, np.nan), where=~done)

    z_bad_equality = (~np.isfinite(z)) & mean_eq_x_mask
    out[z_bad_equality] = np.nan
    done |= z_bad_equality | (~np.isfinite(z))

    # large abs values -> r_d__0
    a = np.abs(z)
    too_large = a >= (2.0 * np.sqrt(np.finfo(np.float64).max))
    out[too_large] = r_d__0
    done |= too_large

    active = ~done
    if not np.any(active):
        return out

    if log:
        out[active] = -(m_ln_sqrt_2pi + 0.5 * a[active] * a[active] + np.log(sd[active]))
        return out

    use_fast = (a < 5.0) | bool(fast_dnorm)
    fast_mask = active & use_fast
    out[fast_mask] = (m_1_sqrt_2pi * np.exp(-0.5 * a[fast_mask] * a[fast_mask])) / sd[fast_mask]

    split_mask = active & (~use_fast)
    if np.any(split_mask):
        boundary = np.sqrt(-2.0 * m_ln2 * (dbl_min_exp + 1 - dbl_mant_dig))
        over_b = split_mask & (a > boundary)
        out[over_b] = np.float64(0.0)
        remaining = split_mask & (~(a > boundary))
        if np.any(remaining):
            a1 = np.ldexp(np.rint(np.ldexp(a[remaining], 16)), -16)
            a2 = a[remaining] - a1
            out[remaining] = (m_1_sqrt_2pi / sd[remaining]) * (np.exp(-0.5 * a1 * a1) * np.exp((-a1 * a2) - (0.5 * a2 * a2)))

    return out


def nrd0_2d(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    x: npt.NDArray[np.float64] = np.asarray(x, dtype=np.float64)
    _check_ndim(x)

    mask = np.isfinite(x)
    counts = mask.sum(axis=COLUMNS)
    if np.any(counts < 2):
        bad = np.nonzero(counts < 2)[0].tolist()
        raise ValueError(f"Each column needs at least 2 data points. Problem columns: {bad}")

    hi: npt.NDArray[np.float64] = np.std(x, ddof=1, axis=COLUMNS)
    q25: npt.NDArray[np.float64] = np.percentile(x, 25, axis=COLUMNS)
    q75: npt.NDArray[np.float64] = np.percentile(x, 75, axis=COLUMNS)

    iqr_over_134: npt.NDArray[np.float64] = (q75 - q25) / 1.34

    # We are using a cascading series of checks on `lo` to  make sure it is a non-zero value
    lo: npt.NDArray[np.float64] = np.minimum(hi, iqr_over_134)

    zero_mask = lo == 0
    lo = np.where(zero_mask & (hi != 0), hi, lo)
    is_still_zero = lo == 0
    if np.any(is_still_zero):
        first_col_vals = np.abs(x[0, :])
        lo = np.where(is_still_zero & (first_col_vals != 0), first_col_vals, 0)
    lo = np.where(lo == 0, 1.0, lo)

    n_per_col = x.shape[0]  # get the number of rows (data points) per column
    return 0.9 * lo * np.asarray(n_per_col, dtype=np.float64) ** (-1.0 / 5.0)


def density2D(x: npt.ArrayLike, n: int = 512, cut: int = 3):
    x = np.asarray(x, dtype=np.float64)
    rows, cols = x.shape
    weights: npt.NDArray[np.float64] = np.full(shape=x, fill_value=1 / n, dtype=np.float64)
    bw_calc = nrd0_2d(x)
    from_ = np.minimum(x) - cut * bw_calc

    nx = n
    weights: npt.NDArray[np.float64] = np.full(shape=nx, fill_value=1 / nx, dtype=np.float64)
    total_mass = np.float64(nx / n)


if __name__ == "__main__":
    x = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    density2D(x)
