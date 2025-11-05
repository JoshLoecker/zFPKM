from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from typing import Callable, Literal, SupportsFloat, cast, overload

import numpy as np
import numpy.typing as npt
from loguru import logger

from zfpkm.approx import approx
from zfpkm.type import AnyRealScalar, ApproxArgs, ApproxResult, DensityResult

__all__ = ["binned_distribution", "density", "dnorm", "nrd0"]

LN2 = 0.693147180559945309417232121458
INV_SQRT_2PI = 0.398942280401432677939946059934
LOG_SQRT_2PI = 0.918938533204672741780329736406
DBL_MIN_EXP = -1074
DBL_MANT_DIG = 53
MAX_SQRT = math.sqrt(sys.float_info.max)
UNDERFLOW_BOUNDARY = math.sqrt(-2.0 * LN2 * (DBL_MIN_EXP + 1 - DBL_MANT_DIG))


def binned_distribution_vec(x: npt.ArrayLike, weights: npt.ArrayLike, lo: SupportsFloat, up: SupportsFloat, n: int) -> npt.NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    lo = np.float64(lo)
    up = np.float64(up)

    ixmin = 0
    ixmax = n - 2
    delta = (up - lo) / (n - 1)

    y = np.zeros(2 * n, dtype=np.float64)
    xpos = (x - lo) / delta
    good = (xpos <= np.float64(sys.maxsize)) & (xpos >= -np.float64(sys.maxsize))
    if not np.any(good):  # if any are not good, return y
        return y

    xpos = xpos[good]
    w = weights[good]
    ix = np.floor(xpos).astype(np.int64)
    fx = (xpos - ix).astype(np.float64)

    in_range = (ix >= ixmin) & (ix <= ixmax)
    left_edge = ix == -1
    right_edge = ix == (ixmax + 1)

    if np.any(in_range):
        ix_in = ix[in_range]
        fx_in = fx[in_range]
        w_in = w[in_range]
        np.add.at(y, ix_in, (1.0 - fx_in) * w_in)
        np.add.at(y, ix_in + 1, fx_in * w_in)

    if np.any(left_edge):
        np.add.at(y, 0, (fx[left_edge] * w[left_edge]).sum())

    if np.any(right_edge):
        ix_r = ix[right_edge]
        np.add.at(y, ix_r, (1.0 - fx[right_edge]) * w[right_edge])

    return y


def binned_distribution(x: npt.ArrayLike, weights: npt.ArrayLike, lo: SupportsFloat, up: SupportsFloat, n: int) -> npt.NDArray[np.float64]:
    """Bin weighted distances."""
    x: npt.NDArray[np.float64] = np.asarray(x, dtype=np.float64)
    weights: npt.NDArray[np.float64] = np.asarray(weights, dtype=np.float64)
    f_lo = np.float64(lo)
    f_up = np.float64(up)
    ixmin: int = 0
    ixmax: int = n - 2
    delta: np.float64 = (f_up - f_lo) / (n - 1)

    y: npt.NDArray[np.float64] = np.zeros((2 * n,), dtype=np.float64)
    for i in range(len(x)):
        i: int
        xpos: np.float64 = (x[i] - f_lo) / np.float64(delta)
        if xpos > sys.maxsize or xpos < -sys.maxsize:  # avoid integer overflows (taken from R's massdist.c)
            continue
        ix = np.int64(np.floor(xpos))
        fx = np.float64(xpos - ix)
        wi = np.float64(weights[i])
        if ixmin <= ix <= ixmax:
            y[ix] += (1 - fx) * wi
            y[ix + 1] += fx * wi
        elif ix == -1:
            y[0] += fx * wi
        elif ix == ixmax + 1:
            y[ix] += (1 - fx) * wi
    return y


def dnorm(x: SupportsFloat, mean: SupportsFloat = 0.0, sd: SupportsFloat = 1.0, log: bool = False, fast_dnorm: bool = False) -> np.float64:
    """Density function for the normal distribution.

    This is a reproduction of R's `density` function.
    Neither SciPy nor NumPy are capable of producing KDE values that align with R, and as a result,
        a manual translation of R's KDE implementation was necessary.

    References:
        1) Documentation: https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/Normal
        2) Source code (2025-OCT-30): https://github.com/wch/r-source/blob/3f7e2528990351bc4b0d1f1b237714668ab4c0c5/src/nmath/dnorm.c

    Args:
        x: Value at which to evaluate the density.
        mean: Mean of the normal distribution.
        sd: Standard deviation of the normal distribution.
        log: If True, return the log density.
        fast_dnorm: If True, use a faster but less accurate calculation for small `x`.


    """
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

    f_x: np.float64 = np.float64(x)
    f_mean: np.float64 = np.float64(mean)
    f_sd: np.float64 = np.float64(sd)

    if np.isnan(f_x) or np.isnan(f_mean) or np.isnan(f_sd):
        return np.float64(np.nan)
    if f_sd < 0.0:
        return np.float64(np.nan)
    if not np.isfinite(f_sd):
        return r_d__0
    if not np.isfinite(x) and x == f_mean:
        return np.float64(np.nan)
    if f_sd == 0.0:
        return np.float64(np.inf) if x == f_mean else r_d__0

    # From this point on, dividing by `sd` is safe because we know it is not 0
    z = np.float64((f_x - f_mean) / f_sd)
    if (not np.isfinite(z)) and (f_x == f_mean):
        return np.float64(np.nan)

    if not np.isfinite(z):
        return r_d__0

    a = np.fabs(z)
    if a >= 2 * np.sqrt(np.finfo(float).max):
        return r_d__0
    if log:
        return -np.float64(m_ln_sqrt_2pi + 0.5 * a * a + np.log(f_sd))
    if a < 5 or fast_dnorm:  # for `x < 5`, this is more accurate but less fast
        return m_1_sqrt_2pi * np.exp(-0.5 * a * a) / f_sd

    # underflow boundary
    boundary = np.sqrt(-2.0 * m_ln2 * (dbl_min_exp + 1 - dbl_mant_dig))
    if a > boundary:
        return np.float64(0.0)

    # Now, to get full accuracy, split x into two parts,
    #   x = x1+x2, such that |x2| <= 2^-16.
    #   Assuming that we are using IEEE doubles, that means that
    #   x1*x1 is error free for x<1024 (but we have x < 38.6 anyway).
    #   If we do not have IEEE this is still an improvement over the naive formula.
    a1 = np.ldexp(np.rint(np.ldexp(a, 16)), -16)
    a2 = a - a1
    return np.float64((m_1_sqrt_2pi / f_sd) * (np.exp(-0.5 * a1 * a1) * np.exp((-a1 * a2) - (0.5 * a2 * a2))))


def dnorm_new(
    x: SupportsFloat,
    mean: SupportsFloat = 0.0,
    sd: SupportsFloat = 1.0,
    log: bool = False,
    fast_dnorm: bool = False,
) -> np.float64:
    """Scalar Normal density, behavior aligned with R's dnorm.

    Optimized for speed by avoiding NumPy on scalar paths.
    """
    # Bind frequently used math funcs as locals (faster than global lookups)
    isnan = math.isnan
    isfinite = math.isfinite
    exp = math.exp
    logf = math.log
    ldexp = math.ldexp
    fabs = math.fabs

    xf = float(x)
    mf = float(mean)
    sdf = float(sd)

    # NaN handling & parameter checks
    if isnan(xf) or isnan(mf) or isnan(sdf):
        return np.float64(math.nan)
    if sdf < 0.0:
        return np.float64(math.nan)
    if not isfinite(sdf):
        return np.float64(-math.inf) if log else np.float64(0.0)
    if (not isfinite(xf)) and (xf == mf):
        # matches R's NA behavior when both are +/-Inf and equal
        return np.float64(math.nan)
    if sdf == 0.0:
        return np.float64(math.inf) if (xf == mf) else (np.float64(-math.inf) if log else np.float64(0.0))

    inv_sd = 1.0 / sdf
    z = (xf - mf) * inv_sd

    if not isfinite(z):
        # If z blew up but x==mean (0/0 case), R gives NaN; otherwise density is 0 (or log  -Inf)
        if xf == mf:
            return np.float64(math.nan)
        return np.float64(-math.inf) if log else np.float64(0.0)

    a = abs(z)

    # Protect a*a from overflow in the pathological case
    if a >= 2.0 * MAX_SQRT:
        return np.float64(-math.inf) if log else np.float64(0.0)

    # Log path: one formula, no exp calls
    if log:
        return np.float64(-(LOG_SQRT_2PI + 0.5 * a * a) - logf(sdf))

    # Fast (and still accurate for |z| < 5) path
    if fast_dnorm or a < 5.0:
        return np.float64(INV_SQRT_2PI * exp(-0.5 * a * a) * inv_sd)

    # Very far tail: check underflow boundary
    if a > UNDERFLOW_BOUNDARY:
        return np.float64(0.0)

    # Tail split for full accuracy (|a2| <= 2^-16)
    a1 = ldexp(round(ldexp(a, 16)), -16)
    a2 = a - a1

    return np.float64((INV_SQRT_2PI * inv_sd) * (exp(-0.5 * a1 * a1) * exp((-a1 * a2) - (0.5 * a2 * a2))))


def dnorm_vec(x: npt.ArrayLike, mean: SupportsFloat = 0.0, sd: SupportsFloat = 1.0, log: bool = False):
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)

    out = np.empty(np.broadcast(x, mean, sd).shape, dtype=np.float64)

    # Standard formula in log-space for stability
    z = (x - mean) / sd
    logpdf = -(LOG_SQRT_2PI + 0.5 * z * z) - np.log(sd)
    out[...] = logpdf if log else np.exp(logpdf)

    out = np.where(np.isnan(x) | np.isnan(mean) | np.isnan(sd) | (sd < 0), np.nan, out)
    out = np.where(~np.isfinite(sd), (-np.inf if log else 0.0), out)
    out = np.where((sd == 0) & (x == mean), np.inf, out)
    out = np.where((sd == 0) & (x != mean), (-np.inf if log else 0.0), out)
    return out


def nrd0(x: npt.ArrayLike) -> np.float64:
    """Calculate nrd0 from R source.

    This bandwidth calculation matches R's

    References:
        1) Documentation: https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/bandwidth
        2) Source code (as of 2025-OCT-30, copied below): https://github.com/wch/r-source/blob/trunk/src/library/stats/R/bandwidths.R
            ```R
            bw.nrd0 <- function (x)
            {
                if(length(x) < 2L) stop("need at least 2 data points")
                hi <- sd(x)
                if(!(lo <- min(hi, IQR(x)/1.34)))# qnorm(.75) - qnorm(.25) = 1.34898
                    (lo <- hi) || (lo <- abs(x[1L])) || (lo <- 1.)
                0.9 * lo * length(x)^(-0.2)
            }
            ```
    """
    x: npt.NDArray[np.float64] = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        raise ValueError("need at least 2 data points")

    hi: np.float64 = np.float64(np.std(x, ddof=1))
    q25, q75 = cast(tuple[np.float64, np.float64], cast(object, np.percentile(x, [25, 75])))
    iqr_over_134 = (q75 - q25) / 1.34

    # We are using a cascading series of checks on `lo` to  make sure it is a non-zero value
    lo = np.float64(min(hi, iqr_over_134))
    if lo == 0:
        if hi != 0:
            lo = hi
        elif abs(x[0]) != 0:
            lo = abs(x[0])
        else:
            lo = 1.0

    return np.float64(0.9 * lo * x.size ** (-1 / 5))


@overload
def density(
    x: npt.ArrayLike,
    /,
    *,
    bw: int | float | Literal["nrd0"] | Callable[[npt.ArrayLike], float | int] = "nrd0",
    adjust: AnyRealScalar = 1,
    kernel: Literal["gaussian", "epanechnikov", "rectangular", "triangular", "biweight", "cosine", "optcosine"] = "gaussian",
    weights: npt.ArrayLike | None = None,
    n: int = 512,
    from_: AnyRealScalar | None = None,
    to_: AnyRealScalar | None = None,
    cut: int = 3,
    ext: int = 4,
    remove_na: bool = False,
    kernel_only: Literal[True] = True,
    approx_args: ApproxArgs | None = None,
) -> float: ...


@overload
def density(
    x: npt.ArrayLike,
    /,
    *,
    bw: int | float | Literal["nrd0"] | Callable[[npt.ArrayLike], float | int] = "nrd0",
    adjust: AnyRealScalar = 1,
    kernel: Literal["gaussian", "epanechnikov", "rectangular", "triangular", "biweight", "cosine", "optcosine"] = "gaussian",
    weights: npt.ArrayLike | None = None,
    n: int = 512,
    from_: AnyRealScalar | None = None,
    to_: AnyRealScalar | None = None,
    cut: int = 3,
    ext: int = 4,
    remove_na: bool = False,
    kernel_only: Literal[False] = False,
    approx_args: ApproxArgs | None = None,
) -> DensityResult: ...


def density(
    x: npt.ArrayLike,
    /,
    *,
    weights: npt.ArrayLike | None = None,
    bw: int | float | Literal["nrd0"] | Callable[[npt.ArrayLike], float | int] = "nrd0",
    adjust: AnyRealScalar = 1,
    kernel: Literal["gaussian", "epanechnikov", "rectangular", "triangular", "biweight", "cosine", "optcosine"] = "gaussian",
    n: int = 512,
    from_: AnyRealScalar | None = None,
    to_: AnyRealScalar | None = None,
    cut: int = 3,
    ext: int = 4,
    remove_na: bool = False,
    kernel_only: bool = False,
    approx_args: ApproxArgs | None = None,
) -> DensityResult | float:
    """Compute kernel density estimates (KDE) using FFT method.

    This is a reproduction of R's `density` function.
    Neither SciPy nor NumPy are capable of producing KDE values that align with R, and as a result,
        a manual translation of R's KDE implementation was necessary.

    :param x: Input data points.
    :param bw: Bandwidth for the kernel. If "nrd0", uses R's nrd0 method.
    :param adjust: Adjustment factor for the bandwidth.
    :param kernel: Kernel type to use.
    :param weights: Optional weights for each data point.
    :param n: Number of points in the output grid.
    :param from_: Start of the grid (calculated automatically if not provided).
    :param to_: End of the grid (calculated automatically if not provided).
    :param cut: Number of bandwidths to extend the grid on each side.
    :param ext: Number of bandwidths to extend the grid for FFT calculation.
    :param remove_na: Whether to remove NA values from `x`.
    :param kernel_only: If True, returns only the integral of the kernel function.
    :param approx_args: Arguments for the approx function

    :returns: If :param kernel_only: is `True`, returns a float indicating the integral of the kernel function.
        Otherwise, returns a DensityResult named tuple containing:
        x: The x-coordinates of the density estimate.
        y: The estimated density values at the x-coordinates.
        x_grid: The grid of x-coordinates used for FFT calculation.
        y_grid: The density values on the FFT grid.
        bw: The bandwidth used.
        n: The number of points in the output grid.
    """
    if not isinstance(x, (Sequence, np.ndarray)) or (len(x) < 2 and bw == "nrd0"):
        raise ValueError("Need at at least two points to select a bandwidth automatically using 'nrd0'")
    if kernel_only:
        if kernel == "gaussian":
            return 1 / (2 * np.sqrt(np.pi))
        elif kernel == "epanechnikov":
            return 3 / (5 * np.sqrt(5))
        elif kernel == "rectangular":
            return np.sqrt(3) / 6
        elif kernel == "triangular":
            return np.sqrt(6) / 9
        elif kernel == "biweight":
            return 5 * np.sqrt(7) / 49
        elif kernel == "cosine":
            return 3 / 4 * np.sqrt(1 / 3 - 2 / np.pi**2)
        elif kernel == "optcosine":
            return np.sqrt(1 - 8 / np.pi**2) * np.pi**2 / 16

    if kernel != "gaussian":
        raise NotImplementedError(f"Only 'gaussian' kernel is implemented; got '{kernel}'")

    x: npt.NDArray[np.float64] = np.asarray(x, dtype=np.float64)

    has_weights = weights is not None
    weights: npt.NDArray[np.float64] | None = np.asarray(weights, np.float64) if weights is not None else None
    if has_weights and (weights is not None and weights.size != x.size):
        raise ValueError(f"The length of provided weights does not match the length of x: {weights.size} != {x.size}")

    x_na: npt.NDArray[np.bool_] = np.isnan(x)
    if np.any(x_na):
        if remove_na:
            x: npt.NDArray[np.float64] = x[~x_na]
            if has_weights and weights is not None:
                true_d: bool = weights.sum().astype(float) == 1
                weights: npt.NDArray[np.float64] = weights[~x_na]
                if true_d:
                    weights: npt.NDArray[np.float64] = weights / weights.sum()
        else:
            raise ValueError("NA values found in 'x'. Set 'remove_na=True' to ignore them.")

    nx = n
    x_finite = np.isfinite(x)
    if np.any(~x_finite):
        x = x[x_finite]
        nx = x.size
    if not has_weights:
        weights: npt.NDArray[np.float64] = np.full(shape=nx, fill_value=1 / nx, dtype=np.float64)
        total_mass = nx / n
    else:
        weights: npt.NDArray[np.float64] = np.asarray(weights, dtype=np.float64)
        if not np.all(np.isfinite(weights)):
            raise ValueError("Non-finite values found in 'weights'.")
        if np.any(weights < 0):
            raise ValueError("Negative values found in 'weights'.")
        wsum: float = weights.sum()
        if np.any(~x_finite):
            weights: npt.NDArray[np.float64] = weights[x_finite]
            total_mass = float(weights.sum() / wsum)
        else:
            total_mass = float(1)

    n_user: int = n
    n = max(n, 512)
    if n > 512:  # round n up to the next power of 2 (i.e., 2^8=512, 2^9=1024)
        n: int = int(2 ** np.ceil(np.log2(n)))

    if isinstance(bw, str) and bw != "nrd0":
        raise TypeError("Bandwidth 'bw' must be a number or 'nrd0'.")
    elif isinstance(bw, str) and bw == "nrd0":
        bw_calc = nrd0(x)
    elif isinstance(bw, (float, int)):
        bw_calc = float(bw)
    elif isinstance(bw, Callable):
        bw_calc = float(bw(x))
    else:
        raise TypeError(f"Bandwidth 'bw' must be a number or 'nrd0'. Got: {type(bw)}")
    if not np.isfinite(bw_calc):
        raise ValueError("Calculated bandwidth 'bw' is not finite.")
    bw_calc *= adjust

    if bw_calc <= 0:
        raise ValueError("Bandwidth 'bw' must be positive.")

    # have to use `... if ... else` because `0` is falsey, resulting in the right-half being used instead of the user-provided value
    from_ = float(from_ if from_ is not None else x.min() - cut * bw_calc)
    to_ = float(to_ if to_ is not None else x.max() + cut * bw_calc)

    if not np.isfinite(from_):
        raise ValueError("'from_' is not finite.")
    if not np.isfinite(to_):
        raise ValueError("'to_' is not finite.")

    lo = float(from_ - ext * bw_calc)
    up = float(to_ + ext * bw_calc)

    y: npt.NDArray[np.float64] = binned_distribution_vec(x, weights, lo, up, n) * total_mass

    kords: npt.NDArray[np.float64] = np.linspace(start=0, stop=((2 * n - 1) / (n - 1) * (up - lo)), num=2 * n, dtype=np.float64)
    kords[n + 1 : 2 * n] = -kords[n:1:-1]  # mirror/negate tail: R's kords[n:2] will index from the reverse if `n`>2

    # Initial diverge here (inside dnorm calculation)
    kords: npt.NDArray[np.float64] = dnorm_vec(kords, sd=bw_calc)

    fft_y: npt.NDArray[np.complex128] = np.fft.fft(y)
    conj_fft_kords: npt.NDArray[np.complex128] = np.conjugate(np.fft.fft(kords))
    # Must multiply by `kords.size` because R does not produce a normalize inverse FFT, but NumPy normalizes by `1/size`
    kords: npt.NDArray[np.complex128] = np.fft.ifft(fft_y * conj_fft_kords) * kords.size
    kords: npt.NDArray[np.float64] = (np.maximum(0, kords.real)[0:n]) / y.size  # for values of kords, get 0 or kords[i], whichever is larger
    xords: npt.NDArray[np.float64] = np.linspace(lo, up, num=n, dtype=np.float64)

    # xp=known x-coords, fp=known y-cords, x=unknown x-coords; returns interpolated (e.g., unknown) y-coords
    interp_x: npt.NDArray[np.float64] = np.linspace(from_, to_, num=n_user, dtype=np.float64)

    approx_args = approx_args or ApproxArgs()
    if approx_args.xout is not None:
        logger.warning(
            "`approx_args.xout` will be overwritten with the density x-coordinates. "
            "To remove this warning, do not provide the `approx.xout` argument."
        )
        approx_args.xout = interp_x

    interp_y: ApproxResult = approx(xords, kords, **approx_args.to_dict())  # xout is provided in approx_args

    return DensityResult(
        x=interp_x,
        y=interp_y.y,
        x_grid=xords,
        y_grid=kords,
        bw=float(bw_calc),
        n=n,
    )
