from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from zfpkm.type import ApproxResult, RegularizedArray

__all__ = ["approx"]


def _coerce_to_float_array(a: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Coerce input to a 1D float array.

    :param a: the array to coerce

    :returns: A floating point 1D array.
    """
    arr: npt.NDArray[np.float64] = np.asarray(a, dtype=np.float64)
    return arr if arr.ndim == 1 else arr.ravel()


def _regularize_values(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    na_rm: bool,
    ties: Callable[[npt.ArrayLike], np.float64] | Literal["mean", "first", "last", "min", "max", "median", "sum"] = "mean",
) -> RegularizedArray:
    """Reimplementation of R's regularize.values() used by approx().

      - Removes NA pairs if na_rm is True (else requires x to have no NA).
      - Sorts by x (stable).
      - Collapses duplicate x via `ties` aggregator.

    :param x: the x values to regularize
    :param y: ties: the corresponding y values
    :param na_rm: should NA values be removed?
    :param ties: how to handle duplicate x values; can be a string or a callable

    :returns: A NamedTuple with:
        - x: sorted unique x
        - y: aggregated y aligned with x
        - not_na: boolean mask of y that are not NaN
        - kept_na: True iff any NaN remained in y after regularization
    """
    x: npt.NDArray[np.float64] = np.asarray(x, dtype=np.float64)
    y: npt.NDArray[np.float64] = np.asarray(y, dtype=np.float64)
    if na_rm:
        # Remove pairs where x or y is NA
        keep = ~np.isnan(x) & ~np.isnan(y)
        x = x[keep]
        y = y[keep]
        kept_na = False
    else:
        # R's C code errors if na_rm=False and any x is NA
        if np.isnan(x).any():
            raise ValueError("approx(x,y, .., na.rm=False): NA values in x are not allowed")
        kept_na = np.isnan(y).any()

    if x.size == 0:
        return RegularizedArray(
            x=np.asarray(x, dtype=np.float64),
            y=np.asarray(y, dtype=np.float64),
            not_na=np.array([], dtype=bool),
            kept_na=kept_na,
        )

    # Use a stable sort (mergesort) to match R's order()
    order: npt.NDArray[np.int64] = np.argsort(x, kind="stable")
    x_sorted: npt.NDArray[np.float64] = x[order]
    y_sorted: npt.NDArray[np.float64] = y[order]

    change = np.empty(x_sorted.shape, dtype=bool)
    change[0] = True
    change[1:] = x_sorted[1:] != x_sorted[:-1]
    start_idx = np.flatnonzero(change)
    counts = np.diff(np.append(start_idx, x_sorted.size))
    unique_x = x_sorted[start_idx]

    # Handle the 'ties' argument
    if callable(ties):
        y_agg = np.empty_like(unique_x)
        for k, (start, count) in enumerate(zip(start_idx, counts)):
            y_agg[k] = ties(y_sorted[start : start + count])
    else:
        if ties in ("mean", "avg", "average"):
            sum_y = np.add.reduceat(y_sorted, start_idx)
            y_agg = sum_y / counts
        elif ties in ("first", "left"):
            y_agg = y_sorted[start_idx]
        elif ties in ("last", "right"):
            y_agg = y_sorted[start_idx + counts - 1]
        elif ties == "min":
            y_agg = np.minimum.reduceat(y_sorted, start_idx)
        elif ties == "max":
            y_agg = np.maximum.reduceat(y_sorted, start_idx)
        elif ties == "median":
            y_agg = np.array([np.median(y_sorted[start : start + count]) for start, count in zip(start_idx, counts)], dtype=np.float64)
        elif ties == "sum":
            y_agg = np.add.reduceat(y_sorted, start_idx)
        else:
            raise ValueError("Unsupported `ties`; use a callable or one of 'mean', 'first', 'last', 'min', 'max', 'median', 'sum'.")

    not_na = ~np.isnan(y_agg)
    kept_na_final = bool(np.isnan(y_agg).any())
    return RegularizedArray(
        x=unique_x.astype(np.float64, copy=False),
        y=y_agg.astype(np.float64, copy=False),
        not_na=not_na.astype(bool, copy=False),
        kept_na=kept_na_final,
    )


def approx(
    x: npt.ArrayLike,
    y: npt.ArrayLike | None = None,
    xout: npt.ArrayLike | None = None,
    method: str | int = "linear",
    n: int = 50,
    yleft: float | None = None,
    yright: float | None = None,
    rule: int | tuple[int, int] = 1,
    f: float = 0.0,
    ties: Callable[[npt.ArrayLike], np.float64] | Literal["mean", "first", "last", "min", "max", "median", "sum"] = "mean",
    na_rm: bool = True,
) -> ApproxResult:
    """Faithful Python port of R's `approx` function.

    This implementation aims to replicate the behavior of R's `approx`
    function, including its handling of `ties`, `rule`, `na_rm`, and
    `method`.

    :param x: Coordinates, or y-values if `y` is None.
    :param y: y-coordinates. If None, `x` is treated as `y` and `x` becomes `1..n`.
    :param xout: Points at which to interpolate.
    :param method: Interpolation method. "linear" (1) or "constant" (2).
    :param n: If `xout` is not provided, interpolation happens at `n` equally spaced points spanning the range of `x`.
    :param yleft: Value to use for extrapolation to the left. Defaults to `NA` (np.nan) if `rule` is 1.
    :param yright: Value to use for extrapolation to the right. Defaults to `NA` (np.nan) if `rule` is 1.
    :param rule: Extrapolation rule.
        - 1: Return `np.nan` for points outside the `x` range.
        - 2: Use `yleft` and `yright` for points outside the range.
    :param f: For `method="constant"`, determines the split point. `f=0` is left-step, `f=1` is right-step, `f=0.5` is midpoint.
    :param ties: How to handle duplicate `x` values. Can be 'mean', 'first', 'last', 'min', 'max', 'median', 'sum', or a callable function.
    :param na_rm: If True, `NA` pairs are removed before interpolation. If False, `NA`s in `x` cause an error, `NA`s in `y` are propagated.

    :returns: `Approx` NamedTuple with:
        - 'x': numpy array of xout used
        - 'y': numpy array of interpolated values
    """
    if y is None:
        y_arr = _coerce_to_float_array(x)
        x_arr = np.arange(1.0, y_arr.size + 1.0, dtype=np.float64)
    else:
        x_arr = _coerce_to_float_array(x)
        y_arr = _coerce_to_float_array(y)
        if x_arr.size != y_arr.size:
            raise ValueError("x and y must have same length")

    # --- Method normalization ---
    # (matches R's pmatch() result: 1=linear, 2=constant)
    if isinstance(method, str):
        m = method.strip().lower()
        if m.startswith("lin"):
            method_code = 1
        elif m.startswith("const"):
            method_code = 2
        else:
            raise ValueError("invalid interpolation method")
    else:
        if method in (1, 2):
            method_code = int(method)
        else:
            raise ValueError("invalid interpolation method")

    # --- Rule normalization ---
    if isinstance(rule, (Sequence, np.ndarray)):
        rule_list: list[int] = list(rule)
        if not (1 <= len(rule_list) <= 2):
            raise ValueError("`rule` must have length 1 or 2")
        if len(rule_list) == 1:
            rule_list = [rule_list[0], rule_list[0]]
    else:
        rule_list = [rule, rule]

    # Sort by x, collapse ties, and handle NAs like R's regularize.values()
    r: RegularizedArray = _regularize_values(x_arr, y_arr, na_rm=na_rm, ties=ties)
    x_reg: npt.NDArray[np.float64] = r.x
    y_reg: npt.NDArray[np.float64] = r.y
    not_na_mask: npt.NDArray[bool] = r.not_na
    # no_na is True if we don't have to worry about NAs in y_reg
    no_na: bool = na_rm or (not r.kept_na)
    # nx is the number of *valid* (non-NA) points for interpolation
    nx: int = x_reg.size if no_na else int(np.count_nonzero(not_na_mask))

    if np.isnan(nx):
        raise ValueError("invalid length(x)")

    # R's C_ApproxTest logic
    if nx <= 1:
        if method_code == 1:
            raise ValueError("need at least two non-NA values to interpolate")
        if nx == 0:
            raise ValueError("zero non-NA points")

    # set extrapolation values (yleft/yright)
    # This logic matches R's C code (R_approxtest)
    if no_na:
        y_first: float = y_reg[0]
        y_last: float = y_reg[-1]
    else:
        # Find first and last non-NA y values
        y_valid: npt.NDArray[np.float64] = y_reg[not_na_mask]
        y_first: float = y_valid[0]
        y_last: float = y_valid[-1]

    yleft_val: float = (np.nan if int(rule_list[0]) == 1 else y_first) if yleft is None else float(yleft)
    yright_val: float = (np.nan if int(rule_list[1]) == 1 else y_last) if yright is None else float(yright)

    # fabricate xout if it is missing
    if xout is None:
        if n <= 0:
            raise ValueError("'approx' requires n >= 1")
        if no_na:
            x_first: np.float64 = x_reg[0]
            x_last: np.float64 = x_reg[nx - 1]
        else:
            x_valid: npt.NDArray[np.float64] = x_reg[not_na_mask]
            x_first: np.float64 = x_valid[0]
            x_last: np.float64 = x_valid[-1]
        xout_arr: npt.NDArray[np.float64] = np.linspace(x_first, x_last, num=int(n), dtype=np.float64)
    else:
        xout_arr: npt.NDArray[np.float64] = _coerce_to_float_array(xout)

    # replicate R's C_ApproxTest checks
    if method_code == 2 and (not np.isfinite(f) or f < 0.0 or f > 1.0):
        raise ValueError("approx(): invalid f value; with `method=2`, `f` must be in the range [0,1] (inclusive) or NA")
    if not no_na:
        # R re-filters x and y here if NAs remained
        x_reg: npt.NDArray[np.float64] = x_reg[not_na_mask]
        y_reg: npt.NDArray[np.float64] = y_reg[not_na_mask]

    # perform interpolation
    # this section is a vectorized form of the logic from R's approx1 and R_approxfun
    yout: npt.NDArray[np.float64] = np.empty_like(xout_arr, dtype=np.float64)
    mask_nan_xout: npt.NDArray[bool] = np.isnan(xout_arr)
    yout[mask_nan_xout] = np.nan
    mask_valid: npt.NDArray[bool] = ~mask_nan_xout

    if x_reg.size == 0:
        # No valid points to interpolate against
        yout[mask_valid] = np.nan
        return ApproxResult(x=xout_arr, y=yout)

    xv: npt.NDArray[np.float64] = xout_arr[mask_valid]
    left_mask: npt.NDArray[bool] = xv < x_reg[0]
    right_mask: npt.NDArray[bool] = xv > x_reg[-1]
    mid_mask: npt.NDArray[bool] = ~(left_mask | right_mask)

    res: npt.NDArray[np.float64] = np.empty_like(xv, dtype=np.float64)
    res[left_mask] = yleft_val
    res[right_mask] = yright_val

    if np.any(mid_mask):
        xm: npt.NDArray[np.float64] = xv[mid_mask]

        # Find indices
        # j_right[k] = index of first x_reg > xm[k]
        j_right: npt.NDArray[np.int64] = cast(npt.NDArray[np.int64], np.searchsorted(x_reg, xm, side="right"))
        # j_left[k] = index of first x_reg >= xm[k]
        j_left: npt.NDArray[np.int64] = cast(npt.NDArray[np.int64], np.searchsorted(x_reg, xm, side="left"))

        # Points that exactly match an x_reg value
        eq_mask: npt.NDArray[bool] = np.asarray(j_left != j_right, dtype=bool)
        # Points that fall between x_reg values
        in_interval_mask: npt.NDArray[bool] = ~eq_mask

        res_mid: npt.NDArray[np.float64] = np.empty_like(xm, dtype=np.float64)

        if np.any(eq_mask):
            # For exact matches, use the corresponding y_reg value
            # R's C code uses the 'j_left' index here
            res_mid[eq_mask] = y_reg[j_left[eq_mask]]

        if np.any(in_interval_mask):
            # R's C code sets i = j-1, where j is the 'right' index
            j: npt.NDArray[np.int64] = j_right[in_interval_mask]
            i: npt.NDArray[np.int64] = j - 1

            # Get the surrounding x and y values
            xi: npt.NDArray[np.float64] = x_reg[i]
            xj: npt.NDArray[np.float64] = x_reg[j]
            yi: npt.NDArray[np.float64] = y_reg[i]
            yj: npt.NDArray[np.float64] = y_reg[j]

            if method_code == 1:  # linear
                t: npt.NDArray[np.float64] = (xm[in_interval_mask] - xi) / (xj - xi)
                res_mid[in_interval_mask] = yi + (yj - yi) * t
            else:  # constant
                # This matches R_approxfun's constant logic
                if f == 0.0:
                    res_mid[in_interval_mask] = yi
                elif f == 1.0:
                    res_mid[in_interval_mask] = yj
                else:
                    # computes R's (1-f)*yi + f*yj
                    f1: np.float64 = np.float64(1.0 - f)
                    f2: np.float64 = np.float64(f)
                    part: npt.NDArray[np.float64] = np.zeros_like(yi, dtype=np.float64)
                    if f1 != 0.0:
                        part += yi * f1
                    if f2 != 0.0:
                        part += yj * f2
                    res_mid[in_interval_mask] = part

        res[mid_mask] = res_mid

    yout[mask_valid] = res
    return ApproxResult(x=xout_arr, y=yout)
