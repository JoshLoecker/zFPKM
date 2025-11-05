from __future__ import annotations

import re
from typing import Literal, cast, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = ["find_peaks"]


def _validate_args(
    x: npt.NDArray[np.float64],
    nups: int,
    ndowns: int,
    zero: Literal["0", "+", "-"],
    min_peak_height: float,
    min_peak_distance: int,
    threshold: float,
):
    """Validate input arguments to `find_peaks` function.

    Function created to reduce the complexity of `find_peaks` (ruff linting rule C901)
    """
    if x.ndim != 1:
        raise ValueError(f"Expected a 1D array, got {x.ndim}D array instead.")
    if np.any(np.isnan(x)):
        raise ValueError("Input x contains NaNs")
    if nups < 0:
        raise ValueError("Argument 'nups' must be non-negative")
    if ndowns < 0:
        raise ValueError("Argument 'ndowns' must be non-negative")
    if zero not in {"0", "+", "-"}:
        raise ValueError("Argument 'zero' must be '0', '+', or '-'")
    if min_peak_height < 0:
        raise ValueError("Argument 'min_peak_height' must be non-negative")
    if min_peak_distance < 0:
        raise ValueError("Argument 'minpeakdistance' must be non-negative")
    if threshold < 0:
        raise ValueError("Argument 'threshold' must be non-negative")


def _encode_signs(x: npt.NDArray[np.float64], zero: str) -> str:
    """Encode the signs of the differences in `x` into a string.

    Function created to reduce the complexity of `find_peaks` (ruff linting rule C901)
    """
    diff_signs = np.sign(np.diff(x))
    chars = ["+" if d > 0 else "-" if d < 0 else "0" for d in diff_signs]
    x_chars = "".join(chars)
    if zero != "0":
        x_chars = x_chars.replace("0", zero)
    return x_chars


@overload
def _enforce_minimum_peak_distance(df: pd.DataFrame, min_peak_distance: int, inplace: Literal[True] = True) -> None: ...


@overload
def _enforce_minimum_peak_distance(df: pd.DataFrame, min_peak_distance: int, inplace: Literal[False] = False) -> pd.DataFrame: ...


def _enforce_minimum_peak_distance(df: pd.DataFrame, min_peak_distance: int, inplace: bool = True) -> None | pd.DataFrame:
    """Enforces a minimum peak distance between identified peaks.

    Function created to reduce the complexity of `find_peaks` (ruff linting rule C901)

    Modifies `df` inplace
    """
    num_rows = df.shape[0]
    good_peaks = np.ones(num_rows, dtype=bool)
    for i in range(num_rows):
        if not good_peaks[i]:
            continue
        dpos = np.abs(df.at[i, "peak_idx"] - df["peak_idx"].values)
        close = (dpos > 0) & (dpos < min_peak_distance)
        good_peaks[close] = False

    if inplace:
        df.drop(df.index[~good_peaks], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return None
    return cast(pd.DataFrame, df.iloc[good_peaks, :].reset_index(drop=True))


def find_peaks(
    y: npt.ArrayLike,
    nups: int = 1,
    ndowns: int | None = None,
    zero: Literal["0", "+", "-"] = "0",
    peak_pattern: str | None = None,
    min_peak_height: float = 0.02,  # default for R's zFPKM `peak_parameters`
    min_peak_distance: int = 1,  # default for R's zFPKM `peak_parameters`
    threshold: float = 0.0,
    npeaks: int = 0,
    sortstr: bool = False,
) -> pd.DataFrame:
    """Identify peaks in a given time series.

    This function is modelled after R's `pracma::findpeaks` function.
    SciPy's `scipy.signal.find_peaks` provides different results than `pracma::findpeaks`,
        resulting in the requirement for this translation.

    References:
        1) pracma::findpeaks: https://rdrr.io/cran/pracma/man/findpeaks.html


    :param y: the stand-alone y-values as a numerical vector taken as a time series (no NA values allowed)
        Linear, arbitrary values will be placed from 0 to `len(y)` to serve as x-values
    :param nups: minimum number of increasing steps before a peak is reached
    :param ndowns: minimum number of decreasing steps after the peak (defaults to the same value as `nups`)
    :param zero: can be '+', '-', or '0'; how to interprete succeeding steps of the same value: increasing, decreasing, or special
    :param peak_pattern: define a peak as a regular pattern, such as the default pattern `[+]{1,}[-]{1,}`
        If a pattern is provided, parameters `nups` and `ndowns` are not taken into account
    :param min_peak_height: the minimum (absolute) height a peak has to have before being recognized
    :param min_peak_distance: the minimum distance (in indices) between peaks before they are counted
    :param threshold: the minimum difference in height between a peak and its surrounding values
    :param npeaks: the number of peaks to return (<=0 returns all)
    :param sortstr: should the peaks be returned in decreasing order of their peak height?

    :returns: A dataframe with the columns:
        height: the height of the peak
        peak_idx: the index (from `x`) of the identified peak
        start_idx: the starting index (from `x`) of the identified peak
        end_idx: the ending index (from `x`) of the identified peak
        If the dataframe is empty, no peaks could be identified
    """
    y: npt.NDArray[np.float64] = np.asarray(y, dtype=np.float64) if not isinstance(y, np.ndarray) else y
    npeaks: int = max(npeaks, 0)
    ndowns: int = ndowns or nups
    peak_pattern: str = peak_pattern or rf"[+]{{{nups},}}[-]{{{ndowns},}}"
    _validate_args(
        x=y,
        nups=nups,
        ndowns=ndowns,
        zero=zero,
        min_peak_height=min_peak_height,
        min_peak_distance=min_peak_distance,
        threshold=threshold,
    )

    # find peaks by regex matching the sign pattern
    derivative_chars = _encode_signs(x=y, zero=zero)
    matches: list[re.Match[str]] = list(re.finditer(peak_pattern, derivative_chars))
    if not matches:
        return pd.DataFrame(columns=["height", "peak_idx", "start_idx", "end_idx"])

    pattern_start_index: npt.NDArray[int] = np.asarray([m.start() for m in matches], dtype=int)
    pattern_end_index: npt.NDArray[int] = np.asarray([m.end() for m in matches], dtype=int)

    num_matches: int = len(pattern_start_index)
    peak_index: npt.NDArray[int] = np.zeros(num_matches, dtype=int)
    peak_height: npt.NDArray[np.float64] = np.zeros(num_matches, dtype=np.float64)

    # for each match region, find the local max
    for i in range(num_matches):
        segment: npt.NDArray[np.float64] = y[pattern_start_index[i] : pattern_end_index[i] + 1]
        segment_max_idx: int = np.argmax(segment).astype(int)
        peak_index[i] = pattern_start_index[i] + segment_max_idx
        peak_height[i] = segment[segment_max_idx]

    # filter for values that are too low or below the threshold difference
    x_left: float = y[pattern_start_index]
    x_right: float = y[np.minimum(pattern_end_index, y.size - 1)]
    valid_peaks: list[int] = list(np.where((peak_height >= min_peak_height) & ((peak_height - np.maximum(x_left, x_right)) >= threshold))[0].astype(int))  # noqa: E501  # fmt: skip

    if len(valid_peaks) == 0:
        return pd.DataFrame(columns=["height", "peak_idx", "start_idx", "end_idx"])

    out_df: pd.DataFrame = pd.DataFrame(
        data={
            "height": peak_height[valid_peaks].astype(np.float64),
            "peak_idx": peak_index[valid_peaks].astype(np.uint64),
            "start_idx": pattern_start_index[valid_peaks].astype(np.uint64),
            "end_idx": pattern_end_index[valid_peaks].astype(np.uint64),
        }
    )

    # sort by peak height in descending order (largest to smallest)
    if sortstr or min_peak_distance > 1:
        out_df.sort_values(by=["height"], ascending=False, ignore_index=True, inplace=True)

    # if provided, enforce a minimum distance between the peaks (removes smaller peaks next to larger ones)
    if min_peak_distance > 1 and len(out_df) > 1:
        _enforce_minimum_peak_distance(df=out_df, min_peak_distance=min_peak_distance)

    # if provided, limit the number of peaks returned
    if 0 < npeaks < out_df.shape[0]:
        out_df = cast(pd.DataFrame, out_df.iloc[:npeaks, :].reset_index(drop=True))

    return out_df
