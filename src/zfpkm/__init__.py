from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from zfpkm.density import density
from zfpkm.peak_finder import find_peaks
from zfpkm.type import ApproxArgs, DensityArgs, DensityResult, FindPeaksArgs

__all__ = ["density", "find_peaks", "zfpkm"]


def zfpkm(
    df: pd.DataFrame,
    density_args: DensityArgs | None = None,
    approx_args: ApproxArgs | None = None,
    find_peaks_args: FindPeaksArgs | None = None,
) -> pd.DataFrame:
    """ZFPKM Transformations.

    References:
        1) zFPKM implementation in R: https://github.com/ronammar/zFPKM
        2) zFPKM publication: https://doi.org/10.1186/1471-2164-14-778

    :param df: The raw FPKM values to perform zFPKM on.
    :param density_args: The default arguments provided to the density calculaton
    :param approx_args: The default arguments provided to the approx calculation
    :param find_peaks_args: The default arguments provided to the find_peaks calculation
    :returns: A dataframe containing the zFPKM values
    """
    density_args = density_args or DensityArgs()
    approx_args = approx_args or ApproxArgs()
    find_peaks_args = find_peaks_args or FindPeaksArgs()

    if isinstance(find_peaks_args.min_peak_height, Sequence) and len(find_peaks_args.min_peak_height) != df.shape[1]:
        raise ValueError(
            f"If providing a sequence for `min_peak_height`, its length must match the number of columns in `df`. "
            f"Input dataframe has {df.shape[1]} columns, but `min_peak_height` only has {len(find_peaks_args.min_peak_height)} values."
        )
    if isinstance(find_peaks_args.min_peak_distance, Sequence) and len(find_peaks_args.min_peak_distance) != df.shape[1]:
        raise ValueError(
            f"If providing a sequence for `min_peak_distance`, its length must match the number of columns in `df`. "
            f"Input dataframe has {df.shape[1]} columns, but `min_peak_distance` only has {len(find_peaks_args.min_peak_distance)} values."
        )
    peak_heights: list[float] = (
        list(find_peaks_args.min_peak_height)
        if isinstance(find_peaks_args.min_peak_height, Sequence)
        else [find_peaks_args.min_peak_height] * df.shape[1]
    )
    peak_distances: list[int] = (
        list(find_peaks_args.min_peak_distance)
        if isinstance(find_peaks_args.min_peak_distance, Sequence)
        else [find_peaks_args.min_peak_distance] * df.shape[1]
    )

    row_names: list[Any] = df.index.tolist()
    col_names: list[Any] = df.columns.tolist()
    fpkm: npt.NDArray[np.float64] = df.values.astype(np.float64)

    # Ignore np.log2(0) errors; we know this will happen, and are removing non-finite values in the density calculation
    # This is required in order to match R's zFPKM calculations, as R's `density` function removes NA values.
    with np.errstate(divide="ignore", invalid="ignore"):
        log2fpkm: npt.NDArray[np.float64] = np.log2(fpkm, dtype=np.float64)

    fpkm_vals_at_mu: npt.NDArray[np.float64] = np.zeros(log2fpkm.shape[1], dtype=np.float64)
    results: npt.NDArray[np.float64] = np.empty_like(df, dtype=np.float64)

    for col_idx in range(log2fpkm.shape[1]):
        log2_col = log2fpkm[:, col_idx]
        d = density(log2_col, approx_args=approx_args, **density_args.to_dict())
        peaks = find_peaks(d.y, min_peak_height=peak_heights[col_idx], min_peak_distance=peak_distances[col_idx], **find_peaks_args.to_dict())
        peak_positions: npt.NDArray[np.float64] = d.x_grid[peaks["peak_idx"]]

        sd = np.float64(1.0)
        mu = np.float64(0.0)
        fpkm_at_mu = np.float64(0.0)
        if peak_positions.size > 0:
            mu = np.float64(peak_positions.max())
            u = np.float64(log2fpkm[log2fpkm > mu].mean())
            fpkm_at_mu = np.float64(d.y_grid[peaks.loc[np.argmax(peak_positions), "peak_idx"]])
            sd = np.float64((u - mu) * np.sqrt(np.pi / 2))
        results[:, col_idx] = (log2_col - mu) / sd
        fpkm_vals_at_mu[col_idx] = fpkm_at_mu

    return pd.DataFrame(results, index=row_names, columns=col_names)
