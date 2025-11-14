import numpy as np
import numpy.typing as npt
import pandas as pd

from zfpkm.approx import approx
from zfpkm.density import bin_distribution, density, dnorm, nrd0
from zfpkm.peak_finder import find_peaks
from zfpkm.plot import zfpkm_plot
from zfpkm.types import ApproxResult, DensityResult, ZFPKMResult

__all__ = [
    "ApproxResult",
    "DensityResult",
    "ZFPKMResult",
    "approx",
    "bin_distribution",
    "density",
    "dnorm",
    "find_peaks",
    "nrd0",
    "zFPKM",
    "zfpkm_plot",
]


def zFPKM(fpkm: pd.DataFrame, log: bool = True) -> tuple[pd.DataFrame, list[ZFPKMResult]]:  # noqa: N802
    """Calculate zFPKM from raw FPKM values.

    This function will perform a zFPKM calculation, following Hart et al's (2013) paper and the zFPKM implementation at: `https://github.com/ronammar/zFPKM`

    The input dataframe should have:
        - Row names as genomic identifier (Entrez Gene ID, Ensembl Gene ID, Gene Symbol, etc.)
        - Column names as sample identifiers

    If `log=True`, the FPKM values should be raw values.

    :param fpkm: raw FPKM values.
    :param log: If True, perform `np.log2` on the values

    :returns: a tuple of:
        1) The zFPKM calculation, where the index and columns are in the same order as the input dataframe
        2) A list of associated metadata, the same length as the number of input columns, containing:
            - Density calculations
            - Mean (peak) Gaussian distribution value
            - Standard deviation of the Gaussian distribution
            - The FPKM value at the Gaussian mean (peak)
    """
    if log:
        with np.errstate(divide="ignore"):
            log2_vals: npt.NDArray[float] = np.log2(fpkm.values).astype(float)
    else:
        log2_vals = fpkm.values.astype(float)

    zfpkm_df: pd.DataFrame = pd.DataFrame(data=0.0, index=fpkm.index, columns=fpkm.columns)
    zfpkm_results: list[ZFPKMResult] = []
    for i, col in enumerate(fpkm.columns):
        log2_values: npt.NDArray[float] = log2_vals[:, i]
        d = density(log2_values)
        peaks: pd.DataFrame = find_peaks(d.y)
        peak_positions = d.x[peaks["peak_idx"]]

        sd = 1.0
        mu = 0.0
        fpkm_at_mu = 0.0
        if peak_positions.size > 0:
            mu = float(peak_positions.max())
            u = float(log2_values[log2_values > mu].mean())
            fpkm_at_mu = float(d.y[int(peaks.loc[np.argmax(peak_positions).astype(int), "peak_idx"])])
            sd = float((u - mu) * np.sqrt(np.pi / 2))
        zfpkm_df[col] = np.asarray((log2_values - mu) / sd, dtype=float)
        zfpkm_results.append(ZFPKMResult(name=col, density=d, mu=mu, sd=sd, fpkm_at_mu=fpkm_at_mu))
    return zfpkm_df, zfpkm_results
