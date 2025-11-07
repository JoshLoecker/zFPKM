import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger

try:
    from scipy.signal import find_peaks
    from sklearn.neighbors import KernelDensity
except ImportError as e:
    logger.warning("Install scipy and sklearn first! python3 -m pip install scipy scikit-learn")
    raise ImportError from e

from zfpkm import DensityResult, ZFPKMResult, zfpkm_plot


def sklearn_zfpkm():
    fpkm = pd.read_csv("../tests/data/fpkm.csv", header=0, index_col=0)

    with np.errstate(divide="ignore"):
        log2values: npt.NDArray[float] = np.log2(fpkm.values)
    finite_mask = np.isfinite(log2values).all(axis=1)
    log2finite = log2values[finite_mask, :]

    zfpkm_results: list[ZFPKMResult] = []
    zfpkm_series: list[pd.Series] = []
    for i, col in enumerate(fpkm.columns):
        # 1D KDE *always* has one feature and many samples. scikit-learn expects data in the format of (n_samples, n_features)
        # Thus, we use `.reshape(-1, 1)` because we know there is a single feature
        # Even though the data is FPKM of many genes for a single sample, it is still one feature over many samples
        # `-1` indicates the unknown dimension (the number of samples)
        # `1` indicates the known dimension (the number of genes [also known as features])
        # https://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html#sphx-glr-auto-examples-neighbors-plot-kde-1d-py
        log2_col = log2finite[:, i]
        kde = KernelDensity(kernel="gaussian").fit(log2_col.reshape(-1, 1))

        x_range: npt.NDArray[float] = np.linspace(log2_col.min(), log2_col.max(), 512).reshape(-1, 1)
        density: npt.NDArray[float] = np.exp(kde.score_samples(x_range))
        peaks, _ = find_peaks(density, height=0.02, distance=1.0)
        peak_positions = x_range[peaks]

        mu = 0
        max_fpkm = 0
        stddev = 1
        if len(peaks) != 0:
            mu = peak_positions.max()
            max_fpkm = density[peaks[np.argmax(peak_positions)]]
            u = log2_col[log2_col > mu].mean()
            stddev = (u - mu) * np.sqrt(np.pi / 2)
        zfpkm_series.append(pd.Series((log2_col - mu) / stddev, dtype=float, name=col))
        zfpkm_results.append(
            ZFPKMResult(name=col, density=DensityResult(x=x_range.ravel(), y=density, bw=1.0, n=512), mu=mu, sd=stddev, fpkm_at_mu=max_fpkm)
        )
    zfpkm_df = pd.concat(zfpkm_series, axis=1)
    fig = zfpkm_plot(zfpkm_results, return_fig=True)
    fig.show()


if __name__ == "__main__":
    sklearn_zfpkm()
