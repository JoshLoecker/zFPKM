from pathlib import Path
from typing import Literal, overload

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger

from zfpkm.density import dnorm
from zfpkm.types import DensityResult, ZFPKMResult


@overload
def zfpkm_plot(
    zfpkm_result: ZFPKMResult | list[ZFPKMResult],
    facet_titles: bool = False,
    plot_xfloor: float = -15,
    ncols: int = 5,
    return_fig: Literal[False] = False,
    save_filepath: str | Path | None = None,
) -> None: ...


@overload
def zfpkm_plot(
    zfpkm_result: ZFPKMResult | list[ZFPKMResult],
    facet_titles: bool = False,
    plot_xfloor: float = -15,
    ncols: int = 5,
    return_fig: Literal[True] = True,
    save_filepath: str | Path | None = None,
) -> plt.Figure: ...


def zfpkm_plot(
    zfpkm_result: ZFPKMResult | list[ZFPKMResult],
    facet_titles: bool = False,
    plot_xfloor: float = -15,
    ncols: int = 5,
    return_fig: bool = False,
    save_filepath: str | Path | None = None,
) -> plt.Figure | None:
    """Generate faceted log2(FPKM) density plots for zFPKM results.

    This function replicates the behavior of R's `zFPKM::zFPKMPlot`, producing a grid of subplots (facets) where each panel represents the density
    distribution of log2(FPKM) values for a given sample, along with its fitted normal density curve scaled to match the maimum density.

    This function automatically aligns all subplots to share consistent x- and y-axis limits,
        ensuring that distributions across samples can be visually compared.
    The layout, legend, and styling are designed to mirror the appearance of ggplot2's `facet_wrap()` and `theme_bw()` (as much as possible)

    :param zfpkm_result: the results from `zFPKM()[1]` (index 1 is the list of zfpkm results required for plotting)
    :param facet_titles: if true, display sample names above each subplot
    :param plot_xfloor: minimum x-axis for all subplots; set to -15 to match Hart et al. implementation of ignoring FPKM < -15
    :param ncols: the number of columns to plot
    :param return_fig: should the figure be returned?
    :param save_filepath: where should the figure be saved? Set to `None` to skip saving.

    :returns: (optionally) the generated figure
    """
    zfpkm_result: list[ZFPKMResult] = [zfpkm_result] if isinstance(zfpkm_result, ZFPKMResult) else zfpkm_result

    plot_dfs: list[pd.DataFrame] = []
    for result in zfpkm_result:
        name: str = result.name
        d: DensityResult = result.density
        mu: float = result.mu
        sd: float = result.sd

        # only used for Gaussian distribution estimation, not actual zFPKM calculation
        fitted: npt.NDArray[float] = np.asarray([dnorm(x, mean=mu, sd=sd) for x in d.x], dtype=float)
        max_fpkm = d.y.max()
        max_fitted: float = fitted.max()
        scale_fitted: npt.NDArray[float] = fitted * (max_fpkm / max_fitted)
        plot_dfs.append(pd.DataFrame({"sample_name": name, "log2fpkm": d.x, "fpkm_density": d.y, "fitted_density_scaled": scale_fitted}))

    mega_df = pd.concat(plot_dfs, ignore_index=True)
    max_x = mega_df["log2fpkm"].max()

    # two `max` calls are required: the first gets the max value in each series and the second max gets the max value in the dataframe
    max_y = mega_df[["fpkm_density", "fitted_density_scaled"]].max().max() * 1.05

    nplots = len(zfpkm_result)
    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 3.5, nrows * 3),
        sharex=True,
        sharey=True,
        squeeze=False,  # allow 1x1 grid remain an array
    )

    axes: npt.NDArray[plt.Axes] = axes.flatten()
    for ax, (sample_name, group) in zip(axes, mega_df.groupby("sample_name")):  # noqa: B905  we are intentionally skipping the last few axes that have been generated
        ax.plot(group["log2fpkm"], group["fpkm_density"], color="teal", alpha=0.7, label="fpkm_density")
        ax.plot(group["log2fpkm"], group["fitted_density_scaled"], color="salmon", alpha=0.7, label="fitted_density_scaled")
        ax.set_xlim(plot_xfloor, max_x)
        ax.set_ylim(0, max_y)
        if facet_titles:
            ax.set_title(sample_name, fontsize=9)
        ax.grid(alpha=0.5, linewidth=0.4)

    # hide unused subplots
    for ax in axes[nplots:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.text(0.5, 0.01, "log2(FPKM)", ha="center", fontsize=11)
    fig.text(0.01, 0.5, "[scaled] density", va="center", rotation="vertical", fontsize=11)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(
        top=0.95,
        bottom=0.05,
        left=0.05,
        right=0.98,
        wspace=0.10,
        hspace=0.10,
    )

    if save_filepath:
        save_filepath = Path(save_filepath)
        save_filepath.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_filepath.as_posix())
    if return_fig:
        return plt.gcf()
    if not save_filepath and not return_fig:
        logger.warning("Neither `save_filepath` nor `return_fig` were set; the generated figure will be discarded.")
    return None
