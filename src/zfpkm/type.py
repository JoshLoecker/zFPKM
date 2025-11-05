from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable, Literal, Protocol, TypeVar, Union, runtime_checkable

import numpy as np
import numpy.typing as npt

__all__ = ["AnyRealScalar", "ApproxArgs", "ApproxResult", "DensityArgs", "DensityResult", "FindPeaksArgs", "RegularizedArray"]
AnyRealScalar = Union[float, int, np.floating, np.integer]


@runtime_checkable
class DataclassLike(Protocol):
    """Validate that a given input contains dataclass fields.

    An example usage of this is seen in :class:_ToDict, where the input must be a dataclass of some kind that is compatible with the `asdict` function.
    """

    __dataclass_fields__: Mapping[str, Any]


T = TypeVar("T", bound=DataclassLike)


class _ToDict:
    def _to_dict(self: T, pop_fields: Sequence[str] | None = None) -> dict[str, Any]:
        dict_: dict[str, Any] = asdict(self)  # type: ignore[invalid-argument-type]
        if pop_fields is None:
            return dict_

        for field in pop_fields:
            dict_.pop(field)
        return dict_


@dataclass(slots=True)
class DensityArgs(_ToDict):
    """Arguments provided to the density calculation.

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
    """

    bw: int | float | Literal["nrd0"] | Callable[[npt.ArrayLike], float | int] = "nrd0"
    adjust: AnyRealScalar = 1
    kernel: Literal["gaussian", "epanechnikov", "rectangular", "triangular", "biweight", "cosine", "optcosine"] = "gaussian"
    weights: npt.ArrayLike | None = None
    n: int = 512
    from_: AnyRealScalar | None = None
    to_: AnyRealScalar | None = None
    cut: int = 3
    ext: int = 4
    remove_na: bool = False
    kernel_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary."""
        return self._to_dict()


@dataclass(slots=True)
class ApproxArgs(_ToDict):
    """Arguments provided to the approximation/interpolation calculation.

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
    """

    xout: npt.ArrayLike | None = None
    method: str | int = "linear"
    n: int = 50
    yleft: float | None = None
    yright: float | None = None
    rule: int | tuple[int, int] = 1
    f: float = 0.0
    ties: Callable[[npt.ArrayLike], np.float64] | Literal["mean", "first", "last", "min", "max", "median", "sum"] = "mean"
    na_rm: bool = True

    def to_dict(self, pop_fields: tuple[str, ...] | None = ("n",)) -> dict[str, Any]:
        """Convert to a dictionary."""
        return self._to_dict(pop_fields=pop_fields)


@dataclass
class FindPeaksArgs(_ToDict):
    """Arguments provided to the find_peaks calculation.

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
    """

    nups: int = 1
    ndowns: int | None = None
    zero: Literal["0", "+", "-"] = "0"
    peak_pattern: str | None = None
    min_peak_height: float | Sequence[float] = 0.02
    min_peak_distance: int | Sequence[int] = 1
    threshold: float = 0.0
    npeaks: int = 0
    sortstr: bool = False

    def to_dict(self, pop_fields: tuple[str, ...] | None = ("min_peak_height", "min_peak_distance")) -> dict[str, Any]:
        """Convert to a dictionary."""
        return self._to_dict(pop_fields=pop_fields)


@dataclass(slots=True, frozen=True)
class ApproxResult:
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]


@dataclass(slots=True, frozen=True)
class DensityResult:
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    x_grid: npt.NDArray[np.float64]
    y_grid: npt.NDArray[np.float64]
    bw: float
    n: int


@dataclass(slots=True, frozen=True)
class RegularizedArray:
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    not_na: npt.NDArray[bool]
    kept_na: bool
