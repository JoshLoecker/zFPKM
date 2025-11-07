from itertools import combinations

import pandas as pd
import pytest

from zfpkm import zFPKM


@pytest.fixture
def fpkm() -> pd.DataFrame:
    return pd.read_csv("tests/data/fpkm.csv", header=0, index_col=0)


@pytest.fixture
def ronammar_zfpkm() -> pd.DataFrame:
    return pd.read_csv("tests/data/ronammar_expected_zfpkm.csv", header=0, index_col=0)


@pytest.fixture
def hart_zfpkm() -> pd.DataFrame:
    return pd.read_csv("tests/data/hart_expected_zfpkm.csv", header=0, index_col=0)


@pytest.fixture
def calc_zfpkm(fpkm: pd.DataFrame) -> pd.DataFrame:
    return zFPKM(fpkm)[0]


def test_indices(fpkm: pd.DataFrame, ronammar_zfpkm: pd.DataFrame, hart_zfpkm: pd.DataFrame, calc_zfpkm: pd.DataFrame):
    assert fpkm.index.name == ronammar_zfpkm.index.name == hart_zfpkm.index.name == calc_zfpkm.index.name


def test_column_names(fpkm: pd.DataFrame, ronammar_zfpkm: pd.DataFrame, hart_zfpkm: pd.DataFrame, calc_zfpkm: pd.DataFrame):
    for left, right in combinations([fpkm, ronammar_zfpkm, hart_zfpkm, calc_zfpkm], 2):
        assert all(col in right.columns for col in left.columns)


def test_calc_to_hart_series(hart_zfpkm: pd.DataFrame, calc_zfpkm: pd.DataFrame):
    calc_to_hart_comparison: dict[str, float] = {
        "A549": 0.054106,
        "AG04450": 0.051436,
        "BJ": 0.036909,
        "GM12878": 0.063916,
        "H1-hESC": 0.046128,
        "HMEC": 0.161435,
        "HSMM": 0.023650,
        "HUVEC": 0.059313,
        "HeLaS3": 0.082333,
        "HepG2": 0.101363,
        "IMR90": 0.122933,
        "K562": 0.006952,
        "MCF7": 0.018529,
        "NHEK": 0.018796,
        "NHLF": 0.101171,
        "SK-N-SH": 0.061993,
        "SK-N-SH_RA": 0.044190,
    }
    for col, atol in calc_to_hart_comparison.items():
        pd.testing.assert_series_equal(left=hart_zfpkm[col], right=calc_zfpkm[col], atol=atol + 1e-6)


def test_calc_to_ronammar(ronammar_zfpkm: pd.DataFrame, calc_zfpkm: pd.DataFrame):
    calc_to_ronammar_comparison: dict[str, float] = {
        "A549": 1.361283e-01,
        "AG04450": 5.329071e-15,
        "BJ": 7.812429e-02,
        "GM12878": 1.017352e-01,
        "H1-hESC": 5.329071e-15,
        "HMEC": 1.374269e-01,
        "HSMM": 1.291804e-01,
        "HUVEC": 1.140164e-01,
        "HeLaS3": 5.329071e-15,
        "HepG2": 7.993606e-15,
        "IMR90": 1.528900e-01,
        "K562": 5.329071e-15,
        "MCF7": 9.769963e-15,
        "NHEK": 8.881784e-15,
        "NHLF": 1.200444e-01,
        "SK-N-SH": 5.329071e-15,
        "SK-N-SH_RA": 1.085291e-01,
    }
    for col, atol in calc_to_ronammar_comparison.items():
        pd.testing.assert_series_equal(left=ronammar_zfpkm[col], right=calc_zfpkm[col], atol=atol + 1e-7)
