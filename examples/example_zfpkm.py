import pandas as pd

from zfpkm import zFPKM, zfpkm_plot


def example_zfpkm():
    fpkm = pd.read_csv("../tests/data/fpkm.csv", header=0, index_col=0)
    ronammar_zfpkm = pd.read_csv("../tests/data/ronammar_expected_zfpkm.csv", header=0, index_col=0)
    hart_zfpkm = pd.read_csv("../tests/data/hart_expected_zfpkm.csv", header=0, index_col=0)

    zfpkm_calc = zFPKM(fpkm)

    fig = zfpkm_plot(zfpkm_calc[1], return_fig=True)
    fig.show()


if __name__ == "__main__":
    example_zfpkm()
