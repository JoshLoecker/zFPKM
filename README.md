# zfpkm
## Overview
This package performs zFPKM on RNA-seq FPKM data. This implementation is adapted from [ronammar/zFPKM](https://github.com/ronammar/zFPKM), which was originally based on [Hart et al. 2013](https://doi.org/10.1186/1471-2164-14-778) ([PMID24215113](https://pubmed.ncbi.nlm.nih.gov/24215113/)). The original article recommends selecting an active/inactive cutoff at `-3`; this value was selected based on experimental data that indicated the number of active promoters becomes more than the number of repressed promoters [Figure 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC3870982/figure/F1/).
## Installation
This package can be installed from GitHub or PyPi
### PyPi

```shell
python3 -m pip install zfpkm
```

### Latest GitHub Version

```shell
python3 -m pip install git+https://github.com/JoshLoecker/zfpkm.git
```

## Usage
To calculate zFPKM, simply import the `zfpkm` function and provide the raw, non-normalized FPKM values as a pandas `DataFrame`. The row names of the input dataframe should be the genomic identifier (Entrez IDs, Ensembl IDs, Gene Symbols, etc.) and the column names should be the sample name. The returned `DataFrame` will have the same number of rows and columns (in the same order provided) with a modified z-transformation applied.

```python
import pandas as pd

from zfpkm import zFPKM, zfpkm_plot


def main():
    fpkm = pd.read_csv("fpkm.csv", index_col=0, header=0)
    zfpkm_df, zfpkm_results = zFPKM(fpkm)
    zfpkm_df.to_csv("zfpkm.csv", index=True)
    zfpkm_plot(zfpkm_results, save_filepath="zfpkm_density.png")


if __name__ == "__main__":
    main()
```

## Results
## Expected zFPKM Distribution
The following figure shows the expected FPKM ('fpkm_density', in teal) and zFPKM ('fpkm_density_scaled', in salmon). The Gaussian curve is fit to the peak of the FPKM density distribution. Values > -3 can be marked as active.
![Expected zFPKM](figures/expected_density.png "Expected zFPKM")
<p align="center">Figure 1: The expected zFPKM Gaussian distribution overlaid on the FPKM distribution</p> 

## Actual zFPKM Distribution (from this package)
The following figure shows the calculated zFPKM from this package. Like Figure 1, the FPKM ('fpkm_density', in teal) and zFPKM ('fpkm_density_scaled', in salmon) are overlaid on the highest FPKM peak.

Example code showing how this graph was generated can be found in `examples/example_zfpkm.py`.


![Actual zFPKM](figures/actual_density.png "Actual zFPKM")
<p align="center">Figure 2: The actual zFPKM Gaussian distribution from this package overlaid on the FPKM distribution</p>


## Comparison with `scikit-learn` and `scipy`
Figure 3 and 4, below, were generated with `scikit-learn==1.7.2` and `scipy==1.16.3`, respsectively. These figures, while **very** similar to the expected and actual zFPKM distributions, have several noteable differences:
1. The maximum density is ~37% greater than the expected and actual zFPKM distributions (~0.131 expected vs ~0.180 for scikit-learn and scipy)
2. The left-hand of the FPKM density distribution is much smoother than the expected results; this comes specifically from a larger bandwidth value than the original R source. 

Taken together, this results in the final zFPKM scores using scikit-learn or scipy being similar to, but distinct enough, from the expected values to potentially cause problems in downstream analysis.

![scikit-learn zFPKM](figures/sklearn_density.png "scikit-learn zFPKM")
<p align="center">Figure 3: scikit-learn FPKM ('fpkm_density', in teal) & zFPKM ('fitted_density_scaled', in salmon) distribution</p>

![scipy zFPKM](figures/scipy_density.png "scipy zFPKM")
<p align="center">Figure 4: scipy FPKM ('fpkm_density', in teal) & zFPKM ('fitted_density_scaled', in salmon) distribution</p>
