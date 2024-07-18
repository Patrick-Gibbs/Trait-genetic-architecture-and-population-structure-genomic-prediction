# Trait Genetic Architecture and Population Structure: Genomic Prediction

Code used for the paper "Trait genetic architecture and population structure determine model selection for genomic prediction in natural Arabidopsis thaliana populations."

The 'Analysis' notebook contains the code used to generate all the figures and cited results in the paper and supplementary materials. The Main module contains all the code to generate the raw results, and the 'data' directory contains the genotypes, phenotypes, and temporary files used in this study.

The results of this study were generated on a virtual machine with 24 cores and 1TB of memory (although ~100GB would be sufficient); most results could be generated within 24 hours of compute time.

## Setup
The code for this project is almost entirely in Python. The attached YAML file contains all the dependencies and can be loaded into a conda environment using the following command:
In addition, `plink 1.9 (Stable (beta 7.2, 11 Dec 2023))` is used. The codebase calls plink using the alias 'plink'. If plink is not installed on your machine, download it from [here](https://www.cog-genomics.org/plink/). Alternatively, set `PATH_TO_PLINK` on `line 17` of `Main/HelperClasses/GetAraData` to the installed plink path.

All genotypes and phenotypes are publicly available and can be added to the 'Main/data' directory by running:

```
python3 -m Main.Setup.setup
```

## Description of Code
The codebase has 4 major components:

1. **data**: A directory that stores all the genotypes, phenotypes, and temporary files (e.g., principal components of the genotype data).

2. **HelperClasses**: These are classes used to run the experiments. Specifically, `HelperClasses.GetAraData` is a getter object which retrieves data used in the experiments. Example usage:
    ```python
    from Main.HelperClasses.GetAraData import GetAraData
    araData = GetAraData(path_to_data='./data', maf=0.05, window_kb=200, r2=0.6)
    y = araData.get_normalized_phenotype('study_12_FT10')
    X = araData.get_genotype('study_12_FT10')
    ```
    This creates a getter object linked to the data and retrieves phenotypes (`study_12_FT10`) and corresponding genotypes with specified genome filtering parameters.

3. **MeasurePerformance**: Used for testing models, contains the `Result10fold` object. Example usage:
    ```python
    from sklearn.linear_model import RidgeCV
    from Main.HelperClasses.MeasurePerformance import Result_10fold
    result = Result_10fold(RidgeCV(), X, y, important_parameters=rf_params, name='Ridge', cv=10)
    df = result.make_results_objects_into_csv([result])
    df.to_csv('Ridge_result_study_12_FT10.csv')
    ```
    This tests Ridge Regression on data `X`, `y` using 10-fold cross-validation and saves results to a CSV file.

4. **Code**: Contains code that uses helper functions to carry out each experiment discussed in the paper.

    - `global_search.py`: Tests linear regression and random forest for every trait downloaded (~550). Example call:
      ```
      python3 -m Main.Code.global_search
      ```
      Note: Fitting models across all traits takes about 1 week; testing one trait takes about an hour.

    - `gwas.py`: Runs GWAS using the `vcftogwas` Python package for each chosen trait.

    - `mlp_pca.py`, `rf_pca.py`, and `linear_pca.py`: Fit each chosen trait using principal components and generate results for figure 3b.

    - `pc_compare.py`: Fits each model to principal components explaining 30% of genomic variance and generates results for figure 2b.

    - `mlp_lasso.py`: Fits the sparse MLP model for each trait as described in Methods 2.3.2.

    These files rely on `results/traits_used.csv`.
