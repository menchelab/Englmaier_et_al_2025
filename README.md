[![DOI](https://zenodo.org/badge/1009427048.svg)](https://doi.org/10.5281/zenodo.16561318)

# Englmaier_et_al_2025
A repository for code associated with codonbias analyses presented in the manuscript

# Installation and Usage
In theory it is as easy as generating the conda environment from the `environment.yml` file like
```bash
conda env create -f environment.yml
```
installing the ipykernel
```bash
conda activate codons
python -m ipykernel install --user --name codons --display-name codons
```
and then simply running `jupyter lab` to run the notebooks. All presented analyses should be reproducable
with the code contained in the respective jupyter notebooks. Most of the code concerning the computation of
the codon biases for coding sequences are contained in `notebooks/codontools` and work as a self contained Python package.
The codontools themselves contain code to compute a variety of different codon bias measures from simple percentage of coding sequence
to clusters of codons including some functionality to visualize the clusters.
