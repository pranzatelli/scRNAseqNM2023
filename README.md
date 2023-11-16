# scRNAseqNM2023

## Software Requirements

All analysis was done on a computer server running Rocky Linux 8.7 (Green Obsidian). This code is not supported on any other Linux distribution or on Windows or Mac. BBKNN was run on an NVIDIA V100-SXM2 with CUDA 11.4.4.

### Single-Cell RNA-Sequencing
Analysis of the single-cell RNA-sequencing was done in Python 3.9.4 using
```
numpy 1.21.2
scipy 1.6.3
statsmodels 0.12.2
scikit-learn 0.24.2
pandas 1.2.4
anndata 0.8.0
scanpy 1.7.2
matplotlib 3.4.2
seaborn 0.11.1
```
### Spatial RNA-Sequencing
The spatial RNA-sequencing data was analyzed in Python 3.9.7 using 
```
numpy 1.21.5
scipy 1.8.0
statsmodels 0.13.2
pandas 1.4.1
networkx 2.7.1
scanpy 1.8.2
matplotlib 3.5.1
seaborn 0.11.2
```
## Instructions for Use

Download the software from this github repository by typing
```
git clone https://github.com/pranzatelli/scRNAseqNM2023.git
```
into the command line. Download time should be negligible.

## Demo

To run the code on the data, simply load the data as `adataSG` using `sc.read()` and proceed. For this toy example, the expected output will be a UMAP that resembles a scatterplot and no significantly differentially expressed genes. Run time should be around 30 minutes.
