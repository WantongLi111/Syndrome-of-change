# Syndrome of change
 This repository contains codes and data for the manuscript: 
 Li, W. et al. Diagnosing syndromes of biosphere-atmosphere-socioeconomic change.

 (i) data_collection.ipynb illustate that we generate a zarr file to combine different sources of biospheric and atmospheric data to a fixed 0.25˚, 8 daily gridded data;
 (ii) data_processing.ipynb works on different steps: aggregate gridded data to national scale --> gap-fill world bank socioeconomic data --> remove trends and country means --> run ridge-based canonical correlation analysis --> visualize explained covariance of CCA components --> test the robustness of CCA components by leave-out experiments
 (iii) example_syndrome.ipynb draws some figures to illustrate user cases, such as global syndrome identification, extreme detection, etc.

# Note:
 (i) We will make a repository version at Zenodo ..., including second-level data which can be used to reproduce all analyses described above;
 (ii) The raw datasets used to compute all analysis results are shared with public links in the paper.

**We are happy to answer your questions! Contact: Wantong Li (wantongli@berkeley.edu); Fabian Gans (fgans@bgc-jena.mpg.de)**

# Conda environment installation
Please use the syndrome.yml to set up the environment for runing provided codes.
The Linux command for environment installation: conda env create -f syndrome.yml

# References