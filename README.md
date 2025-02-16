# Syndrome of change
 This repository contains codes for the manuscript: 
 Li, W. et al. Diagnosing syndromes of biosphere-atmosphere-socioeconomic change.

 (i) data_collection.ipynb illustate that we generate a zarr file to combine different sources of biospheric and atmospheric data to a fixed 0.25˚, 8 daily gridded data;
 
 (ii) data_processing.ipynb works on different steps: aggregate gridded data to national scale --> gap-fill world bank socioeconomic data --> remove trends and country means --> run ridge-based canonical correlation analysis --> visualize explained covariance of CCA components --> test the robustness of CCA components by leave-out experiments;
 
 (iii) example_syndrome.ipynb draws some figures to illustrate user cases, such as global syndrome identification, extreme detection, etc.

# Note:
 (i) If you need to run the codes, please download the data.zip firest. Due to the size of data is large, we will make a repository of data at Zenodo Li, W. (2025). Data for Diagnosing Syndromes of Biosphere-Atmosphere-Socioeconomic Change [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14876723, which includes second-level data which can be used to reproduce all analyses described above;
 
 (ii) The raw datasets used to compute all analysis results are shared with public links in the paper.

**We are happy to answer your questions! Contact: Wantong Li (wantongli@berkeley.edu); Fabian Gans (fgans@bgc-jena.mpg.de)**

# Conda environment installation
Please use the syndrome.yml to set up the environment for runing provided codes.
The Linux command for environment installation: conda env create -f syndrome.yml

# References
