# Code repository for Sorensen et al.

This is the code repository accompanying the manuscript [Sorensen et al.](https://www.biorxiv.org/content/10.1101/2023.11.25.568393v1), containing  analysis scripts, data files, file manifest, and figure generation code.

## Makefile

The example Makefile in the `figures` directory illustrates the inputs, outputs, and dependencies of most of the code in the repository. It is organized by figure, where each section contains the figure generation script along with additional scripts used to generate the required inputs. Note that some of the file path variables that refer to internal systems have placeholder values.

## File manifest

The file manifest contains links to specific data files for each specimen in the appropriate data archives. These can be associated with values in the processed data files in this repository by specimen ID.

The links for Patch-seq morphology SWC files are links to directories for each cell specimen that contain several versions of the morphological reconstructions (e.g., original orientation, upright orientation, upright and aligned to a common set of layer thicknesses).

## Level of support

This code is provided for reference purposes only. Parts of the code base rely on internal Allen Institute for Brain Science systems and are not expected to run without modification outside of those systems.

## System requirements

The code has been run using Python 3.9 and R 4.3 on a Rocky Linux 8.8 workstation. Software dependencies include:

Python packages:
- h5py 3.14.0
- matplotlib 3.9.2
- numpy 1.26.4
- pandas 2.3.3
- seaborn 0.13.2
- argschema 3.0.4
- allensdk 2.16.2
- ipfx 1.0.2
- scipy 1.13.1
- scikit-posthocs 0.11.4
- ccf-streamlines 1.1.4
- shapely 2.0.6
- skeleton_keys 0.1.1
- neuron_morphology 1.2.2
- tqdm 4.67.1
- scikit-image 0.24.0
- pynrrd 1.1.3
- umap-learn 0.5.3
- adjusttext 1.3.0
- scikit-learn 1.6.1
- feather-format 0.4.1
- drcme 0.1.0
- python-igraph 0.11.9
- leidenalg 0.10.2
- networkx 3.2.1

R packages:
- scrattch.hicat 1.0.0
- arrow 14.0.2
- matrixStats
- Matrix 1.6-1.1
- WGCNA 1.73
- dynamicTreeCut 1.63-1
- tibble 3.2.1
- dplyr 2.3.4
- glmnet 4.1-8
- progress 1.2.2
- readr 2.1.4
- purrr 1.0.2
- MuMIn 1.47.5
- rhdf5 2.46.1
- rjson 0.2.21
- limma 3.58.1
- data.table 1.14.8
- caret 6.0-94

The code has not been comprehensively tested beyond the versions listed here.

## Installation

To install the software, clone this repository and install the requirements listed above. It is recommended to use an environment manager such as conda to install the packages and their associated dependencies. Installing the repository itself should take at most a few minutes; installing all the required packages (if starting from scratch) may take an hour or two.

## Instructions for Use / Demo

The example Makefile can be used to generate the analysis results and figures presented in the manuscript. The provided example Makefile (`Makefile.example`) should be renamed to `Makefile` before running the commands below. Depending on the particular desired script/output, some additional files may need to be downloaded and paths adjusted (note that some of the paths refer to internal systems that would be difficult to reproduce). To generate a figure, run a command such as:
```
make fig_ephys_supplement_it.pdf
```
The expected output is a PDF file that matches the corresponding figure in the manuscript. Running the scripts that generate PDF figures typically take a few minutes; running some of the underlying analysis steps (such as the cross-validated spare reduced-rank regression) can take several hours.




