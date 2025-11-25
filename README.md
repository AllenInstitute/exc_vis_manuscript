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
- h5py
- matplotlib
- numpy
- pandas
- seaborn
- argschema
- allensdk
- ipfx
- scipy
- scikit_posthocs
- ccf_streamlines
- shapely
- skeleton_keys
- neuron_morphology
- tqdm
- scikit-image
- nrrd
- umap-learn
- adjustText
- scikit-learn
- feather
- drcme
- igraph
- leidenalg
- networkx

R packages:
- scrattch.hicat
- arrow
- matrixStats
- Matrix
- WGCNA
- dynamicTreeCut
- tibble
- dplyr
- glmnet
- progress
- readr
- purrr
- MuMIn
- rhdf5
- rjson
- limma
- data.table

The code has not been tested beyond the versions listed here.

## Installation

To install the software, clone this repository and install the requirements listed above. It is recommended to use an environment manager such as conda to install the packages and their associated dependencies. Installing the repository itself should take at most a few minutes; installing all the required packages (if starting from scratch) may take an hour or two.

## Instructions for Use / Demo

The example Makefile can be used to generate the analysis results and figures presented in the manuscript. The provided example Makefile (`Makefile.example`) should be renamed to `Makefile` before running the commands below. Depending on the particular desired script/output, some additional files may need to be downloaded and paths adjusted (note that some of the paths refer to internal systems that would be difficult to reproduce). To generate a figure, run a command such as:
```
make fig_ephys_supplement_it.pdf
```
The expected output is a PDF file that matches the corresponding figure in the manuscript. Running the scripts that generate PDF figures typically take a few minutes; running some of the underlying analysis steps (such as the cross-validated spare reduced-rank regression) can take several hours.




