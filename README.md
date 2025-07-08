# Code repository for Sorensen et al.

This is the code repository accompanying the manuscript [Sorensen et al.](https://www.biorxiv.org/content/10.1101/2023.11.25.568393v1), containing  analysis scripts, data files, file manifest, and figure generation code.

## Makefile

The example Makefile in the `figures` directory illustrates the inputs, outputs, and dependencies of most of the code in the repository. It is organized by figure, where each section contains the figure generation script along with additional scripts used to generate the required inputs. Note that some of the file path variables that refer to internal systems have placeholder values.

## File manifest

The file manifest contains links to specific data files for each specimen in the appropriate data archives. These can be associated with values in the processed data files in this repository by specimen ID.

The links for Patch-seq morphology SWC files are links to directories for each cell specimen that contain several versions of the morphological reconstructions (e.g., original orientation, upright orientation, upright and aligned to a common set of layer thicknesses).

## Level of support

This code is provided for reference purposes only. Parts of the code base rely on internal Allen Institute for Brain Science systems and are not expected to run without modification outside of those systems.
