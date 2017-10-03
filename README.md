# DBC

Discrimantive Bag-of-Cells (DBC) is a machine learning tool for predicting arbitrary variables of cellular images. We have developed this tool primarily for hematoxylin and eosin (H&E) stained images, though it could be applied to any cellular imaging modality. Variables of interest to predict could include clinical variables, such as patient outlook, or genomic variables, such as molecular subtype for breast cancer.

The code for the algorithm is contained in the folder 'DBoW', which is a more general machine learning tool for learning discrimantive bag-of-word representations of data. 

## Installation

To use the code, you must first install the 'dbow' module within the 'DBoW' folder, which is done by running
```
python setup.py install
```
from the command line from within the 'DBoW' folder. If you are installing globally, remember to use `sudo`.

### Requirements

Python libraries: numpy; pandas; tensorflow; opencv; matplotlib

## Preprocessing CellProfiler Features

This tool pairs particularly well with CellProfiler (http://cellprofiler.org/). CellProfiler can be used to first extract features of cells and nuclei from cellular images and DBC can then be used to predict image-level variables from the extracted features of each cell and/or nucleus.

In order to use features from CellProfiler, it is likely that some preprocessing is needed before using DBC. In particular, the CSV files created by CellProfiler may contain unnecessary columns (such as Image or Object ID) and some features, such as those of texture and intensity distribution, need to be made invariant to orientation, either by using instead their average or maximum value. The function `feature_set_modication()` in `preprocessing.py` in DBoW will load a list of features to keep, to average, and to maximize from three respective files and save the resulting features in new CSV files in a separate directory. Examples of such files are found in the 'cellprofiler' directory.

## Diagnosing Basal Subtype in TCGA-BRCA Patients

See 'example\_predict\_script.sh' for an example python script that predicts the subtype (Basal or non-Basal) of a sample TCGA-BRCA patient.

The necessary sample data is available here:
http://www.andrew.cmu.edu/user/bchidest/software/

Copy the extracted folder 'DBC\_sample\_data' into the root directory ('DBC').
