# DBoNW

Discrimantive Bag-of-Nuclear-Words (DBoNW) is a machine learning tool for predicting arbitrary variables of cellular images. We have developed this tool primarily for hematoxylin and eosin (H&E) stained images, though it could be applied to any cellular imaging modality. Variables of interest to predict could include clinical variables, such as patient outlook, or genomic variables, such as molecular subtype for breast cancer.

The code for the algorithm is contained in the folder 'DBoW', which is a more general machine learning tool for learning discrimantive bag-of-word representations of data. 

## Diagnosing Basal Subtype in TCGA-BRCA Patients

See 'example\_predict\_script.py' for an example python script that predicts the subtype (Basal or non-Basal) of a sample TCGA-BRCA patient.

The necessary sample data is available here:
http://www.andrew.cmu.edu/user/bchidest/software/

Copy the extracted folder 'DBONW\_sample\_data' into the root directory ('DBoNW').
