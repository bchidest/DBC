#!/bin/bash

python DBoW/scripts/predict_sample_script.py DBoNW_sample_data/basal_dbonw_model DBoNW_sample_data/tcga_basal_reference.pkl DBoNW_sample_data/TCGA-AN-A0FJ-01_Nuclei.csv --features_filename_suffix=-01_Nuclei --labels_filename DBoNW_sample_data/tcgaSubtype.csv
