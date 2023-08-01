## PET-LAB
This repository contains scripts from the PET-LAB project - an outline of the files is provided below:

# CV-scatter
Generating stripplots (a type of scatter plot) for the coefficient of variation (CV) for radiomics features of different MRI sequences. The CV is a measure of dispersion of the data.
# CV-table
Generates a dataframe of the median CV from dataframes of CV for all patients and timepoints studied (i.e. gets the median across timepoints).
# ICC-heatmap
Plots heatmaps of intra-class correlation coefficients for radiomics features, for different radiomics classes (e.g. first order, shape-based, etc).
# ME_image_generation
Code for generating Maximum Enhancement (ME) images from DCE image series.
# Radiomics
Runs radiomics analysis on Before, During, and After-PST (Primary Systemic Therapy) DCE-MRI (ME), T1w-MRI (DCE pre-contrast), T2w-MRI, and ADC (DWI-MRI) images, including isotropic voxel resampling, z-score normalisation (only for T1w and T2w images), grey-level discretisation, and cropping to speed up processing. Note that the z-score normalisation does not work well for T2w images (unresolved issue).
# Stats-across-timepoints
Calculate statistics on radiomics features across timepoints. Statistics include CV, wCV (within-subject CV from QIBA paper), relative difference from baseline and assessment of normality of relative differences using Shapiro-Wilk test, mean and standard deviation from the relative differences, and also calculate the intra-class correlation coefficient (ICC).
QIBA paper: van Houdt PJ, Saeed H, Thorwarth D, Fuller CD, Hall WA, McDonald BA, Shukla-Dave A, Kooreman ES, Philippens ME, van Lier AL, Keesman R. Integration of quantitative imaging biomarkers in clinical trials for MR-guided radiotherapy: Conceptual guidance for multicentre studies from the MR-Linac Consortium Imaging Biomarker Working Group. European Journal of Cancer. 2021 Aug 1;153:64-71.
# Stats-within-effects
Calculate statistics comparing between the different bin counts/normalisations rather than different time points (choose 1 time point) (CV and ICC). It probably makes more sense to look for changes across timepoints, as these will reflect biological/physical changes.
# T2w_breast_mask_registration
Code to register T2w images (timepoint 1) to timepoint 2 and 3, and then after registering the images, code to propagate the transforms (on the masks). After the transforms are propagated, split the masks into left and right (L and R) breasts.
# Tumour_delta
Obtain delta radiomics for radiomics of tumours.
# Convert_and_process_data.ipynb
Code to process DICOMM contours to NIFTI (written by Rob Finnegan).
# erode_masks.py
Code to erode the breast masks used for whole-breast radiomics (to remove suspected skin and muscle tissue from the analysis).
# prepare_dicom.ipynb
Code to prepare dicom files for conversion to nifti by convert_and_process_data script. Code incomplete.
# test_df
Test file for concatenating dataframes in pandas.
# testing
Testing resampling code for different MRI sequences.
# wes_014_tumour_code
Code to mask out second tumour for WES_014 patient tumour contour data.
