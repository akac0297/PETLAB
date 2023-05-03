#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import SimpleITK as sitk

img_1 = sitk.ReadImage("/home/alicja/PET-LAB Code/PET-LAB/PETLAB_CONTOUR_PROCESSED/PROCESSED/WES_014/STRUCTURES/WES_014_TIMEPOINT_1_GTV.nii.gz")
arr_1 = sitk.GetArrayFromImage(img_1)
arr_1[33:,:,:]=0
img_1_processed = sitk.GetImageFromArray(arr_1)
img_1_processed.CopyInformation(img_1)
sitk.WriteImage(img_1_processed,"/home/alicja/PET-LAB Code/PET-LAB/PETLAB_CONTOUR_PROCESSED/PROCESSED/WES_014/STRUCTURES/WES_014_TIMEPOINT_1_GTV_new.nii.gz")

img_2 = sitk.ReadImage("/home/alicja/PET-LAB Code/PET-LAB/PETLAB_CONTOUR_PROCESSED/PROCESSED/WES_014/STRUCTURES/WES_014_TIMEPOINT_2_GTV.nii.gz")

arr_2 = sitk.GetArrayFromImage(img_2)
arr_2[33:,:,:]=0
img_2_processed = sitk.GetImageFromArray(arr_2)
img_2_processed.CopyInformation(img_2)
sitk.WriteImage(img_2_processed,"/home/alicja/PET-LAB Code/PET-LAB/PETLAB_CONTOUR_PROCESSED/PROCESSED/WES_014/STRUCTURES/WES_014_TIMEPOINT_2_GTV_new.nii.gz")

img_3 = sitk.ReadImage("/home/alicja/PET-LAB Code/PET-LAB/PETLAB_CONTOUR_PROCESSED/PROCESSED/WES_014/STRUCTURES/WES_014_TIMEPOINT_3_GTV.nii.gz")

arr_3 = sitk.GetArrayFromImage(img_3)
arr_3[33:,:,:]=0
img_3_processed = sitk.GetImageFromArray(arr_3)
img_3_processed.CopyInformation(img_3)
sitk.WriteImage(img_3_processed,"/home/alicja/PET-LAB Code/PET-LAB/PETLAB_CONTOUR_PROCESSED/PROCESSED/WES_014/STRUCTURES/WES_014_TIMEPOINT_3_GTV_new.nii.gz")

# I have renamed the original contours to "....GTV_original.nii.gz" and changed the processed images to _GTV.nii.gz