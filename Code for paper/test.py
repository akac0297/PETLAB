#!/usr/bin/env python3
# coding: utf-8

import SimpleITK as sitk
import numpy as np

#breast_mask_1=sitk.ReadImage("/home/alicja/PET-LAB Code/PET-LAB/PET segmentation/PET Breast Masks/PET_plan_breast_seg_16_1.nii.gz")
#PET_img_2=sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/WES_016/IMAGES/WES_016_TIMEPOINT_2_PET.nii.gz")
#PET_img_3=sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/WES_016/IMAGES/WES_016_TIMEPOINT_3_PET.nii.gz")

#breast_mask_2=sitk.Resample(breast_mask_1,PET_img_2,sitk.Transform(),sitk.sitkNearestNeighbor)
#breast_mask_3=sitk.Resample(breast_mask_1,PET_img_3,sitk.Transform(),sitk.sitkNearestNeighbor)

#sitk.WriteImage(breast_mask_2,"/home/alicja/PET-LAB Code/PET-LAB/PET segmentation/PET Breast Masks/PET_plan_breast_seg_16_2.nii.gz")
#sitk.WriteImage(breast_mask_3,"/home/alicja/PET-LAB Code/PET-LAB/PET segmentation/PET Breast Masks/PET_plan_breast_seg_16_3.nii.gz")

breast_mask_1=sitk.ReadImage("/home/alicja/PET-LAB Code/PET-LAB/PET segmentation/PET breast contours/breast_contour_dilate_14_0.nii.gz")
breast_mask_1=sitk.BinaryDilate(breast_mask_1,(2,2,2))
arr_1=sitk.GetArrayFromImage(breast_mask_1)
arr_1[:,243:,:]=0
mask_1=sitk.GetImageFromArray(arr_1)
mask_1.CopyInformation(breast_mask_1)

#sitk.WriteImage(mask_1,"/home/alicja/PET-LAB Code/PET-LAB/PET segmentation/PET breast contours/breast_contour_dilate_14_0_new.nii.gz")

breast_mask_2=sitk.ReadImage("/home/alicja/PET-LAB Code/PET-LAB/PET segmentation/PET breast contours/breast_contour_dilate_14_1.nii.gz")
#breast_mask_2=sitk.BinaryDilate(breast_mask_2,(2,2,2))
arr_2=sitk.GetArrayFromImage(breast_mask_2)
arr_2[:,235:,:]=0
mask_2=sitk.GetImageFromArray(arr_2)
mask_2.CopyInformation(breast_mask_2)

#sitk.WriteImage(mask_2,"/home/alicja/PET-LAB Code/PET-LAB/PET segmentation/PET breast contours/breast_contour_dilate_14_1_new.nii.gz")

breast_mask_3=sitk.ReadImage("/home/alicja/PET-LAB Code/PET-LAB/PET segmentation/PET breast contours/breast_contour_dilate_14_2.nii.gz")
breast_mask_3=sitk.BinaryDilate(breast_mask_3,(2,2,2))
arr_3=sitk.GetArrayFromImage(breast_mask_3)
arr_3[:,240:,:]=0
mask_3=sitk.GetImageFromArray(arr_3)
mask_3.CopyInformation(breast_mask_3)

sitk.WriteImage(mask_3,"/home/alicja/PET-LAB Code/PET-LAB/PET segmentation/PET breast contours/breast_contour_dilate_14_2_new.nii.gz")