#!/usr/bin/env python
# coding: utf-8
"""
Import modules
"""
import numpy as np

import SimpleITK as sitk
import matplotlib.pyplot as plt

from platipy.imaging import ImageVisualiser
from platipy.imaging.label.utils import get_com

patient_no="04"
timepoint="1"

#image_ct_0=sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/WES_004/IMAGES/WES_004_TIMEPOINT_1_CT_AC.nii.gz")
#image_pt_0_raw=sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/WES_004/IMAGES/WES_004_TIMEPOINT_1_PET.nii.gz")
image_ct_plan = sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/WES_007/IMAGES/WES_007_CT_RTSIM.nii.gz")
contour_breast_plan = sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/WES_007/LABELS/WES_007_RTSIM_LABEL_WHOLE_BREAST_CTV.nii.gz")

#image_ct_plan_arr=sitk.GetArrayFromImage(image_ct_plan)
#image_ct_plan_arr[image_ct_plan_arr.shape[0]//5*3:,:,:]=-1000
#image_ct_plan_new=sitk.GetImageFromArray(image_ct_plan_arr)
#image_ct_plan_new.CopyInformation(image_ct_plan)
#image_ct_plan=image_ct_plan_new

vis = ImageVisualiser(image_ct_plan, cut=get_com(image_ct_plan), window=[-250, 500])
vis.add_contour(contour_breast_plan)
fig = vis.show()
fig.savefig(f"/home/alicja/PET_LAB_PROCESSED/test_planning_CT.jpeg",dpi=400)

#test_pet=sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/WES_012/IMAGES/WES_012_TIMEPOINT_1_PET.nii.gz")
#test_pet_array=sitk.GetArrayFromImage(test_pet)
#print("Min SUV: ", np.min(test_pet_array))
#print("Max SUV:", np.max(test_pet_array))