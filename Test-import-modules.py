#!/usr/bin/env python3
# coding: utf-8

import SimpleITK as sitk
import numpy as np

x=np.array([3,4,5])

print(np.max(x))

path="/home/alicja/"
filename=path+"PET_LAB_PROCESSED/WES_004/IMAGES/WES_004_TIMEPOINT_1_MRI_DWI_ADC.nii.gz"
print(filename)
#test=sitk.ReadImage(filename)

#print(test)