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

from platipy.imaging.registration.linear import linear_registration
from platipy.imaging.registration.deformable import fast_symmetric_forces_demons_registration
from platipy.imaging.registration.utils import apply_transform
from platipy.imaging.registration.utils import apply_linear_transform

patient_no="04"
timepoint="1"

image_ct_0=sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/WES_004/IMAGES/WES_004_TIMEPOINT_1_CT_AC.nii.gz")
image_pt_0_raw=sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/WES_004/IMAGES/WES_004_TIMEPOINT_1_PET.nii.gz")
image_ct_plan = sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/WES_004/IMAGES/WES_004_CT_RTSIM.nii.gz")
contour_breast_plan = sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/WES_004/LABELS/WES_004_RTSIM_LABEL_CHESTWALL_LT_CTV.nii.gz")

image_ct_plan_arr=sitk.GetArrayFromImage(image_ct_plan)
image_ct_plan_arr[image_ct_plan_arr.shape[0]//5*4:,:,:]=-1000
image_ct_plan_new=sitk.GetImageFromArray(image_ct_plan_arr)
image_ct_plan_new.CopyInformation(image_ct_plan)
image_ct_plan=image_ct_plan_new

image_ct_0_arr=sitk.GetArrayFromImage(image_ct_0)
image_ct_0_arr[:image_ct_0_arr.shape[0]//2,:,:]=-1000
image_ct_0_new=sitk.GetImageFromArray(image_ct_0_arr)
image_ct_0_new.CopyInformation(image_ct_0)
image_ct_0_raw=sitk.Resample(image_ct_0_new,image_ct_0)

image_ct_0=sitk.Resample(image_ct_0_raw, image_ct_0)

#register planning CT to CT
image_ct_plan_to_0_rigid, tfm_plan_to_0_rigid = linear_registration(
    image_ct_0,
    image_ct_plan,
    shrink_factors = [8,4],
    smooth_sigmas = [0,0],
    sampling_rate = 0.5,
    final_interp = 2,
    metric = 'correlation',
    optimiser = 'gradient_descent_line_search',
    number_of_iterations = 25,
    reg_method='Rigid')

vis = ImageVisualiser(image_ct_0, cut=[160,260,256], window=[-250, 500])
vis.add_comparison_overlay(image_ct_plan_to_0_rigid)
fig = vis.show()
fig.savefig(f"/home/alicja/PET_LAB_PROCESSED/PET_rigid.jpeg",dpi=400)

image_ct_plan_to_0_dir, tfm_plan_to_0_dir, def_field = fast_symmetric_forces_demons_registration(
    image_ct_0,
    image_ct_plan_to_0_rigid,
    resolution_staging=[4,2],
    iteration_staging=[10,10]
)

vis = ImageVisualiser(image_ct_0, cut=[160,260,256], window=[-250, 500])
vis.add_comparison_overlay(image_ct_plan_to_0_dir)
fig = vis.show()
fig.savefig(f"/home/alicja/PET_LAB_PROCESSED/PET_dir.jpeg",dpi=400)

#register breast structure to CT
contour_breast_plan_to_0_rigid = apply_transform(
    contour_breast_plan,
    image_ct_0,
    tfm_plan_to_0_rigid
)

contour_breast_plan_to_0_dir = apply_transform(
    contour_breast_plan_to_0_rigid,
    image_ct_0,
    tfm_plan_to_0_dir
)


image_pt_0=sitk.Resample(image_pt_0_raw,image_ct_0)

vis = ImageVisualiser(image_ct_0, cut=get_com(image_ct_0), window=[-250, 500])
vis.add_scalar_overlay(image_pt_0, name='PET SUV', colormap=plt.cm.magma, min_value=0.1, max_value=10000)
vis.add_contour(contour_breast_plan_to_0_dir,name='R BREAST', color='g')
fig = vis.show()
fig.savefig(f"/home/alicja/PET_LAB_PROCESSED/PET_transformed_contour.jpeg",dpi=400)

contour_breast_plan_to_0_dir_arr=sitk.GetArrayFromImage(contour_breast_plan_to_0_dir)
contour_breast_plan_to_0_dir_arr[196:,:,:]=0
contour_breast_plan_to_0_dir2=sitk.GetImageFromArray(contour_breast_plan_to_0_dir_arr)
contour_breast_plan_to_0_dir2.CopyInformation(contour_breast_plan_to_0_dir)
contour_breast_plan_to_0_dir=contour_breast_plan_to_0_dir2

vis = ImageVisualiser(image_ct_0, axis='z', cut=180, window=[-250, 500])
vis.add_scalar_overlay(image_pt_0, name='PET SUV', colormap=plt.cm.magma, min_value=0.1, max_value=10000)
vis.add_contour(contour_breast_plan_to_0_dir, name='R BREAST', color='g')
fig = vis.show()
fig.savefig(f"/home/alicja/PET_LAB_PROCESSED/PET_masked_contour.jpeg",dpi=400)

sitk.WriteImage(contour_breast_plan_to_0_dir,"/home/alicja/PET_LAB_PROCESSED/contour_breast_plan_PET_004.nii.gz")

##use structure information for breast to mask out all but the breast area (create new array with everything but
##this area set to 0)
#masked_pet_breast = sitk.Mask(image_pt_0, contour_breast_plan_to_0_dir)

#sitk.WriteImage(masked_pet_breast, "masked_pet_breast_WES_0" + patient_no + "_" + timepoint + ".nii.gz")

##get 95th percentile, then mask the breast volume
#masked_pet_breast=sitk.Resample(masked_pet_breast, image_pt_0_raw)
#mask_arr=sitk.GetArrayFromImage(masked_pet_breast)
#mask_arr=mask_arr.flatten() 

#p = np.percentile(mask_arr[mask_arr>0], 97)
#print(p)

#tum = sitk.Mask(image_pt_0_raw, masked_pet_breast>p)

#sitk.WriteImage(tum, "pet_seg_0"+patient_no+"_"+timepoint+"_97pc.nii.gz")