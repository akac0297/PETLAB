#!/usr/bin/env python
# coding: utf-8

"""
Need to save the final breast contour and also images of the breast contour and contour dilate

"""

#import modules
import SimpleITK as sitk

from platipy.imaging import ImageVisualiser
from platipy.imaging.label.utils import get_com

from platipy.imaging.registration.linear import linear_registration
from platipy.imaging.registration.deformable import fast_symmetric_forces_demons_registration
from platipy.imaging.registration.utils import apply_transform
from platipy.imaging.registration.utils import apply_linear_transform

breast=sitk.ReadImage("/home/alicja//Downloads/PET-LAB Nifti files/R-breast.nii.gz") #right breast
#Left breast=sitk.ReadImage("contralateral_segmentation.nii.gz")

pat_no="07"
timept="4"

WES_1_B50T=sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/WES_007/IMAGES/WES_007_TIMEPOINT_1_MRI_DWI_B50.nii.gz")
#WES_1_B800T=sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/WES_007/IMAGES/")
#WES_1_T2w=sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/WES_007/IMAGES/")
#WES_1_MPE=sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/WES_007/IMAGES/")

WES_010_4_B50T=sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/WES_010/IMAGES/WES_010_TIMEPOINT_1_MRI_DWI_B50.nii.gz")

#image_to_0_rigid, tfm_to_0_rigid = linear_registration(
#    WES_1_B50T,
#    WES_010_4_B50T,
#    shrink_factors= [8,4],
#    smooth_sigmas= [0,0],
#    sampling_rate= 0.5,
#    final_interp= 2,
#    metric= 'mean_squares',
#    optimiser= 'gradient_descent_line_search',
#    number_of_iterations= 25,
#    reg_method='Rigid')

#image_to_0_dir, tfm_to_0_dir, def_field = fast_symmetric_forces_demons_registration(
#    WES_1_B50T,
#    image_to_0_rigid,
#    resolution_staging=[4,2],
#    iteration_staging=[10,10]
#)

#breast_to_0_rigid = apply_linear_transform(
#    breast,
#    WES_1_B50T,
#    tfm_to_0_rigid
#)

#breast_to_0_dir = apply_transform(
#    breast_to_0_rigid,
#    WES_1_B50T,
#    tfm_to_0_dir
#)

#vis = ImageVisualiser(WES_1_B50T, axis='z', cut=get_com(breast_to_0_dir), window=[-50, 700])
#vis.add_contour(breast_to_0_dir, name='BREAST', color='g')
#fig = vis.show()

#fig.savefig(f"/home/alicja/PET_LAB_PROCESSED/dir_contour.jpeg",dpi=400)

#breast_contour_dilate=sitk.BinaryDilate(breast_to_0_dir, (2,2,2))

#vis = ImageVisualiser(WES_1_B50T, axis='z', cut=get_com(breast_to_0_dir), window=[-50, 700])
#vis.add_contour(breast_contour_dilate, name='BREAST', color='g')
#fig = vis.show()

#fig.savefig(f"/home/alicja/PET_LAB_PROCESSED/dir_dilate_contour.jpeg",dpi=400)

#sitk.WriteImage(breast_contour_dilate,"/home/alicja/PET_LAB_PROCESSED/breast_contour_dilate_007_4.nii.gz")
breast_contour_dilate=sitk.ReadImage("/home/alicja/PET_LAB_PROCESSED/breast_contour_dilate_007_4.nii.gz")

masked_breast = sitk.Mask(WES_1_B50T, breast_contour_dilate)

#vis = ImageVisualiser(masked_breast, axis='z', cut=get_com(masked_breast), window=[-50, 700])
#fig = vis.show()
#fig.savefig(f"/home/alicja/PET_LAB_PROCESSED/masked_breast.jpeg",dpi=400)

def estimate_tumour_vol(img_mri, lowerthreshold=300, upperthreshold=5000, hole_size=1):
    label_threshold = sitk.BinaryThreshold(img_mri, lowerThreshold=lowerthreshold, upperThreshold=upperthreshold)
    sitk.WriteImage(label_threshold,"/home/alicja/PET_LAB_PROCESSED/label_threshold.nii.gz")
    label_threshold_cc = sitk.RelabelComponent(sitk.ConnectedComponent(label_threshold))
    label_threshold_cc_x = (label_threshold_cc==1)
    sitk.WriteImage(label_threshold,"/home/alicja/PET_LAB_PROCESSED/connected_breast.nii.gz")
    label_threshold_cc_x_f = sitk.BinaryMorphologicalClosing(label_threshold_cc_x, (hole_size,hole_size,hole_size))
    return(label_threshold_cc_x_f)

image_mri=WES_1_B50T
arr_mri = sitk.GetArrayFromImage(image_mri)
arr_mri[:,:,arr_mri.shape[2]//2:] = 0 #if laterality is RIGHT
image_mri_masked=sitk.GetImageFromArray(arr_mri)
image_mri_masked.CopyInformation(image_mri)

label_threshold_cc_x_f=estimate_tumour_vol(image_mri_masked, lowerthreshold=900, upperthreshold=5000, hole_size=1)

sitk.WriteImage(label_threshold_cc_x_f,"/home/alicja/PET_LAB_PROCESSED/WES_007_tumour_seg.nii.gz")
