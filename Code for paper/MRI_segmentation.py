#!/usr/bin/env python3
# coding: utf-8

# MRI_segmentation.py
# Semi-automatically obtain tumour masks from DCE post-contrast 1 images and breast masks
# Run code in terminal and use Slicer software to check manually inputted parameters and confirm final tumour mask
# to-do: Test on 2 patients (3 time points) DCE post-contrast 1 images. Clean up code again.

import SimpleITK as sitk
from platipy.imaging.label.utils import get_com
from platipy.imaging.visualisation.visualiser import ImageVisualiser

# input parameters
patient_id="023"
timepoint="3"
laterality = "L"
img_filepath=f"/home/alicja/PET_LAB_PROCESSED/WES_{patient_id}/IMAGES/"
breast_mask_filepath=f"/home/alicja/PET-LAB Code/PET-LAB/MRI segmentation/PETLAB Breast Masks/T2W Breast masks/"
tumour_mask_filepath=f"/home/alicja/PET-LAB Code/PET-LAB/MRI segmentation/DCE_MRI_tumour_masks/"
plots_path=f"/home/alicja/PET-LAB Code/PET-LAB/MRI segmentation/DCE_tumour_mask_plots/"

# read in DCE image and breast mask
img_name=f"WES_{patient_id}_TIMEPOINT_{timepoint}_MRI_T1W_DCE_ACQ_1.nii.gz"
mask_name=f"WES_{patient_id}_TIMEPOINT_{timepoint}_T2W_EDIT_{laterality}_breast.nii.gz"

DCE_img=sitk.ReadImage(img_filepath+img_name)
breast_mask=sitk.ReadImage(breast_mask_filepath+mask_name)

vis = ImageVisualiser(DCE_img, axis='z', cut=get_com(breast_mask), window=[-250, 500])
vis.add_contour(breast_mask, name='BREAST', color='g')
fig = vis.show()
fig.savefig(plots_path+f"WES_{patient_id}_TIMEPOINT_{timepoint}_breast_mask_over_DCE_img.jpeg",dpi=400)

# erode breast mask if necessary (bright skin artefacts) - use breast mask with DCE image in Slicer software to manually choose threshold
def erodeBreastContour(breast_mask,x=3,y=3,z=3):
    breast_mask_erode=sitk.BinaryErode(breast_mask, (x,y,z))
    breast_mask_arr=sitk.GetArrayFromImage(breast_mask_erode)
    #breast_mask_arr[:50,:,:]=0
    #breast_mask_arr[:5,:,:]=0
    #breast_mask_arr[20:,:,:]=0
    #breast_mask_arr[:,:165,:]=0
    #breast_mask_arr[:,230:,:]=0
    #breast_mask_arr[:,200:,:]=0
    #breast_mask_arr[:,240:,:]=0
    #breast_mask_arr[:,180:,:]=0
    breast_mask_erode=sitk.GetImageFromArray(breast_mask_arr)
    breast_mask_erode.CopyInformation(breast_mask)
    vis = ImageVisualiser(DCE_img, axis='z', cut=get_com(breast_mask_erode), window=[-250, 500])
    vis.add_contour(breast_mask_erode, name='BREAST', color='g')
    fig = vis.show()
    fig.savefig(plots_path+f"WES_{patient_id}_TIMEPOINT_{timepoint}_eroded_breast_mask_over_DCE_img.jpeg",dpi=400)
    return(breast_mask_erode)

breast_mask=erodeBreastContour(breast_mask,2,2,2)

# mask breast in DCE img
breast_mask=sitk.Resample(breast_mask,DCE_img)
masked_breast=sitk.Mask(DCE_img,breast_mask)

# estimate tumour mask with manually selected upper and lower thresholds (can use Slicer software to estimate thresholds of DCE image)
def estimate_tumour_mask(DCE_img, lowerthreshold=200, upperthreshold=5000, hole_size=1):
    label_threshold = sitk.BinaryThreshold(DCE_img, lowerThreshold=lowerthreshold, upperThreshold=upperthreshold)
    label_threshold_cc = sitk.RelabelComponent(sitk.ConnectedComponent(label_threshold))
    label_threshold_cc_x = (label_threshold_cc==1)
    label_threshold_cc_x_f = sitk.BinaryMorphologicalClosing(label_threshold_cc_x, (hole_size,hole_size,hole_size))
    return(label_threshold_cc_x_f)

test_tumour_mask=estimate_tumour_mask(masked_breast, lowerthreshold=150, upperthreshold=5000, hole_size=1)
vis = ImageVisualiser(DCE_img, axis='z', cut=get_com(test_tumour_mask), window=[-250, 500])
vis.add_contour(test_tumour_mask, name='BREAST', color='g')
fig = vis.show()
fig.savefig(plots_path+f"WES_{patient_id}_TIMEPOINT_{timepoint}_tumour_mask_over_DCE_img.jpeg",dpi=400)

# after testing specific thresholds, confirm final tumour mask by visualising it over the DCE image in the Slicer software
sitk.WriteImage(test_tumour_mask,tumour_mask_filepath+f"tumour_mask_WES_{patient_id}_TIMEPOINT_{timepoint}_DCE_ACQ_1.nii.gz")