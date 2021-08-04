#!/usr/bin/env python
# coding: utf-8

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from platipy.imaging.utils.tools import get_com
from platipy.imaging.visualisation.tools import ImageVisualiser
from platipy.imaging.registration.registration import (
    initial_registration,
    fast_symmetric_forces_demons_registration,
    transform_propagation,
    apply_field
)

#read in MR images
pat_no="06"
timept="4"
laterality = "L"

filenameB50T_1="WES_006_4_20180307_MR_TEST_EP2D_DIFF_TRA_SPAIR_ZOOMIT_EZ_B50T_TEST_EP2D_DIFF_TRA_SPAIR_ZOOMIT_TRACEW_DFC_MIX_6.nii.gz"
filenameB800T_1="WES_006_4_20180307_MR_TEST_EP2D_DIFF_TRA_SPAIR_ZOOMIT_EZ_B800T_TEST_EP2D_DIFF_TRA_SPAIR_ZOOMIT_TRACEW_DFC_MIX_6.nii.gz"
filenameT2w_1="WES_006_4_20180307_MR_T2_TSE_TRA_SPAIR_RL_TSE2D1_8_T2_TSE_TRA_SPAIR_RL_5.nii.gz"
filenameMPE_1="max_img_WES_0" +pat_no+"_"+timept+".nii.gz"

WES_1_B50T=sitk.ReadImage("/home/alicja/Documents/WES_0" + pat_no + "/IMAGES/" +filenameB50T_1)
WES_1_B800T=sitk.ReadImage("/home/alicja/Documents/WES_0" + pat_no + "/IMAGES/" +filenameB800T_1)
WES_1_T2w=sitk.ReadImage("/home/alicja/Documents/WES_0" + pat_no + "/IMAGES/" +filenameT2w_1)
WES_1_MPE=sitk.ReadImage(filenameMPE_1)
WES_010_4_B50T=sitk.ReadImage("/home/alicja/Documents/WES_010/IMAGES/WES_010_4_20180829_MR_EP2D_DIFF_TRA_SPAIR_ZOOMIT_EZ_B50T_EP2D_DIFF_TRA_SPAIR_ZOOMIT_TRACEW_DFC_MIX_5.nii.gz")

if laterality == "R":
    breast=sitk.ReadImage("/home/alicja/Downloads/R-breast.nii.gz")
elif laterality == "L":
    breast=sitk.ReadImage("/home/alicja/Downloads/L-breast.nii.gz")

def regBreasttoMRI(WES_1_T2w,WES_010_4_B50T):
    image_to_0_rigid, tfm_to_0_rigid = initial_registration(
        WES_1_T2w,
        WES_010_4_B50T,
        options={
            'shrink_factors': [8,4],
            'smooth_sigmas': [0,0],
            'sampling_rate': 0.5,
            'final_interp': 2,
            'metric': 'mean_squares',
            'optimiser': 'gradient_descent_line_search',
            'number_of_iterations': 25},
        reg_method='Rigid')

    _, tfm_to_0_dir = fast_symmetric_forces_demons_registration(
        WES_1_T2w,
        image_to_0_rigid,
        resolution_staging=[4,2],
        iteration_staging=[10,10]
    )

    breast_to_0_rigid = transform_propagation(
        WES_1_T2w,
        breast,
        tfm_to_0_rigid,
        structure=True
    )

    breast_to_0_dir = apply_field(
        breast_to_0_rigid,
        tfm_to_0_dir,
        structure=True
    )

    vis = ImageVisualiser(WES_1_T2w, axis='z', cut=get_com(breast_to_0_dir), window=[-250, 500])
    vis.add_contour(breast_to_0_dir, name='BREAST', color='g')
    fig = vis.show()

    breast_contour=breast_to_0_dir

    return(breast_contour,fig)

breast_contour, fig = regBreasttoMRI(WES_1_T2w,WES_010_4_B50T)

def dilateBreastContour(breast_contour,x=10,y=10,z=10):
    breast_contour_dilate=sitk.BinaryDilate(breast_contour, (x,y,z))
    vis = ImageVisualiser(WES_1_T2w, axis='z', cut=get_com(breast_contour), window=[-250, 500])
    vis.add_contour(breast_contour_dilate, name='BREAST', color='g')
    fig = vis.show()

    return(breast_contour_dilate)

breast_contour_dilate=dilateBreastContour(breast_contour,10,10,10)

def maskBreast(breast_contour,image):
    breast_contour=sitk.Resample(breast_contour,image)
    masked_breast = sitk.Mask(image, breast_contour)

    return(masked_breast)

masked_breast=maskBreast(breast_contour_dilate,WES_1_B50T)

def plotHistogram(masked_breast):
    values = sitk.GetArrayViewFromImage(masked_breast).flatten()
    fig, ax = plt.subplots(1,1)
    ax.hist(values, bins=np.linspace(1,1000,50), histtype='stepfilled', lw=2)
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Frequency')
    fig.show()

plotHistogram(masked_breast)

def estimate_tumour_vol(img_mri, lowerthreshold=300, upperthreshold=5000, hole_size=1):
    label_threshold = sitk.BinaryThreshold(img_mri, lowerThreshold=lowerthreshold, upperThreshold=upperthreshold)
    label_threshold_cc = sitk.RelabelComponent(sitk.ConnectedComponent(label_threshold))
    label_threshold_cc_x = (label_threshold_cc==1)
    label_threshold_cc_x_f = sitk.BinaryMorphologicalClosing(label_threshold_cc_x, (hole_size,hole_size,hole_size))
    return(label_threshold_cc_x_f)

def maskedImage(image_mri,laterality):
    arr_mri = sitk.GetArrayFromImage(image_mri)
    if laterality == "R":
        arr_mri[:,:,arr_mri.shape[2]//2:] = 0
    elif laterality == "L":
        arr_mri[:,:,:arr_mri.shape[2]//2] = 0
    image_mri_masked=sitk.GetImageFromArray(arr_mri)
    image_mri_masked.CopyInformation(image_mri)

    return image_mri_masked

image_mri=WES_1_B50T
image_mri_masked=maskedImage(image_mri,laterality)

label_threshold_cc_x_f=estimate_tumour_vol(image_mri_masked, lowerthreshold=300, upperthreshold=5000, hole_size=1)
sitk.WriteImage(label_threshold_cc_x_f,"test_label_threshold_0" + pat_no + "_" +timept +"_B50T_hist.nii.gz")

#B800T TRACE segmentation

masked_breast=maskBreast(breast_contour_dilate,WES_1_B800T)
plotHistogram(masked_breast)
image_mri=WES_1_B800T
image_mri_masked=maskedImage(image_mri,laterality)

label_threshold_cc_x_f=estimate_tumour_vol(image_mri_masked, lowerthreshold=120, upperthreshold=5000, hole_size=1)
sitk.WriteImage(label_threshold_cc_x_f,"test_label_threshold_0" + pat_no + "_" +timept +"_B800T_hist.nii.gz")

#T2w segmentation
WES_1_T2w=sitk.Resample(WES_1_T2w,WES_1_B50T)
masked_breast=maskBreast(breast_contour_dilate,WES_1_T2w)
plotHistogram(masked_breast)
image_mri=WES_1_T2w
image_mri_masked=maskedImage(image_mri,laterality)

label_threshold_cc_x_f=estimate_tumour_vol(image_mri_masked, lowerthreshold=110, upperthreshold=500, hole_size=1)
sitk.WriteImage(label_threshold_cc_x_f,"test_label_threshold_0" + pat_no + "_" +timept +"_T2w_hist.nii.gz")

#MPE segmentation
WES_1_MPE=sitk.Resample(WES_1_MPE,WES_1_B50T)
masked_breast=maskBreast(breast_contour_dilate,WES_1_MPE)
plotHistogram(masked_breast)
image_mri=WES_1_MPE
image_mri_masked=maskedImage(image_mri,laterality)

#MPE segmentation required further masking to isolate the breast on MRI
def cutCoronalSection(image_mri,cut="halfway"):
    arr_mri = sitk.GetArrayFromImage(image_mri)
    if cut == "halfway":
        arr_mri[:,arr_mri.shape[1]//2:,:] = 0
    elif cut == "one third":
        arr_mri[:,arr_mri.shape[1]//3:,:] = 0
    elif cut == "two thirds":
        arr_mri[:,arr_mri.shape[1]*2//3:,:] = 0
    image_mri_masked=sitk.GetImageFromArray(arr_mri)
    image_mri_masked.CopyInformation(image_mri)
    return(image_mri_masked)

image_mri_masked=cutCoronalSection(image_mri,cut="halfway")

label_threshold_cc_x_f=estimate_tumour_vol(image_mri_masked, lowerthreshold=200, upperthreshold=5000, hole_size=1)
sitk.WriteImage(label_threshold_cc_x_f,"test_label_threshold_0" + pat_no + "_" +timept +"_MPE_hist.nii.gz")

#add segmentations
def addSegmentations(pat_no,timept):
    seg_B50T=sitk.ReadImage("test_label_threshold_0" + pat_no + "_" +timept +"_B50T_hist.nii.gz")
    seg_B800T=sitk.ReadImage("test_label_threshold_0" + pat_no + "_" +timept +"_B800T_hist.nii.gz")
    seg_T2=sitk.ReadImage("test_label_threshold_0" + pat_no + "_" +timept +"_T2w_hist.nii.gz")
    seg_MPE=sitk.ReadImage("test_label_threshold_0" + pat_no + "_" +timept +"_MPE_hist.nii.gz")

    seg_B50T=sitk.Resample(seg_B50T,seg_T2)
    seg_B800T=sitk.Resample(seg_B800T,seg_T2)
    seg_MPE=sitk.Resample(seg_MPE,seg_T2)

    new_seg_T2=sitk.LabelMapToBinary(sitk.Cast(seg_T2, sitk.sitkLabelUInt8))
    new_seg_B50T=sitk.LabelMapToBinary(sitk.Cast(seg_B50T, sitk.sitkLabelUInt8))
    new_seg_B800T=sitk.LabelMapToBinary(sitk.Cast(seg_B800T, sitk.sitkLabelUInt8))
    new_seg_MPE=sitk.LabelMapToBinary(sitk.Cast(seg_MPE, sitk.sitkLabelUInt8))

    new_TRACE_seg=(new_seg_B50T+new_seg_B800T)/2
    new_seg_1=(sitk.Cast(new_seg_T2,sitk.sitkFloat64)+sitk.Cast(new_TRACE_seg,sitk.sitkFloat64)+sitk.Cast(new_seg_MPE,sitk.sitkFloat64))
    vis=ImageVisualiser(new_seg_1, cut=get_com(new_seg_1), window=[0,3])
    fig=vis.show()

    new_seg_1_1=sitk.BinaryThreshold(new_seg_1, lowerThreshold=2)

    vis=ImageVisualiser(new_seg_1_1, cut=get_com(new_seg_1), window=[0,1])
    fig=vis.show()
    sitk.WriteImage(new_seg_1_1,"new_seg_0"+pat_no+"_"+timept+"_mri.nii.gz")
    return(new_seg_1_1,fig)

new_seg,fig=addSegmentations(pat_no,timept)

#I need to work on this segmentation procedure. For the time being I may put the input parameters for generating my segmentations into a spreadsheet.
#The method requires a lot of manual testing so is not really reproducible at the moment, but having the parameters could allow users to reproduce
#my specific manual/semi-automatic segmentations.
