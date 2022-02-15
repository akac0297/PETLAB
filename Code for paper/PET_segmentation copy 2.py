#!/usr/bin/env python3
# coding: utf-8

import SimpleITK as sitk
import numpy as np
import pandas as pd
from platipy.imaging.registration.linear import linear_registration
from platipy.imaging.registration.deformable import fast_symmetric_forces_demons_registration
from platipy.imaging.registration.utils import apply_transform

def getContourPlan(patient_no):
    df=pd.DataFrame(columns=["PATIENT_ID","CONTOUR_PLAN"])
    patient_list=["04","05","06","07","08","09","10","12","13","14","15","16","18","19","21","23"]
    contour_plans=["WES_004_RTSIM_LABEL_CHESTWALL_LT_PTV.nii.gz",
                "WES_005_RTSIM_LABEL_WHOLE_BREAST_PTV.nii.gz",
                "WES_006_RTSIM_LABEL_CHEST_WALL_PTV.nii.gz",
                "WES_007_RTSIM_LABEL_WHOLE_BREAST_PTV.nii.gz",
                "WES_008_RTSIM_LABEL_CHESTWALL_PTV.nii.gz",
                "WES_009_RTSIM_LABEL_CHESTWALL_PTV.nii.gz",
                "WES_010_RTSIM_LABEL_WHOLE_BREAST_PTV.nii.gz",
                "WES_012_RTSIM_LABEL_WHOLE_BREAST_PTV.nii.gz",
                "WES_013_RTSIM_LABEL_WHOLE_BREAST_PTV.nii.gz",
                "WES_014_RTSIM_LABEL_WHOLE_BREAST_PTV.nii.gz",
                "WES_015_RTSIM_LABEL_WHOLE_BREAST_PTV.nii.gz",
                "WES_016_RTSIM_LABEL_WHOLE_BREAST_PTV.nii.gz",
                "WES_018_RTSIM_LABEL_WHOLE_BREAST_PTV.nii.gz",
                "WES_019_RTSIM_LABEL_WHOLE_BREAST_PTV.nii.gz",
                "WES_021_RTSIM_LABEL_CW_RIGHT_PTV.nii.gz",
                "WES_023_RTSIM_LABEL_WHOLE_BREAST_PTV.nii.gz"
                ]
    for patient in patient_list:
        idx=patient_list.index(patient)
        df.loc[idx]=[int(patient)]+[contour_plans[patient_list.index(patient)]]
    test_df=df[df["PATIENT_ID"]==int(patient_no)]
    contour_plan = test_df["CONTOUR_PLAN"].values[0]
    return(contour_plan)

def getPETimages(patient_no,timepoint,path):
    folder="WES_0"+patient_no+"/IMAGES/"
    ct="WES_0"+patient_no+"_TIMEPOINT_"+timepoint+"_CT_AC.nii.gz"
    pet="WES_0"+patient_no+"_TIMEPOINT_"+timepoint+"_PET.nii.gz"
    ct_plan="WES_0"+patient_no+"_CT_RTSIM.nii.gz"
    image_ct_0=sitk.ReadImage(path+folder+ct)
    image_pt_0_raw=sitk.ReadImage(path+folder+pet)
    image_ct_plan = sitk.ReadImage(path+folder+ct_plan)
    contour_plan=getContourPlan(patient_no)
    contour_breast_plan = sitk.ReadImage(path+"WES_0"+patient_no+"/LABELS/"+contour_plan)
    image_pt_0=sitk.Resample(image_pt_0_raw, image_ct_0)
    return(image_ct_0,image_pt_0_raw,image_pt_0,image_ct_plan,contour_breast_plan)

def registerCTplantoCT(image_ct_0,image_ct_plan):
    image_ct_plan=sitk.Resample(image_ct_plan,image_ct_0,sitk.Transform(),sitk.sitkNearestNeighbor)
    image_ct_plan_to_0_rigid, tfm_plan_to_0_rigid = linear_registration(
        image_ct_0,
        image_ct_plan,
        shrink_factors= [8,4],
        smooth_sigmas= [0,0],
        sampling_rate= 0.5,
        final_interp= 2,
        metric= 'mean_squares',
        optimiser= 'gradient_descent_line_search',
        number_of_iterations= 25,
        reg_method='rigid')
    
    print("initial registration complete")

    image_ct_plan_to_0_dir, tfm_plan_to_0_dir,_ = fast_symmetric_forces_demons_registration(
        image_ct_0,
        image_ct_plan_to_0_rigid,
        resolution_staging=[4,2],
        iteration_staging=[10,10]
    )

    return(image_ct_plan_to_0_rigid,tfm_plan_to_0_rigid,image_ct_plan_to_0_dir,tfm_plan_to_0_dir)

def registerBreastStructtoCT(image_ct_0,contour_breast_plan,tfm_plan_to_0_rigid,tfm_plan_to_0_dir,patient_no,timepoint,output_path):
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

    sitk.WriteImage(contour_breast_plan_to_0_dir,output_path+"PET_plan_breast_seg_"+patient_no+"_"+timepoint+".nii.gz")
    return(contour_breast_plan_to_0_dir)

def maskPET(image_pt_0,image_pt_0_raw,contour_breast_plan_to_0_dir,patient_no,timepoint,output_path):
    contour_breast_plan_to_0_dir=sitk.Resample(contour_breast_plan_to_0_dir,image_pt_0,sitk.Transform(),sitk.sitkNearestNeighbor)
    masked_pet_breast = sitk.Mask(image_pt_0, contour_breast_plan_to_0_dir)
    sitk.WriteImage(masked_pet_breast, output_path+"WES_0" + patient_no + "_TIMEPOINT_" + timepoint + "_PET_IPSI_BREAST.nii.gz")
    
    masked_pet_breast=sitk.Resample(masked_pet_breast, image_pt_0_raw)
    return(masked_pet_breast)

def registerMasks(masked_pet_breast,patient_no,path):
    folder="WES_0"+patient_no+"/IMAGES/"
    mask1=sitk.ReadImage(path+folder+"WES_0"+patient_no+"_TIMEPOINT_1_PET_IPSI_BREAST.nii.gz")
    mask1=sitk.Resample(mask1,masked_pet_breast)
    image_mask1_to_0_rigid, tfm_mask1_to_0_rigid = linear_registration(
        masked_pet_breast,
        mask1,
        shrink_factors= [8,4],
        smooth_sigmas= [0,0],
        sampling_rate= 0.5,
        final_interp= 2,
        metric= 'mean_squares',
        optimiser= 'gradient_descent_line_search',
        number_of_iterations= 25,
        reg_method='Rigid')

    image_mask1_to_0_dir, tfm_mask1_to_0_dir,_ = fast_symmetric_forces_demons_registration(
        masked_pet_breast,
        image_mask1_to_0_rigid,
        resolution_staging=[4,2],
        iteration_staging=[10,10]
    )

    return(image_mask1_to_0_rigid, tfm_mask1_to_0_rigid,image_mask1_to_0_dir,tfm_mask1_to_0_dir)

def maskWithTumour(path,patient_no,timepoint,masked_pet_breast,tfm_mask1_to_0_rigid,tfm_mask1_to_0_dir):
    folder="WES_0"+patient_no+"/IMAGES/"
    tum=sitk.ReadImage(path+folder+"WES_0"+patient_no+"_TIMEPOINT_1_PET_TUMOUR.nii.gz")
    tum_to_0_rigid = apply_transform(
        tum,
        masked_pet_breast,
        tfm_mask1_to_0_rigid
    )

    tum_to_0_dir = apply_transform(
        tum_to_0_rigid,
        masked_pet_breast,
        tfm_mask1_to_0_dir
    )
    
    tum_dilate=sitk.BinaryDilate(tum_to_0_dir, (20,20,20))
    masked_pet_breast=sitk.Mask(masked_pet_breast,tum_dilate==1)
    sitk.WriteImage(masked_pet_breast, path+folder+"WES_0"+patient_no+"_TIMEPOINT_"+timepoint+"_PET_IPSI_BREAST_MASKED.nii.gz")
    return(masked_pet_breast)

def getPETseg(masked_pet_breast,image_pt_0_raw,patient_no,timepoint,output_path):
    mask_arr=sitk.GetArrayFromImage(masked_pet_breast)
    mask_arr=mask_arr.flatten()
    suv_max=np.max(mask_arr)
    t=0.4*suv_max
    tum = sitk.Mask(image_pt_0_raw, masked_pet_breast>t)
    tum = sitk.Cast(tum, sitk.sitkInt64)
    tum_cc = sitk.RelabelComponent(sitk.ConnectedComponent(tum))
    tum = (tum_cc==1)
    sitk.WriteImage(tum, output_path+"WES_0"+patient_no+"_TIMEPOINT_"+timepoint+"_PET_TUMOUR.nii.gz")

    return(masked_pet_breast,tum)

path="/home/alicja/PET_LAB_PROCESSED/"
#output_path="/PET-LAB Code/PET-LAB/PET segmentation/PET_40pc_SUVmax_tumours/"
#patient_list=["04","05","06","07","08","09","10","12"]#,"13","14","15","16","18","19"]
#timepoints=["1","2","3"]

output_path="/home/alicja/PET-LAB Code/PET-LAB/PET segmentation/PET Breast Masks/"
patient_no="16"
timepoint="3"

image_ct_0,image_pt_0_raw,image_pt_0,image_ct_plan,contour_breast_plan=getPETimages(patient_no,timepoint,path)
image_ct_plan_to_0_rigid,tfm_plan_to_0_rigid,image_ct_plan_to_0_dir,tfm_plan_to_0_dir=registerCTplantoCT(image_ct_0,image_ct_plan)
print("CT plan registered to CT")
contour_breast_plan_to_0_dir=registerBreastStructtoCT(image_ct_0,contour_breast_plan,tfm_plan_to_0_rigid,tfm_plan_to_0_dir,patient_no,timepoint,output_path)
#masked_pet_breast=maskPET(image_pt_0,image_pt_0_raw,contour_breast_plan_to_0_dir,patient_no,timepoint,output_path)

"""
for patient_no in patient_list:
    for timepoint in timepoints:
        image_ct_0,image_pt_0_raw,image_pt_0,image_ct_plan,contour_breast_plan=getPETimages(patient_no,timepoint,path)
        image_ct_plan_to_0_rigid,tfm_plan_to_0_rigid,image_ct_plan_to_0_dir,tfm_plan_to_0_dir=registerCTplantoCT(image_ct_0,image_ct_plan)
        print("CT plan registered to CT")
        output_path="/home/alicja/PET-LAB Code/PET-LAB/PET segmentation/PET Breast Masks/"
        contour_breast_plan_to_0_dir=registerBreastStructtoCT(image_ct_0,contour_breast_plan,tfm_plan_to_0_rigid,tfm_plan_to_0_dir,patient_no,timepoint,output_path)
        print("Breast structure registered to CT")
        masked_pet_breast=maskPET(image_pt_0,image_pt_0_raw,contour_breast_plan_to_0_dir,patient_no,timepoint,path)
        print("PET image masked")
        if (timepoint=="2" or timepoint=="3"):
            image_mask1_to_0_rigid, tfm_mask1_to_0_rigid,image_mask1_to_0_dir,tfm_mask1_to_0_dir=registerMasks(
                masked_pet_breast,patient_no,path)
            "Masks registered"
            masked_pet_breast=maskWithTumour(path,patient_no,timepoint,masked_pet_breast,tfm_mask1_to_0_rigid,tfm_mask1_to_0_dir)
            "PET breast masked with tumour"
        masked_pet_breast,tum=getPETseg(masked_pet_breast,image_pt_0_raw,patient_no,timepoint,output_path)
        print(f"Patient {patient_no} timepoint {timepoint} PET tumour segmentation complete")
"""