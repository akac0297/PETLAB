#!/usr/bin/env python3
# coding: utf-8

import SimpleITK as sitk
import numpy as np
from platipy.imaging.visualisation.tools import ImageVisualiser
from platipy.imaging.utils.tools import get_com
from platipy.imaging.registration.registration import (
    initial_registration,
    fast_symmetric_forces_demons_registration)

path="/home/alicja/"
patient_list=["04","05","06","07","08","09","10","12","13","14","15","16","18","19","21","23"]
timepoints=["1","2","3"]

def getImages(patient_no,timepoint,path):
    folder="PET_LAB_PROCESSED/WES_0"+patient_no+"/IMAGES/"
    ct="WES_0"+patient_no+"_TIMEPOINT_"+timepoint+"_CT_AC.nii.gz"
    pet="WES_0"+patient_no+"_TIMEPOINT_"+timepoint+"_PET.nii.gz"
    ipsilateral_breast="WES_0" + patient_no + "_TIMEPOINT_" + timepoint + "_PET_IPSI_BREAST.nii.gz"
    ct=sitk.ReadImage(path+folder+ct)
    pet=sitk.ReadImage(path+folder+pet)
    ipsilateral_breast=sitk.ReadImage(path+folder+ipsilateral_breast)
    return(pet,ct,ipsilateral_breast)

patient_no=patient_list[0]
timepoint=timepoints[0]
pet,ct,ipsilateral_breast=getImages(patient_no,timepoint,path)

def flipImage(ipsilateral_breast,patient_no,timepoint):
    folder="PET_LAB_PROCESSED/WES_0"+patient_no+"/IMAGES/"
    ipsi_breast_arr=sitk.GetArrayFromImage(ipsilateral_breast)
    size=np.shape(ipsi_breast_arr)
    contra_breast_arr=np.zeros(size)
    for i in range(size[2]):
        contra_breast_arr[:,:,i]=ipsi_breast_arr[:,:,size[2]-i-1]
    contra_breast_arr[contra_breast_arr>0]=1
    contra_breast_arr[contra_breast_arr==0]=-1000
    contralateral_breast=sitk.GetImageFromArray(contra_breast_arr)
    contralateral_breast.CopyInformation(ipsilateral_breast)
    c_breast_path="WES_0" + patient_no + "_TIMEPOINT_" + timepoint + "_PET_CONTRA_BREAST.nii.gz"
    sitk.WriteImage(contralateral_breast,path+folder+c_breast_path)
    return(contralateral_breast)

contralateral_breast=flipImage(ipsilateral_breast,patient_no,timepoint)

def registerBreastMasktoCT(breast,ct,visualise="T"):
    breast_rigid, tfm_breast_rigid = initial_registration(
        ct,
        breast,
        options={
            'shrink_factors': [8,4],
            'smooth_sigmas': [0,0],
            'sampling_rate': 0.5,
            'final_interp': 2,
            'metric': 'mean_squares',
            'optimiser': 'gradient_descent_line_search',
            'number_of_iterations': 25},
        reg_method='Rigid')
    
    if visualise=="T":
        vis = ImageVisualiser(ct, cut=get_com(breast_rigid), window=[-250, 500])
        vis.add_comparison_overlay(breast_rigid)
        fig = vis.show()

    breast_dir, tfm_breast_dir = fast_symmetric_forces_demons_registration(
        ct,
        breast_rigid,
        resolution_staging=[4,2],
        iteration_staging=[10,10])

    if visualise=="T":
        vis = ImageVisualiser(ct, cut=get_com(breast_dir), window=[-250, 500])
        vis.add_comparison_overlay(breast_dir)
        fig = vis.show()

    return(breast_rigid, tfm_breast_rigid, breast_dir, tfm_breast_dir)

contralateral_breast_rigid,tfm_breast_rigid, contralateral_breast_dir, tfm_breast_dir = registerBreastMasktoCT(contralateral_breast,ct)

folder="PET_LAB_PROCESSED/WES_0"+patient_no+"/IMAGES/"
sitk.WriteImage(contralateral_breast_dir,path+folder+"WES_0" + patient_no + "_TIMEPOINT_" + timepoint + "_PET_CONTRA_BREAST_DIR.nii.gz")