#!/usr/bin/env python
# coding: utf-8

import SimpleITK as sitk
import numpy as np

patient_no="02"
timepoint="0"

def getPETVolume(patient_no,timepoint):
    seg=sitk.ReadImage("pet_seg_0"+patient_no+"_"+timepoint+"_97pc.nii.gz")
    seg_array=sitk.GetArrayFromImage(seg)
    volume=np.sum(seg_array>0)*(seg.GetSpacing()[0]*seg.GetSpacing()[1]*seg.GetSpacing()[2])
    #print("Patient: " +patient_no+ " Timepoint: " + timepoint + " Volume: " + str(volume))
    return(volume)

def getPETRadius(patient_no,timepoint):
    volume=getPETVolume(patient_no,timepoint)
    radius=np.cbrt(3*volume/(4*np.pi))
    #print("Patient:", patient_no, "Timepoint:", timepoint, "Radius:",radius)
    return(radius)

patient_no="03"
timepoint="2"

def getMRIVolume(patient_no,timepoint):
    seg=sitk.ReadImage("new_seg_0"+patient_no+"_"+timepoint+"_mri.nii.gz")
    seg_array=sitk.GetArrayFromImage(seg)
    volume=np.sum(seg_array==1)*(seg.GetSpacing()[0]*seg.GetSpacing()[1]*seg.GetSpacing()[2])
    #print("Patient: " +patient_no+ " Timepoint: " + timepoint + " Volume: " + str(volume))
    return(volume)

def getMRIRadius(patient_no,timepoint):
    volume=getMRIVolume(patient_no,timepoint)
    radius=np.cbrt(3*volume/(4*np.pi))
    #print("Patient:", patient_no, "Timepoint:", timepoint, "Radius:",radius)
    return(radius)

#need to write a PET and an MRI function to run the above code on each patient and time point (can use one representative MRI and the one PET per time point)
#and return a dataframe of volumes (patient, timepoint, tumour volume, tumour radius // one for MRI and one for PET)