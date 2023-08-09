#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
DCE-MRI Tumour segmentation testing
"""


# In[2]:


import SimpleITK as sitk
from platipy.imaging.visualisation.tools import ImageVisualiser
from platipy.imaging.utils.tools import get_com
import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[59]:


"""
Mask DCE-MRI images with dilated T2W breast mask (and potentially threshold out the background)
"""
patient_list=["04","05","06","07","08","09","10","12","13","14","15","16","18","19","21","23"]
timepoints=["1","2","3"]

patient_no=patient_list[0]
timepoint=timepoints[0]

mask_path="/home/alicja/PET-LAB Code/BREAST_MASKS/Edited breast masks/"
MRI_folder="/home/alicja/PET_LAB_PROCESSED/WES_0"+patient_no+"/IMAGES/"

DCE_images=[]
DCE_arrays=[]
for i in range(0,6):
    DCE_image = sitk.ReadImage(MRI_folder+f"WES_0{patient_no}_TIMEPOINT_{timepoint}_MRI_T1W_DCE_ACQ_{i}.nii.gz")
    DCE_images.append(DCE_image)
    DCE_arr = sitk.GetArrayFromImage(DCE_image)
    DCE_arrays.append(DCE_arr)


# In[8]:


"""
Create a 4D array of all the masked DCE images
"""
DCE_4D_arr = np.stack([sitk.GetArrayFromImage(i) for i in DCE_images])


# In[129]:


"""
See if the maximum of the normalised intensity curve is >1.5 (or 1.3 or 1.4)
"""
def normalise(DCE_4D_arr):
    DCE_0_for_norm=DCE_4D_arr[0,:,:,:]
    DCE_0_for_norm[DCE_0_for_norm==0]=1
    DCE_4D_arr_norm=np.ones((np.shape(DCE_4D_arr)))
    for i in range(1,5):
        DCE_4D_arr_norm[i,:,:,:]=(DCE_4D_arr[i,:,:,:]/(DCE_0_for_norm[:,:,:]))
    DCE_4D_arr_norm[0,:,:,:]=1
    return(DCE_4D_arr_norm)

DCE_4D_arr_norm=normalise(DCE_4D_arr)

#see if the maximum of DCE_4D_arr_norm[1:5,:,:,:] is less than 1.3
def maxEnhancement(DCE_4D_arr_norm,DCE_images):
    stacked_arr=DCE_4D_arr_norm
    stacked_arr=stacked_arr[1:5,:,:,:]
    print(np.shape(stacked_arr))
    max_arr_values = np.max(stacked_arr, axis=0)
    tumour_arr=np.ones(np.shape(sitk.GetArrayFromImage(DCE_images[0])))
    tumour_arr[max_arr_values<=1.5]=0
    return(tumour_arr)

"""
See if maximum signal intensity is reached before the end of scan time: 
- this doesn't work for our data, as the tumour doesn't reach maximum intensity before the end of the scan time
"""
#max(all time points minus the last one) > [value at final timepoint]
def maxIntensity(DCE_4D_arr,tumour_arr):
    stacked_arr=DCE_4D_arr
    final_tp=stacked_arr[5,:,:]
    stacked_arr=stacked_arr[0:4,:,:]
    max_arr_values = np.max(stacked_arr, axis=0)
    tumour_arr[max_arr_values<final_tp]=0
    return(tumour_arr)


# In[130]:


def getTumourSeg(DCE_4D_arr,DCE_images):
    DCE_4D_arr_norm=normalise(DCE_4D_arr)
    tumour_arr=maxEnhancement(DCE_4D_arr_norm,DCE_images)
    tumour_arr=maxIntensity(DCE_4D_arr,tumour_arr)
    tumour=sitk.GetImageFromArray(tumour_arr)
    tumour.CopyInformation(DCE_images[0])
    sitk.WriteImage(tumour,mask_path+"test_final_tumour.nii.gz")
    return(tumour)


# In[132]:


tumour=getTumourSeg(DCE_4D_arr,DCE_images)
vis=ImageVisualiser(tumour,cut=[100,50,100],window=[0,1])
fig=vis.show()


# In[105]:


"""
Testing
"""

test_img=max_arr_values
test_img=sitk.GetImageFromArray(test_img)
test_img.CopyInformation(DCE_images[3])
sitk.WriteImage(test_img,mask_path+"max_arr_values.nii.gz")

def getPostContrastShape(DCE_4D_arr):
    shape=(np.shape(DCE_4D_arr)[0]-1,np.shape(DCE_4D_arr)[1],np.shape(DCE_4D_arr)[2],np.shape(DCE_4D_arr)[3])
    return(shape)

shape=getPostContrastShape(DCE_4D_arr_norm)
print(shape)

