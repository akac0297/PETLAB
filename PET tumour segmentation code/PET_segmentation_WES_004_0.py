#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Import modules
"""

import pathlib
import numpy as np

import SimpleITK as sitk
import matplotlib.pyplot as plt

from platipy.imaging.visualisation.tools import ImageVisualiser
from platipy.imaging.utils.tools import get_com


from platipy.imaging.registration.registration import (
    initial_registration,
    fast_symmetric_forces_demons_registration,
    transform_propagation,
    apply_field
)

get_ipython().run_line_magic('matplotlib', 'notebook')


# In[82]:


patient_no="04"
timepoint="0"

ct="WES_004_0_20170502_CT_10_PETCT_WBHDIN_ONC_3.nii.gz"
pet="WES_004_0_20170502_PT_AC_4.nii.gz"
ct_plan="WES_004_3_20171127_CT_2.nii.gz"
breast_struct="WES_004_3_0_RTSTRUCT_CHESTWALL_LT_CTV.nii.gz" #if present

image_ct_0=sitk.ReadImage("/home/alicja/Documents/WES_0" + patient_no + "/IMAGES/"+ct)
image_pt_0_raw=sitk.ReadImage("/home/alicja/Documents/WES_0" + patient_no + "/IMAGES/"+pet)
image_ct_plan = sitk.ReadImage("/home/alicja/Documents/WES_0" + patient_no + "/IMAGES/"+ct_plan)
contour_breast_plan = sitk.ReadImage("/home/alicja/Documents/WES_0" + patient_no + "/STRUCTURES/"+breast_struct)
#L contour_breast_plan = sitk.ReadImage("/home/alicja/Documents/WES_012/STRUCTURES/WES_012_3_0_RTSTRUCT_BREAST_LT_PTV.nii.gz")
#R contour_breast_plan = sitk.ReadImage("/home/alicja/Documents/WES_010/STRUCTURES/WES_010_3_0_RTSTRUCT_BREAST_RT_PTV.nii.gz")

image_ct_plan_arr=sitk.GetArrayFromImage(image_ct_plan)
image_ct_plan_arr[image_ct_plan_arr.shape[0]//5*4:,:,:]=-1000
image_ct_plan_new=sitk.GetImageFromArray(image_ct_plan_arr)
image_ct_plan_new.CopyInformation(image_ct_plan)
image_ct_plan=image_ct_plan_new

#image_pt_0_raw_arr=sitk.GetArrayFromImage(image_pt_0_raw)
#image_pt_0_raw_arr[:image_pt_0_raw_arr.shape[0]//2,:,:]=0
#pet_new=sitk.GetImageFromArray(image_pt_0_raw_arr)
#pet_new.CopyInformation(image_pt_0_raw)
#pet_new=sitk.Resample(pet_new,image_ct_0)

image_pt_0=sitk.Resample(image_pt_0_raw, image_ct_0)


# In[77]:


vis = ImageVisualiser(image_ct_0, cut=[180,220,256], window=[-250, 500])
fig = vis.show()


# In[83]:


vis = ImageVisualiser(pet_new, colormap=plt.cm.magma, cut=[180,220,256], window=[0.1, 10000])
fig = vis.show()


# In[59]:


vis = ImageVisualiser(image_ct_plan, window=[-250, 500], figure_size_in=8)#, axis='z', cut=60)
vis.add_contour({'BREAST' :contour_breast_plan})
fig = vis.show()


# In[84]:


#register planning CT to CT
image_ct_plan_to_0_rigid, tfm_plan_to_0_rigid = initial_registration(
    image_ct_0,
    image_ct_plan,
    options={
        'shrink_factors': [8,4],
        'smooth_sigmas': [0,0],
        'sampling_rate': 0.5,
        'final_interp': 2,
        'metric': 'mean_squares',
        'optimiser': 'gradient_descent_line_search',
        'number_of_iterations': 25},
    reg_method='Rigid')


# In[85]:


vis = ImageVisualiser(image_ct_0, cut=[160,260,256], window=[-250, 500])
vis.add_comparison_overlay(image_ct_plan_to_0_rigid)
fig = vis.show()


# In[86]:


image_ct_plan_to_0_dir, tfm_plan_to_0_dir = fast_symmetric_forces_demons_registration(
    image_ct_0,
    image_ct_plan_to_0_rigid,
    resolution_staging=[4,2],
    iteration_staging=[10,10]
)


# In[87]:


vis = ImageVisualiser(image_ct_0, cut=[180,220,256], window=[-250, 500])
vis.add_comparison_overlay(image_ct_plan_to_0_dir)
fig = vis.show()


# In[106]:


#register breast structure to CT
contour_breast_plan_to_0_rigid = transform_propagation(
    image_ct_0,
    contour_breast_plan,
    tfm_plan_to_0_rigid,
    structure=True
)

contour_breast_plan_to_0_dir = apply_field(
    contour_breast_plan_to_0_rigid,
    tfm_plan_to_0_dir,
    structure=True
)


# In[107]:


contour_breast_plan_to_0_dir_arr=sitk.GetArrayFromImage(contour_breast_plan_to_0_dir)
contour_breast_plan_to_0_dir_arr[196:,:,:]=0
#contour_breast_plan_to_0_dir_arr[:163,:,:]=0
#contour_breast_plan_to_0_dir_arr[186:,:,:]=0
contour_breast_plan_to_0_dir2=sitk.GetImageFromArray(contour_breast_plan_to_0_dir_arr)
contour_breast_plan_to_0_dir2.CopyInformation(contour_breast_plan_to_0_dir)
contour_breast_plan_to_0_dir=contour_breast_plan_to_0_dir2


# In[108]:


breast_contour_dilate=sitk.BinaryDilate(contour_breast_plan_to_0_dir, (4,4,4)) #if using different structure


# In[109]:


sitk.WriteImage(breast_contour_dilate,"breast_contour_dilate_"+patient_no+"_"+timepoint+".nii.gz")


# In[111]:


vis = ImageVisualiser(image_ct_0, axis='z', cut=180, window=[-250, 500])
vis.add_scalar_overlay(image_pt_0, name='PET SUV', colormap=plt.cm.magma, min_value=0.1, max_value=10000)
vis.add_contour(contour_breast_plan_to_0_dir, name='R BREAST', color='g') #or breast_contour_dilate
fig = vis.show()


# In[112]:


#use structure information for breast to mask out all but the breast area (create new array with everything but
#this area set to 0)
masked_pet_breast = sitk.Mask(image_pt_0, contour_breast_plan_to_0_dir) #or breast_contour_dilate


# In[113]:


values = sitk.GetArrayViewFromImage(masked_pet_breast).flatten()

fig, ax = plt.subplots(1,1)
ax.hist(values, bins=np.linspace(1,30000,50), histtype='stepfilled', lw=2)
ax.set_yscale('log')
ax.grid()
ax.set_axisbelow(True)
ax.set_xlabel('PET value')
ax.set_ylabel('Frequency')
fig.show()


# In[114]:


sitk.WriteImage(masked_pet_breast, "masked_pet_breast_WES_0" + patient_no + "_" + timepoint + ".nii.gz")


# In[115]:


#get 95th percentile, then mask the breast volume
masked_pet_breast=sitk.Resample(masked_pet_breast, image_pt_0_raw)
mask_arr=sitk.GetArrayFromImage(masked_pet_breast)
mask_arr=mask_arr.flatten() 

p = np.percentile(mask_arr[mask_arr>0], 95)
print(p)

tum = sitk.Mask(image_pt_0_raw, masked_pet_breast>p)


# In[ ]:





# In[116]:


sitk.WriteImage(tum, "pet_seg_0"+patient_no+"_"+timepoint+"_95pc.nii.gz")


# In[117]:


p = np.percentile(mask_arr[mask_arr>0], 90)
print(p)

tum = sitk.Mask(image_pt_0_raw, masked_pet_breast>p)

sitk.WriteImage(tum, "pet_seg_0"+patient_no+"_"+timepoint+"_90pc.nii.gz")


# In[118]:


p = np.percentile(mask_arr[mask_arr>0], 97)
print(p)

tum = sitk.Mask(image_pt_0_raw, masked_pet_breast>p)

sitk.WriteImage(tum, "pet_seg_0"+patient_no+"_"+timepoint+"_97pc.nii.gz")


# In[ ]:




