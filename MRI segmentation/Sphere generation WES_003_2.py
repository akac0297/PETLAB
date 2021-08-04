#!/usr/bin/env python
# coding: utf-8

import SimpleITK as sitk
import numpy as np
from platipy.imaging.visualisation.tools import ImageVisualiser
from platipy.imaging.utils.tools import get_com
from platipy.imaging.registration.registration import initial_registration
from platipy.imaging.registration.registration import fast_symmetric_forces_demons_registration
from platipy.imaging.registration.registration import apply_field
from platipy.imaging.registration.registration import smooth_and_resample

image_template = sitk.ReadImage("TEMPLATE_MRI_T2W_TSE_2D_SPAIR.nii.gz")
contour_lb_template = sitk.ReadImage("TEMPLATE_CONTOUR_L_BREAST.nii.gz")
contour_rb_template = sitk.ReadImage("TEMPLATE_CONTOUR_R_BREAST.nii.gz")

img_label="T2_TSE_TRA_SPAIR"
patient_no="03"
timepoint="2"
laterality = "L"
radius = 21.58 #calculated from tumour volume
image_test = sitk.ReadImage("/home/alicja/Documents/WES_0"+patient_no+"/IMAGES/WES_003_2_20170207_MR_T2_TSE_TRA_SPAIR_TSE2D1_11_T2_TSE_TRA_SPAIR_3.nii.gz")

def visualiseImages(image_template,contour_lb_template,contour_rb_template,image_test,laterality):
    if laterality == "L":
        vis = ImageVisualiser(image_template, cut=get_com(contour_lb_template), window=(0,200), figure_size_in=5)
    elif laterality == "R":
        vis = ImageVisualiser(image_template, cut=get_com(contour_rb_template), window=(0,200), figure_size_in=5)

    vis.add_contour({"contour_lb_template":contour_lb_template, "contour_rb_template":contour_rb_template})
    fig = vis.show()

    vis = ImageVisualiser(image_test, window=(0,200), figure_size_in=5)
    fig = vis.show()

visualiseImages(image_template,contour_lb_template,contour_rb_template,image_test,laterality)

def RegisterContoursToImage(image_test,image_template,contour_lb_template, contour_rb_template):
    image_template_reg_linear, tfm_template_linear = initial_registration(
        image_test,
        image_template,
        default_value=0,
        options={
        'shrink_factors': (16,8,4),
        'smooth_sigmas': [0,0,0],
        'sampling_rate': 0.5,
        'final_interp': 2,
        'metric': 'mean_squares',
        'optimiser': 'gradient_descent_line_search',
        'number_of_iterations': 25},
    )

    contour_lb_template_reg_linear = apply_field(
        contour_lb_template,
        #reference_image=image_test,
        transform=tfm_template_linear,
        default_value=0,
        interp=1
    )

    contour_rb_template_reg_linear = apply_field(
        contour_rb_template,
        #reference_image=image_test,
        transform=tfm_template_linear,
        default_value=0,
        interp=1
    )

    vis = ImageVisualiser(image_test, window=(0,200), figure_size_in=5)
    vis.add_comparison_overlay(image_template_reg_linear)
    vis.add_contour({"contour_lb_template_reg_linear":contour_lb_template_reg_linear})
    fig1 = vis.show()

    vis = ImageVisualiser(image_test, window=(0,200), figure_size_in=5)
    vis.add_comparison_overlay(image_template_reg_linear)
    vis.add_contour({"contour_rb_template_reg_linear":contour_rb_template_reg_linear})
    fig2 = vis.show()

    _, tfm_template_deformable = fast_symmetric_forces_demons_registration(
        image_test,
        image_template_reg_linear,
        resolution_staging=[12, 6, 3],
        iteration_staging=[20, 20, 20],
        isotropic_resample=True,
        initial_displacement_field=None,
        smoothing_sigma_factor=1,
        smoothing_sigmas=False,
        default_value=0,
        ncores=8,
        interp_order=2
    )

    contour_lb_template_reg_deformable = apply_field(
        contour_lb_template_reg_linear,
        #reference_image=image_test,
        transform=tfm_template_deformable,
        default_value=0,
        interp=1
    )

    contour_rb_template_reg_deformable = apply_field(
        contour_rb_template_reg_linear,
        #reference_image=image_test,
        transform=tfm_template_deformable,
        default_value=0,
        interp=1
    )

    vis = ImageVisualiser(image_test, cut=get_com(contour_lb_template_reg_deformable), window=(0,200), figure_size_in=5)
    #vis.add_comparison_overlay(image_template_reg_deformable)
    vis.add_contour({"contour_lb_template_reg_deformable":contour_lb_template_reg_deformable, 
                    "contour_rb_template_reg_deformable":contour_rb_template_reg_deformable})
    fig = vis.show()

    return(contour_lb_template_reg_deformable,contour_rb_template_reg_deformable)

contour_lb_template_reg_deformable,contour_rb_template_reg_deformable = RegisterContoursToImage(image_test,image_template,contour_lb_template, contour_rb_template)

def WriteSmoothContour(contour_lb_template_reg_deformable,contour_rb_template_reg_deformable,laterality="L"):
    contour_lb_template_reg_deformable_smooth = sitk.BinaryMorphologicalClosing(contour_lb_template_reg_deformable, (15,15,15))
    contour_rb_template_reg_deformable_smooth = sitk.BinaryMorphologicalClosing(contour_rb_template_reg_deformable, (15,15,15))

    sitk.WriteImage(contour_lb_template_reg_deformable_smooth,"contour_lb_template_reg_deformable_smooth_WES_0"+patient_no+"_"+timepoint+img_label+".nii.gz")
    sitk.WriteImage(contour_rb_template_reg_deformable_smooth,"contour_rb_template_reg_deformable_smooth_WES_0"+patient_no+"_"+timepoint+img_label+".nii.gz")

    if laterality == "L":
        vis = ImageVisualiser(image_test, cut=get_com(contour_lb_template_reg_deformable), window=(0,200), figure_size_in=5) 
    elif laterality == "R":
        vis = ImageVisualiser(image_test, cut=get_com(contour_rb_template_reg_deformable), window=(0,200), figure_size_in=5) 
    #vis.add_comparison_overlay(image_template_reg_deformable)
    vis.add_contour({"contour_lb_template_reg_deformable_smooth":contour_lb_template_reg_deformable_smooth,
    "contour_rb_template_reg_deformable_smooth":contour_rb_template_reg_deformable_smooth})
    fig = vis.show()
    return(contour_lb_template_reg_deformable_smooth,contour_rb_template_reg_deformable_smooth)

contour_lb_template_reg_deformable_smooth,contour_rb_template_reg_deformable_smooth=WriteSmoothContour(contour_lb_template_reg_deformable,contour_rb_template_reg_deformable,laterality="L")

def insert_sphere(arr,sp_radius,sp_centre):
    sp_radius=int(sp_radius)
    for x in range(sp_centre[0]-sp_radius,sp_centre[0]+sp_radius+1):
        for y in range(sp_centre[1]-sp_radius,sp_centre[1]+sp_radius+1):
            for z in range(sp_centre[2]-sp_radius,sp_centre[2]+sp_radius+1):
                dist_squared=sp_radius**2-abs(sp_centre[0]-x)**2-abs(sp_centre[1]-y)**2-abs(sp_centre[2]-z)**2
                sign=np.sign(dist_squared)
                dist=np.sqrt(abs(dist_squared))*sign
                if dist>=0:
                    arr[x,y,z]=1
    return(arr)

def generateSphere(contour_template_reg_deformable=contour_lb_template_reg_deformable,
contour_template_reg_deformable_smooth=contour_lb_template_reg_deformable_smooth,radius=radius):
    blank_image_res = smooth_and_resample(contour_template_reg_deformable, smoothing_sigma=0,shrink_factor=1,isotropic_resample=True)

    centre = get_com(blank_image_res)
    print("Centre is", centre)

    blank_arr = sitk.GetArrayFromImage(blank_image_res*0)
    sphere_arr = insert_sphere(blank_arr, sp_radius=radius, sp_centre=centre)
    image_sphere = sitk.GetImageFromArray(sphere_arr)
    image_sphere.CopyInformation(blank_image_res)
    image_sphere = sitk.Resample(image_sphere, contour_template_reg_deformable)

    vis = ImageVisualiser(image_test, cut=get_com(contour_template_reg_deformable), window=(0,200), figure_size_in=5)
    vis.add_contour(
        {"contour_template_reg_deformable_smooth":contour_template_reg_deformable_smooth,
        "image_sphere":image_sphere
        }
    )
    fig = vis.show()

    sitk.WriteImage(image_sphere,"image_sphere_WES_0"+patient_no+"_"+timepoint+".nii.gz")
    return(image_sphere)

image_sphere=generateSphere(contour_template_reg_deformable=contour_lb_template_reg_deformable,
contour_template_reg_deformable_smooth=contour_lb_template_reg_deformable_smooth,radius=radius)
