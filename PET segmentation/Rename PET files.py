#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import os
import copy

input_fp="/home/alicja/PET-LAB Code/PET-LAB/PET tumour segmentations/"

patient_no="04"

def renameFiles(input_fp,patient_no,percentile):
    exp=f"pet_seg_0{patient_no}"
    filenames=os.listdir(input_fp)
    good_filenames = [name for name in filenames if name.startswith(exp)]
    files_pc = [name for name in good_filenames if name.endswith(f'{percentile}pc.nii.gz')]
    new_files_pc = []
    files_pc_copy=copy.deepcopy(files_pc)
    for file in files_pc_copy:
        print(file)
        if float((file.lstrip(f'pet_seg_0{patient_no}')).rstrip(f'_{percentile}pc.nii.gz') == ''):
            new_file=file
            new_files_pc.append(new_file)
            files_pc.remove(file)
    files_pc.sort(key=lambda x: float((x.lstrip(f'pet_seg_0{patient_no}')).rstrip(f'_{percentile}pc.nii.gz')))
    for file in files_pc:
        new_files_pc.append(file)
    new_exp=f"WES_0{patient_no}_TIMEPOINT_"
    output_files=[new_exp+f"1_PET_TUMOUR_{percentile}_pc.nii.gz",new_exp+f"2_PET_TUMOUR_{percentile}_pc.nii.gz",new_exp+f"3_PET_TUMOUR_{percentile}_pc.nii.gz"]
    for i in range(len(new_files_pc)):
        os.rename(r''+input_fp+new_files_pc[i],r''+output_files[i])
    print('renaming complete')

patient_list=["18","19","21","23"]
for patient_no in patient_list:
    for percentile in ['90','95','97']:
        renameFiles(input_fp,patient_no,percentile)