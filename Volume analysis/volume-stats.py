from scipy import stats
import pandas as pd
import scikit_posthocs as sp

"""
Kruskal-Wallis test - volume analysis
"""

volume_data=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Volume analysis/tumour_volume_analysis_new.csv")

timepoints=[1,2,3]

def runKW(image_type,timepoints,volume_data):
    subset1=volume_data[volume_data["TIMEPOINT"]==timepoints[0]]
    subset1=subset1[subset1["IMAGE_TYPE"]==image_type]
    tp1=subset1["TUMOUR VOLUME_CM3"].to_list()

    subset2=volume_data[volume_data["TIMEPOINT"]==timepoints[1]]
    subset2=subset2[subset2["IMAGE_TYPE"]==image_type]
    tp2=subset2["TUMOUR VOLUME_CM3"].to_list()

    subset3=volume_data[volume_data["TIMEPOINT"]==timepoints[2]]
    subset3=subset3[subset3["IMAGE_TYPE"]==image_type]
    tp3=subset3["TUMOUR VOLUME_CM3"].to_list()
    _, pvalue = stats.kruskal(tp1,tp2,tp3)
    if pvalue!=1:
        print(image_type, stats.kruskal(tp1,tp2,tp3))

def runDunn(image_type,timepoints,volume_data):
    subset1=volume_data[volume_data["TIMEPOINT"]==timepoints[0]]
    subset1=subset1[subset1["IMAGE_TYPE"]==image_type]
    tp1=subset1["TUMOUR VOLUME_CM3"].to_list()

    subset2=volume_data[volume_data["TIMEPOINT"]==timepoints[1]]
    subset2=subset2[subset2["IMAGE_TYPE"]==image_type]
    tp2=subset2["TUMOUR VOLUME_CM3"].to_list()

    subset3=volume_data[volume_data["TIMEPOINT"]==timepoints[2]]
    subset3=subset3[subset3["IMAGE_TYPE"]==image_type]
    tp3=subset3["TUMOUR VOLUME_CM3"].to_list()

    data=[tp1, tp2, tp3]
    result=sp.posthoc_dunn(data,p_adjust='bonferroni')
    #result.to_csv(f"/home/alicja/PET-LAB Code/PET-LAB/Volume analysis/Dunn dataframes/{image_type}_vol_dataframe_Dunn.csv")
    return(result)

image_types = ["MPE MRI", "T2w MRI", "B50T MRI", "B800T MRI"]
for image_type in image_types:
    runKW(image_type,timepoints,volume_data)
    result=runDunn(image_type,timepoints,volume_data)
    print(image_type)
    print(result)

PET_data=volume_data[volume_data["IMAGE_TYPE"]=="PET"]
PET_data=PET_data.drop([195,200,205,210,215,220,225,230,235])
runKW("PET",timepoints,PET_data)
result=runDunn("PET",timepoints,PET_data)
print("PET")
print(result)