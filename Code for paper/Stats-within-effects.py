import pandas as pd
import numpy as np
import pingouin as pg

MRI_patient_list=("06","14","15","16","18","19","21","23")
modalities = ["DCE ME", "T2w SPAIR", "T1w", "ADC"]
binCounts = ["32","64","128","256","None"]

# will be comparing between the different bin counts / normalisations rather than different time points (choose 1 time point)

def splitDfs(patient_id,binCount,modality,normalised):
    df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Radiomics_features_contralateral_WES_0{}_binCount{}_df.csv".format(patient_id,binCount))
    # should I exclude shape features from the "within effects" analysis? Shape and first order features should remain stable.
    # df=pd.concat([df.iloc[:,0:19],df.iloc[:,33:]],axis=1)
    df = df.loc[df['modality'] == modality]

    df_z=df[df[list(df)[-1]] == True]
    df=df[df[list(df)[-1]] == False]

    if normalised == True:
        df=df_z

    Bef_df=df[df["time label"] == "Before PST"]
    Dur_df=df[df["time label"] == "During PST"]
    Post_df=df[df["time label"] == "Post PST"]

    Bef_df=Bef_df.drop(['time label','modality','Unnamed: 0','Z-score normalisation'], axis=1)
    Dur_df=Dur_df.drop(['time label','modality','Unnamed: 0','Z-score normalisation'], axis=1)
    Post_df=Post_df.drop(['time label','modality','Unnamed: 0','Z-score normalisation'], axis=1)

    return Bef_df, Dur_df, Post_df

#calculate the coefficient of variation to assess the repeatability of the radiomics features
def getCV(dataframes):
    table = pd.concat(dataframes).pivot_table(index='Patient',aggfunc=[np.mean, np.std])
    CV = table["std"].div(abs(table["mean"]))*100 #take the absolute value of the means
    CV_ranked = CV.median(axis=0).sort_values(ascending=True)
    top_5 = CV_ranked[:5]
    features=list(top_5.index)
    CV_summarised=CV[features]
    return CV, CV_summarised, top_5

def getICC(dfs,CV_df):
    icc_dict={}
    features=list(CV_df)
    for feature in features:
        feature_df = pd.DataFrame()
        idx=0
        for df in dfs:
            tp_df=pd.DataFrame()
            tp_df["Patient"] = ("06","14","15","16","18","19","21","23")
            tp_df[feature] = df[feature]
            tp_df["Rater"] = idx
            feature_df = pd.concat([feature_df,tp_df],ignore_index=True)
            idx += 1
        column_names=list(feature_df)
        icc=pg.intraclass_corr(data=feature_df,targets='Patient',raters="Rater",ratings=column_names[1],nan_policy="omit")
        icc.set_index("Type")
        icc3=icc.iloc[2,2] # We want the 'single fixed raters' ICC
        icc_dict[feature]=icc3

    return icc_dict

def filterICC(icc_dict,value):
    icc_dict_repr=dict()
    for (key, val) in icc_dict.items():
        if val > value:
            icc_dict_repr[key]=val
    return icc_dict_repr

# Combine the information for all the patients
ADC_Bef_list = []
ADC_Dur_list = []
ADC_Post_list = []
DCE_Bef_list = []
DCE_Dur_list = []
DCE_Post_list = []
T1w_Bef_list = []
T1w_Dur_list = []
T1w_Post_list = []
T2w_Bef_list = []
T2w_Dur_list = []
T2w_Post_list = []
T1w_Z_Bef_list = []
T1w_Z_Dur_list = []
T1w_Z_Post_list = []
T2w_Z_Bef_list = []
T2w_Z_Dur_list = []
T2w_Z_Post_list = []
for binCount in binCounts:
    ADC_Bef=pd.DataFrame()
    ADC_Dur=pd.DataFrame()
    ADC_Post=pd.DataFrame()

    T2w_Bef=pd.DataFrame()
    T2w_Dur=pd.DataFrame()
    T2w_Post=pd.DataFrame()

    T1w_Bef=pd.DataFrame()
    T1w_Dur=pd.DataFrame()
    T1w_Post=pd.DataFrame()

    DCE_Bef=pd.DataFrame()
    DCE_Dur=pd.DataFrame()
    DCE_Post=pd.DataFrame()

    T2w_Z_Bef=pd.DataFrame()
    T2w_Z_Dur=pd.DataFrame()
    T2w_Z_Post=pd.DataFrame()

    T1w_Z_Bef=pd.DataFrame()
    T1w_Z_Dur=pd.DataFrame()
    T1w_Z_Post=pd.DataFrame()
    for patient_id in MRI_patient_list:
        ADC_Bef_df, ADC_Dur_df, ADC_Post_df = splitDfs(patient_id,binCount,"ADC",normalised=False)
        ADC_Bef=pd.concat([ADC_Bef,ADC_Bef_df], axis=0, ignore_index=True)
        ADC_Dur=pd.concat([ADC_Dur, ADC_Dur_df], axis=0, ignore_index=True)
        ADC_Post=pd.concat([ADC_Post, ADC_Post_df], axis=0, ignore_index=True)

        T2w_Bef_df, T2w_Dur_df, T2w_Post_df = splitDfs(patient_id,binCount,"T2w SPAIR",normalised=False)
        T2w_Bef=pd.concat([T2w_Bef,T2w_Bef_df], axis=0, ignore_index=True)
        T2w_Dur=pd.concat([T2w_Dur, T2w_Dur_df], axis=0, ignore_index=True)
        T2w_Post=pd.concat([T2w_Post, T2w_Post_df], axis=0, ignore_index=True)

        T1w_Bef_df, T1w_Dur_df, T1w_Post_df = splitDfs(patient_id,binCount,"T1w",normalised=False)
        T1w_Bef=pd.concat([T1w_Bef,T1w_Bef_df], axis=0, ignore_index=True)
        T1w_Dur=pd.concat([T1w_Dur, T1w_Dur_df], axis=0, ignore_index=True)
        T1w_Post=pd.concat([T1w_Post, T1w_Post_df], axis=0, ignore_index=True)

        DCE_Bef_df, DCE_Dur_df, DCE_Post_df = splitDfs(patient_id,binCount,"DCE ME",normalised=False)
        DCE_Bef=pd.concat([DCE_Bef,DCE_Bef_df], axis=0, ignore_index=True)
        DCE_Dur=pd.concat([DCE_Dur, DCE_Dur_df], axis=0, ignore_index=True)
        DCE_Post=pd.concat([DCE_Post, DCE_Post_df], axis=0, ignore_index=True)

        T2w_Z_Bef_df, T2w_Z_Dur_df, T2w_Z_Post_df = splitDfs(patient_id,binCount,"T2w SPAIR",normalised=True)
        T2w_Z_Bef=pd.concat([T2w_Z_Bef,T2w_Z_Bef_df], axis=0, ignore_index=True)
        T2w_Z_Dur=pd.concat([T2w_Z_Dur, T2w_Z_Dur_df], axis=0, ignore_index=True)
        T2w_Z_Post=pd.concat([T2w_Z_Post, T2w_Z_Post_df], axis=0, ignore_index=True)

        T1w_Z_Bef_df, T1w_Z_Dur_df, T1w_Z_Post_df = splitDfs(patient_id,binCount,"T1w",normalised=True)
        T1w_Z_Bef=pd.concat([T1w_Z_Bef,T1w_Z_Bef_df], axis=0, ignore_index=True)
        T1w_Z_Dur=pd.concat([T1w_Z_Dur, T1w_Z_Dur_df], axis=0, ignore_index=True)
        T1w_Z_Post=pd.concat([T1w_Z_Post, T1w_Z_Post_df], axis=0, ignore_index=True)

    ADC_Bef_list.append(ADC_Bef)
    ADC_Dur_list.append(ADC_Dur)
    ADC_Post_list.append(ADC_Post)
    DCE_Bef_list.append(DCE_Bef)
    DCE_Dur_list.append(DCE_Dur)
    DCE_Post_list.append(DCE_Post)
    T1w_Bef_list.append(T1w_Bef)
    T1w_Dur_list.append(T1w_Dur)
    T1w_Post_list.append(T1w_Post)
    T2w_Bef_list.append(T2w_Bef)
    T2w_Dur_list.append(T2w_Dur)
    T2w_Post_list.append(T2w_Post)
    T1w_Z_Bef_list.append(T1w_Z_Bef)
    T1w_Z_Dur_list.append(T1w_Z_Dur)
    T1w_Z_Post_list.append(T1w_Z_Post)
    T2w_Z_Bef_list.append(T2w_Z_Bef)
    T2w_Z_Dur_list.append(T2w_Z_Dur)
    T2w_Z_Post_list.append(T2w_Z_Post)

DCE_CV_full, DCE_CV, DCE_top_5 = getCV(DCE_Bef_list)
ADC_CV_full, ADC_CV, ADC_top_5 = getCV(ADC_Bef_list)
T1w_CV_full, T1w_CV, T1w_top_5 = getCV(T1w_Bef_list)
T2w_CV_full, T2w_CV, T2w_top_5 = getCV(T2w_Bef_list)
T1w_Z_CV_full, T1w_Z_CV, T1w_Z_top_5 = getCV(T1w_Z_Bef_list)
T2w_Z_CV_full, T2w_Z_CV, T2w_Z_top_5 = getCV(T2w_Z_Bef_list)
T1w_Z_effect_CV_full, T1w_Z_effect_CV, T1w_Z_effect_top_5 = getCV([T1w_Bef_list[1],T1w_Z_Bef_list[1]])
T2w_Z_effect_CV_full, T2w_Z_effect_CV, T2w_Z_effect_top_5 = getCV([T2w_Bef_list[1],T2w_Z_Bef_list[1]])

DCE_CV_full.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_DCE_CV_full.csv")
ADC_CV_full.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_ADC_CV_full.csv")
T1w_CV_full.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T1w_CV_full.csv")
T2w_CV_full.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T2w_CV_full.csv")
T1w_Z_CV_full.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T1w_Z_CV_full.csv")
<<<<<<< HEAD
T2w_Z_CV_full.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T2w_Z_CV_full.csv")
T1w_Z_effect_CV_full.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T1w_Z_effect_CV_full.csv")
T2w_Z_effect_CV_full.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T2w_Z_effect_CV_full.csv")
=======
# T2w_Z_CV_full.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T2w_Z_CV_full.csv")
T1w_Z_effect_CV_full.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T1w_Z_effect_CV_full.csv")
# T2w_Z_effect_CV_full.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T2w_Z_effect_CV_full.csv")
>>>>>>> a6f0e5bc56eb66c70b50ebc81df9bc2547b0c541

DCE_CV.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_DCE_CV_top_5_features.csv")
ADC_CV.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_ADC_CV_top_5_features.csv")
T1w_CV.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T1w_CV_top_5_features.csv")
T2w_CV.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T2w_CV_top_5_features.csv")
T1w_Z_CV.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T1w_Z_CV_top_5_features.csv")
<<<<<<< HEAD
T2w_Z_CV.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T2w_Z_CV_top_5_features.csv")
T1w_Z_effect_CV.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T1w_Z_effect_CV_top_5_features.csv")
T2w_Z_effect_CV.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T2w_Z_effect_CV_top_5_features.csv")
=======
# T2w_Z_CV.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T2w_Z_CV_top_5_features.csv")
T1w_Z_effect_CV.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T1w_Z_effect_CV_top_5_features.csv")
# T2w_Z_effect_CV.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_T2w_Z_effect_CV_top_5_features.csv")
>>>>>>> a6f0e5bc56eb66c70b50ebc81df9bc2547b0c541

# print("DCE:",DCE_top_5)
# print("ADC:",ADC_top_5)
# print("T1w:",T1w_top_5)
# print("T2w:",T2w_top_5)
# print("T1w_Z:",T1w_Z_top_5)
# print("T2w_Z:",T2w_Z_top_5)

top_5_df = pd.concat([DCE_top_5,ADC_top_5,T1w_top_5,T2w_top_5,T1w_Z_top_5,T2w_Z_top_5], axis=1)
top_5_df = top_5_df.rename(columns={0: 'DCE',1:'ADC',2:'T1w',3:'T2w',4:'T1w Z',5:'T2w Z'})
top_5_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_top_5_CV_features.csv")
# print(top_5_df)

# ICC analysis
dce_icc_dict = getICC(DCE_Bef_list,DCE_CV_full)
adc_icc_dict = getICC(ADC_Bef_list,ADC_CV_full)
T1w_icc_dict = getICC(T1w_Bef_list,T1w_CV_full)
T2w_icc_dict = getICC(T2w_Bef_list,T2w_CV_full)
T1w_Z_icc_dict = getICC(T1w_Z_Bef_list,T1w_Z_CV_full)
T2w_Z_icc_dict = getICC(T2w_Z_Bef_list,T2w_Z_CV_full)
T1w_Z_effect_icc_dict = getICC([T1w_Bef_list[1],T1w_Z_Bef_list[1]],T1w_Z_effect_CV_full)
T2w_Z_effect_icc_dict = getICC([T2w_Bef_list[1],T2w_Z_Bef_list[1]],T2w_Z_effect_CV_full)

dce_icc=pd.Series(dce_icc_dict)
adc_icc=pd.Series(adc_icc_dict)
T1w_icc=pd.Series(T1w_icc_dict)
T2w_icc=pd.Series(T2w_icc_dict)
T1w_Z_icc=pd.Series(T1w_Z_icc_dict)
T2w_Z_icc=pd.Series(T2w_Z_icc_dict)
T1w_Z_effect_icc=pd.Series(T1w_Z_effect_icc_dict)
T2w_Z_effect_icc=pd.Series(T2w_Z_effect_icc_dict)
icc_df = pd.concat([dce_icc,adc_icc,T1w_icc,T2w_icc,T1w_Z_icc,T2w_Z_icc,T1w_Z_effect_icc,T1w_Z_effect_icc], axis=1)
icc_df = icc_df.rename(columns={0: 'DCE',1:'ADC',2:'T1w',3:'T2w',4:'T1w Z',5:'T2w Z',6:"T1w Z effect",7:"T2w Z effect"})
print(icc_df)
<<<<<<< HEAD
icc_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_icc_df.csv")

dce_icc_repr=pd.Series(filterICC(dce_icc_dict,0.90))
adc_icc_repr=pd.Series(filterICC(adc_icc_dict,0.90))
T1w_icc_repr=pd.Series(filterICC(T1w_icc_dict,0.90))
T2w_icc_repr=pd.Series(filterICC(T2w_icc_dict,0.90))
T1w_Z_icc_repr=pd.Series(filterICC(T1w_Z_icc_dict,0.90))
T2w_Z_icc_repr=pd.Series(filterICC(T2w_Z_icc_dict,0.90))

icc_repr_df = pd.concat([dce_icc_repr,adc_icc_repr,T1w_icc_repr,T2w_icc_repr,T1w_Z_icc_repr,T2w_Z_icc_repr],axis=1)
icc_repr_df = icc_repr_df.rename(columns={0:"DCE",1:"ADC",2:"T1w",3:"T2w",4:"T1w Z-norm",5:"T2W Z-norm"})
print(icc_repr_df)
icc_repr_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_icc_df_90.csv")

dce_icc_repr=pd.Series(filterICC(dce_icc_dict,0.75))
adc_icc_repr=pd.Series(filterICC(adc_icc_dict,0.75))
T1w_icc_repr=pd.Series(filterICC(T1w_icc_dict,0.75))
T2w_icc_repr=pd.Series(filterICC(T2w_icc_dict,0.75))
T1w_Z_icc_repr=pd.Series(filterICC(T1w_Z_icc_dict,0.75))
T2w_Z_icc_repr=pd.Series(filterICC(T2w_Z_icc_dict,0.75))

icc_repr_df = pd.concat([dce_icc_repr,adc_icc_repr,T1w_icc_repr,T2w_icc_repr,T1w_Z_icc_repr,T2w_Z_icc_repr],axis=1)
icc_repr_df = icc_repr_df.rename(columns={0:"DCE",1:"ADC",2:"T1w",3:"T2w",4:"T1w Z-norm",5:"T2W Z-norm"})
print(icc_repr_df)
icc_repr_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_icc_df_75.csv")
=======
icc_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Bef_PST_contralateral_icc_df.csv")
>>>>>>> a6f0e5bc56eb66c70b50ebc81df9bc2547b0c541
