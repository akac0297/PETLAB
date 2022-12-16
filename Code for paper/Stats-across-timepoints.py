import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg

MRI_patient_list=("06","14","15","16","18","19","21","23")
modalities = ["DCE ME", "T2w SPAIR", "T1w", "ADC"]
binCounts = ["32","64","128","256","None"]

def splitDfs(patient_id,binCount,modality,normalised):
    df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/Radiomics_features_contralateral_WES_0{}_binCount{}_df.csv".format(patient_id,binCount))
    df=pd.concat([df.iloc[:,0:19],df.iloc[:,33:]],axis=1) # Exclude shape features from "across time points" analysis
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

# extract columns with the same name (they should be in the same order) and calculate relative differences
def get_relative_diff(df1, df2):
    feat_baseline=df1
    feat_validation=df2
    rel_diff=(feat_baseline - feat_validation)/((feat_baseline + feat_validation)/2)
    rel_diff=rel_diff.replace(np.nan,0)
    return rel_diff

# assess normality of the relative differences using the Shapiro-Wilk test:
# - if p>0.05, the null hypothesis is accepted and the sample is from a normal distribution
# - if p<0.05, the sample is not from a normal distribution

def ShapiroWilk(df):
    SW_df=df.copy(deep=True)
    SW_df=SW_df.iloc[0]
    for column in df:
        shapiro_test=stats.shapiro(df[column])
        SW_df[column]=shapiro_test.pvalue
    return SW_df

def filtersw(SW_df):
    df = SW_df[SW_df > 0.05]
    return df

# get mean and standard deviation estimates from the relative differences
# also get a 95% CI for the lower and upper limits of repeatability: x +/- 1.96 x SD/sqrt(n)
def getStats(df,df_normal):
    n=8
    mean_df=pd.DataFrame()
    sd_df=pd.DataFrame()
    CIs=pd.DataFrame()
    features=list(df_normal.index)
    df=df[features]
    mean_df=df.mean(axis=0)
    sd_df=df.std(axis=0)
    CIs={"Lower CI value":100*(mean_df-1.96*sd_df/np.sqrt(n)), "Higher CI value":100*(mean_df+1.96*sd_df/np.sqrt(n))}
    CIs=pd.DataFrame(CIs)
    stats_df = pd.concat([mean_df,sd_df,CIs],axis=1)
    stats_df = stats_df.rename(columns={0: 'Mean',1:'Std'})
    return stats_df

# This ICC can be used for the case of Bef_df, Dur_df, and Post_df, but also a single time point across different pre-processing steps e.g. different binCounts.
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

# Combine the information for all the patients
for binCount in binCounts:
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

    DCE_CV_full, DCE_CV, DCE_top_5 = getCV([DCE_Bef,DCE_Dur,DCE_Post])
    ADC_CV_full, ADC_CV, ADC_top_5 = getCV([ADC_Bef,ADC_Dur,ADC_Post])
    T1w_CV_full, T1w_CV, T1w_top_5 = getCV([T1w_Bef,T1w_Dur,T1w_Post])
    T2w_CV_full, T2w_CV, T2w_top_5 = getCV([T2w_Bef,T2w_Dur,T2w_Post])
    T1w_Z_CV_full, T1w_Z_CV, T1w_Z_top_5 = getCV([T1w_Z_Bef,T1w_Z_Dur,T1w_Z_Post])
    T2w_Z_CV_full, T2w_Z_CV, T2w_Z_top_5 = getCV([T2w_Z_Bef,T2w_Z_Dur,T2w_Z_Post])

    DCE_CV_full.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_DCE_CV_full.csv".format(binCount))
    ADC_CV_full.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_ADC_CV_full.csv".format(binCount))
    T1w_CV_full.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_T1w_CV_full.csv".format(binCount))
    T2w_CV_full.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_T2w_CV_full.csv".format(binCount))
    T1w_Z_CV_full.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_T1w_Z_CV_full.csv".format(binCount))
    # T2w_Z_CV_full.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_T2w_Z_CV_full.csv".format(binCount))
    
    DCE_CV.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_DCE_CV_top_5_features.csv".format(binCount))
    ADC_CV.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_ADC_CV_top_5_features.csv".format(binCount))
    T1w_CV.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_T1w_CV_top_5_features.csv".format(binCount))
    T2w_CV.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_T2w_CV_top_5_features.csv".format(binCount))
    T1w_Z_CV.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_T1w_Z_CV_top_5_features.csv".format(binCount))
    # T2w_Z_CV.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_T2w_Z_CV_top_5_features.csv".format(binCount))

    # print("DCE:",DCE_top_5)
    # print("ADC:",ADC_top_5)
    # print("T1w:",T1w_top_5)
    # print("T2w:",T2w_top_5)
    # print("T1w_Z:",T1w_Z_top_5)
    # print("T2w_Z:",T2w_Z_top_5)

    top_5_df = pd.concat([DCE_top_5,ADC_top_5,T1w_top_5,T2w_top_5,T1w_Z_top_5,T2w_Z_top_5], axis=1)
    top_5_df = top_5_df.rename(columns={0: 'DCE',1:'ADC',2:'T1w',3:'T2w',4:'T1w Z',5:'T2w Z'})
    top_5_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_top_5_CV_features.csv".format(binCount))
    # print(top_5_df)

    # Relative differences

    diff_ADC_12 = get_relative_diff(ADC_Bef,ADC_Dur)
    diff_ADC_13 = get_relative_diff(ADC_Bef,ADC_Post)
    #print(diff_ADC_12)

    diff_DCE_12 = get_relative_diff(DCE_Bef,DCE_Dur)
    diff_DCE_13 = get_relative_diff(DCE_Bef,DCE_Post)
    # print(diff_DCE_12)
    # print(diff_DCE_13)

    diff_T1w_Z_12 = get_relative_diff(T1w_Z_Bef,T1w_Z_Dur)
    diff_T1w_Z_13 = get_relative_diff(T1w_Z_Bef,T1w_Z_Post)
    # print(diff_T1w_Z_12)
    # print(diff_T1w_Z_13)

    diff_T1w_12 = get_relative_diff(T1w_Bef,T1w_Dur)
    diff_T1w_13 = get_relative_diff(T1w_Bef,T1w_Post)
    # print(diff_T1w_12)
    # print(diff_T1w_13)

    diff_T2w_Z_12 = get_relative_diff(T2w_Z_Bef,T2w_Z_Dur)
    diff_T2w_Z_13 = get_relative_diff(T2w_Z_Bef,T2w_Z_Post)
    # print(diff_T2w_Z_12)
    # print(diff_T2w_Z_13)

    diff_T2w_12 = get_relative_diff(T2w_Bef,T2w_Dur)
    diff_T2w_13 = get_relative_diff(T2w_Bef,T2w_Post)
    # print(diff_T2w_12)
    # print(diff_T2w_13)

    ADC_SW_12=ShapiroWilk(diff_ADC_12)
    ADC_SW_12_normal=filtersw(ADC_SW_12)
    DCE_SW_12=ShapiroWilk(diff_DCE_12)
    DCE_SW_12_normal=filtersw(DCE_SW_12)
    T1w_SW_12=ShapiroWilk(diff_T1w_12)
    T1w_SW_12_normal=filtersw(T1w_SW_12)
    T2w_SW_12=ShapiroWilk(diff_T2w_12)
    T2w_SW_12_normal=filtersw(T2w_SW_12)
    T1w_Z_SW_12=ShapiroWilk(diff_T1w_Z_12)
    T1w_Z_SW_12_normal=filtersw(T1w_Z_SW_12)
    T2w_Z_SW_12=ShapiroWilk(diff_T2w_Z_12)
    T2w_Z_SW_12_normal=filtersw(T2w_Z_SW_12)
    #print(ADC_SW_12_normal)

    ADC_SW_13=ShapiroWilk(diff_ADC_13)
    ADC_SW_13_normal=filtersw(ADC_SW_13)
    DCE_SW_13=ShapiroWilk(diff_DCE_13)
    DCE_SW_13_normal=filtersw(DCE_SW_13)
    T1w_SW_13=ShapiroWilk(diff_T1w_13)
    T1w_SW_13_normal=filtersw(T1w_SW_13)
    T2w_SW_13=ShapiroWilk(diff_T2w_13)
    T2w_SW_13_normal=filtersw(T2w_SW_13)
    T1w_Z_SW_13=ShapiroWilk(diff_T1w_Z_13)
    T1w_Z_SW_13_normal=filtersw(T1w_Z_SW_13)
    T2w_Z_SW_13=ShapiroWilk(diff_T2w_Z_13)
    T2w_Z_SW_13_normal=filtersw(T2w_Z_SW_13)

    # For duplicated indices:
    #df[df.index.duplicated(keep=False)]
    # Returns a dataframe with duplicated values e.g. array([True, False, True, False, True]) for ['lama', 'cow', 'lama', 'beetle', 'lama']
    # not sure, may be able to differentiate between these inherently if pandas accounts for this

    ADC_stats_df=getStats(diff_ADC_12,ADC_SW_12_normal)
    DCE_stats_df=getStats(diff_DCE_12,DCE_SW_12_normal)
    T1w_stats_df=getStats(diff_T1w_12,T1w_SW_12_normal)
    T2w_stats_df=getStats(diff_T2w_12,T2w_SW_12_normal)
    T1w_Z_stats_df=getStats(diff_T1w_Z_12,T1w_Z_SW_12_normal)
    T2w_Z_stats_df=getStats(diff_T2w_Z_12,T2w_Z_SW_12_normal)
    # print(ADC_stats_df)

    ADC_stats_df_13=getStats(diff_ADC_13,ADC_SW_13_normal)
    DCE_stats_df_13=getStats(diff_DCE_13,DCE_SW_13_normal)
    T1w_stats_df_13=getStats(diff_T1w_13,T1w_SW_13_normal)
    T2w_stats_df_13=getStats(diff_T2w_13,T2w_SW_13_normal)
    T1w_Z_stats_df_13=getStats(diff_T1w_Z_13,T1w_Z_SW_13_normal)
    T2w_Z_stats_df_13=getStats(diff_T2w_Z_13,T2w_Z_SW_13_normal)

    ADC_stats_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_ADC_stats_12.csv".format(binCount))
    DCE_stats_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_DCE_stats_12.csv".format(binCount))
    T1w_stats_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_T1w_stats_12.csv".format(binCount))
    T2w_stats_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_T2w_stats_12.csv".format(binCount))
    T1w_Z_stats_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_T1w_Z_stats_12.csv".format(binCount))
    # T2w_Z_stats_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_T2w_Z_stats_12.csv".format(binCount))

    ADC_stats_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_ADC_stats_13.csv".format(binCount))
    DCE_stats_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_DCE_stats_13.csv".format(binCount))
    T1w_stats_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_T1w_stats_13.csv".format(binCount))
    T2w_stats_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_T2w_stats_13.csv".format(binCount))
    T1w_Z_stats_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_T1w_Z_stats_13.csv".format(binCount))
    # T2w_Z_stats_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_T2w_Z_stats_13.csv".format(binCount))

    # ICC analysis
    dce_icc_dict = getICC([DCE_Bef,DCE_Dur,DCE_Post],DCE_CV_full)
    adc_icc_dict = getICC([ADC_Bef,ADC_Dur,ADC_Post],ADC_CV_full)
    T1w_icc_dict = getICC([T1w_Bef,T1w_Dur,T1w_Post],T1w_CV_full)
    T2w_icc_dict = getICC([T2w_Bef,T2w_Dur,T2w_Post],T2w_CV_full)
    T1w_Z_icc_dict = getICC([T1w_Z_Bef,T1w_Z_Dur,T1w_Z_Post],T1w_Z_CV_full)
    T2w_Z_icc_dict = getICC([T2w_Z_Bef,T2w_Z_Dur,T2w_Z_Post],T2w_Z_CV_full)
    #dce_icc_rep = filterICC(dce_icc_dict,0.75)
    #print(dce_icc_rep)
    dce_icc=pd.Series(dce_icc_dict)
    print(dce_icc)
    adc_icc=pd.Series(adc_icc_dict)
    T1w_icc=pd.Series(T1w_icc_dict)
    T2w_icc=pd.Series(T2w_icc_dict)
    T1w_Z_icc=pd.Series(T1w_Z_icc_dict)
    T2w_Z_icc=pd.Series(T2w_Z_icc_dict)
    icc_df = pd.concat([dce_icc,adc_icc,T1w_icc,T2w_icc,T1w_Z_icc,T2w_Z_icc], axis=1)
    icc_df = icc_df.rename(columns={0: 'DCE',1:'ADC',2:'T1w',3:'T2w',4:'T1w Z',5:'T2w Z'})
    print(icc_df)
    icc_df.to_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/BinCount{}_contralateral_icc_df.csv".format(binCount))