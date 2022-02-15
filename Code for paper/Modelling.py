#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# Input data (DCE-MRI)
#DCE_volumes=[0.38054322,   0.12242212,   0.45979818,   0.13357694,  -0.04038771, 0.62797084,   0.87002433,   0.12339629,   0.87001655,   0.70248771,   0.66075925,   0.71046931,  -0.07199637,   0.69350888,   0.95480148, -14.12608326]
DCE_volumes=np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0]) # This is a vector of response classifications of the patients (1 = responder)
#DCE_relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_relevant_df.csv",index_col=0) # This is a df of reduced radiomics features. This is the original dataframe used
DCE_relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_relevant_df_Feb_14_22.csv",index_col=0)
number_of_MRI_patients=16
#test_splits_MRI=[[0,1,5],[2,3,6],[4,7,8],[9,10,11,12],[13,14,15]] # Manually calculated test splits
#test_splits=[[0,1,5,2,9],[3,10,11,12,4,6],[13,14,15,7,8]]
#train_splits=[[i for i in range(number_of_patients) if i not in j] for j in test_splits]

# Input data (PET)
#PET_relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/PET_relevant_df.csv",index_col=0) # original dataframe used
PET_relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/PET_relevant_df_Feb_14_22.csv",index_col=0)
#PET_volumes=[-0.04193548,  0.41010401,  0.7903525,   0.23210634,  0.54054054, 0.786372007366482, 0.72744015,  0.65995976, 0.558282208588957, 0.59474091,  0.8634436,   0.64009112,  0.61558442]
PET_volumes=np.array([0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1])
number_of_PET_patients=13
#test_splits_PET=[[i] for i in range(13)] #[[0,1,2],[3,4,5],[6,7,8],[9,10,11,12]]

# Input data (combined DCE-MRI and PET)
#relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_and_PET_relevant_df_PET-CT_vols.csv",index_col=0) # original dataframe used
relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_PET_relevant_df_Feb_14_22.csv",index_col=0)
volumes=np.array([0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1])
number_of_patients=13
#test_splits=[[0,1,2],[3,4,5],[6,7,8],[9,10,11,12]]

# Remaining input data (not yet created)
adc_relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/ADC_relevant_df_Feb_14_22.csv")
t2w_relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/T2W_relevant_df_Feb_14_22.csv")
dce_adc_relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_ADC_relevant_df_Feb_14_22.csv")
dce_t2w_relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_T2W_relevant_df_Feb_14_22.csv")
dce_pet_relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_PET_relevant_df_Feb_14_22.csv")
adc_t2w_relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/ADC_T2W_relevant_df_Feb_14_22.csv")
adc_pet_relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/ADC_PET_relevant_df_Feb_14_22.csv")
t2w_pet_relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/T2W_PET_relevant_df_Feb_14_22.csv")
dce_adc_t2w_relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_ADC_T2W_relevant_df_Feb_14_22.csv")
dce_adc_pet_relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_ADC_PET_relevant_df_Feb_14_22.csv")
adc_t2w_pet_relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/ADC_T2W_PET_relevant_df_Feb_14_22.csv")
dce_adc_t2w_pet_relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_ADC_T2W_PET_relevant_df_Feb_14_22.csv")

def runModelling(volumes,relevant_df,number_of_patients,modality):
    LR = LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5,max_iter=10000)
    ABC = AdaBoostClassifier(n_estimators=100, random_state=0)

    y_pred_LRs=[]
    y_pred_ABCs=[]

    LR_y_pred_probas=np.empty((number_of_patients,2),float)
    ABC_y_pred_probas=np.empty((number_of_patients,2),float)

    for i in range(number_of_patients):
        train_indices = list(range(number_of_patients))
        train_indices.remove(i)
        df_train = relevant_df.iloc[train_indices]
        vol_train=volumes[train_indices]
        df_test = relevant_df.iloc[i]
        #vol_test=volumes[i]

        ss = StandardScaler()
        ss.fit(df_train)
        X_train = ss.transform(df_train)
        X_test = ss.transform([df_test])
        y_train = volumes #np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0])
        #y_test=np.array(y_train[i])
        y_train=list(y_train)
        y_train.pop(i)
        y_train=np.array(y_train)
        #print("y train:",y_train)
        LR.fit(X_train, y_train)
        LR.predict_proba(X_test)

        LR.fit(X_train,vol_train)
        ABC.fit(X_train,vol_train)

        y_pred_LR = LR.predict(X_test)
        y_pred_ABC = ABC.predict(X_test)

        LR_y_pred_proba = LR.predict_proba(X_test)
        ABC_y_pred_proba = ABC.predict_proba(X_test)

        y_pred_LRs.append(y_pred_LR)
        y_pred_ABCs.append(y_pred_ABC)

        LR_y_pred_probas[i,:]=LR_y_pred_proba
        ABC_y_pred_probas[i,:]=ABC_y_pred_proba

    print(f"Number of LR mislabeled points out of a total {volumes.shape[0]} points : {np.sum(volumes != np.array(y_pred_LRs).reshape(1,-1)[0])}")
    print(f"Number of ABC mislabeled points out of a total {volumes.shape[0]} points : {np.sum(volumes != np.array(y_pred_ABCs).reshape(1,-1)[0])}")

    """
    Compute and plot an ROC curve
    """

    fpr, tpr, _ = roc_curve(volumes, LR_y_pred_probas[:,1])
    fig,ax = plt.subplots(1,1)
    ax.plot(fpr, tpr)
    ax.plot([0,1],[0,1],c='r', lw=2, ls='--')
    plt.title(f"ROC curve for Gaussian Naive Bayes in {modality}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    print(f"AUC for LR in {modality}:",metrics.auc(fpr, tpr))
    plt.savefig(f'/home/alicja/PET-LAB Code/PET-LAB/Radiomics/ROC_{modality}_LR_7_Feb_22.png')

    fpr, tpr, _ = roc_curve(volumes, ABC_y_pred_probas[:,1])
    fig,ax = plt.subplots(1,1)
    ax.plot(fpr, tpr)
    ax.plot([0,1],[0,1],c='r', lw=2, ls='--')
    plt.title(f"ROC curve for Gaussian Naive Bayes in {modality}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    print(f"AUC for ABC in {modality}:",metrics.auc(fpr, tpr))
    plt.savefig(f'/home/alicja/PET-LAB Code/PET-LAB/Radiomics/ROC_{modality}_ABC_7_Feb_22.png')

    """
    Accuracy, sensitivity, specificity
    """

    # Confusion matrix, Accuracy, sensitivity and specificity

    cm1_LR = confusion_matrix(volumes,y_pred_LRs)
    print('Confusion Matrix LR: \n', cm1_LR)

    total1=sum(sum(cm1_LR))
    # from confusion matrix calculate accuracy
    accuracy1=(cm1_LR[0,0]+cm1_LR[1,1])/total1
    print ('Accuracy LR : ', accuracy1)

    sensitivity1 = cm1_LR[0,0]/(cm1_LR[0,0]+cm1_LR[0,1])
    print('Sensitivity LR : ', sensitivity1 )

    specificity1 = cm1_LR[1,1]/(cm1_LR[1,0]+cm1_LR[1,1])
    print('Specificity LR : ', specificity1)

    cm1_ABC = confusion_matrix(volumes,y_pred_ABCs)
    print('Confusion Matrix ABC: \n', cm1_ABC)

    total1=sum(sum(cm1_ABC))

    ##from confusion matrix calculate accuracy
    accuracy1=(cm1_ABC[0,0]+cm1_ABC[1,1])/total1
    print ('Accuracy ABC: ', accuracy1)

    sensitivity1 = cm1_ABC[0,0]/(cm1_ABC[0,0]+cm1_ABC[0,1])
    print('Sensitivity ABC: ', sensitivity1 )

    specificity1 = cm1_ABC[1,1]/(cm1_ABC[1,0]+cm1_ABC[1,1])
    print('Specificity ABC: ', specificity1)

runModelling(DCE_volumes,DCE_relevant_df,number_of_MRI_patients,"DCE")
runModelling(PET_volumes,PET_relevant_df,number_of_PET_patients,"PET")
runModelling(volumes,relevant_df,number_of_patients,"DCE_and_PET")

# The following have not been generated yet
runModelling(volumes,adc_relevant_df,number_of_patients,"ADC")
runModelling(volumes,t2w_relevant_df,number_of_patients,"T2W")
runModelling(volumes,dce_adc_relevant_df,number_of_patients,"DCE_and_ADC")
runModelling(volumes,dce_t2w_relevant_df,number_of_patients,"DCE_and_T2W")
runModelling(volumes,dce_pet_relevant_df,number_of_patients,"DCE_and_PET")
runModelling(volumes,adc_t2w_relevant_df,number_of_patients,"ADC_and_T2W")
runModelling(volumes,adc_pet_relevant_df,number_of_patients,"ADC_and_PET")
runModelling(volumes,t2w_pet_relevant_df,number_of_patients,"T2W_and_PET")
runModelling(volumes,dce_adc_t2w_relevant_df,number_of_patients,"DCE_and_ADC_and_T2w")
runModelling(volumes,dce_adc_pet_relevant_df,number_of_patients,"DCE_and_ADC_and_PET")
runModelling(volumes,adc_t2w_pet_relevant_df,number_of_patients,"ADC_and_T2W_and_PET")
runModelling(volumes,dce_adc_t2w_pet_relevant_df,number_of_patients,"DCE_and_ADC_and_T2W_and_PET")