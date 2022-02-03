#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler 

LR = LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5,max_iter=10000)
ABC = AdaBoostClassifier(n_estimators=100, random_state=0)

#kf = KFold(n_splits=5)

#DCE_volumes=[0.38054322,   0.12242212,   0.45979818,   0.13357694,  -0.04038771, 0.62797084,   0.87002433,   0.12339629,   0.87001655,   0.70248771,   0.66075925,   0.71046931,  -0.07199637,   0.69350888,   0.95480148, -14.12608326]
DCE_volumes=np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0])
DCE_relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_relevant_df.csv",index_col=0)
LRaccuracies=[]
ABCaccuracies=[]
LR_AUCs=[]
ABC_AUCs=[]

y_pred_LRs=[]
y_pred_ABCs=[]

LR_y_pred_probas=np.empty((16,2),float)
ABC_y_pred_probas=np.empty((16,2),float)
LR_fprs=[]
LR_tprs=[]
ABC_fprs=[]
ABC_tprs=[]

LR_sensitivities=[]
LR_specificities=[]
ABC_sensitivities=[]
ABC_specificities=[]

test_splits=[[0,1,5],[2,3,6],[4,7,8],[9,10,11,12],[13,14,15]]
#test_splits=[[0,1,5,2,9],[3,10,11,12,4,6],[13,14,15,7,8]]
train_splits=[[i for i in range(16) if i not in j] for j in test_splits]

for i in range(16):
    train_indices = list(range(16))
    train_indices.remove(i)
    df_train = DCE_relevant_df.iloc[train_indices]
    DCE_vol_train=DCE_volumes[train_indices]
    df_test = DCE_relevant_df.iloc[i]
    DCE_vol_test=DCE_volumes[i]

    ss = StandardScaler()
    ss.fit(df_train)
    X_train = ss.transform(df_train)
    X_test = ss.transform([df_test])
    y_train = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0])
    y_test=np.array(y_train[i])
    y_train=list(y_train)
    y_train.pop(i)
    y_train=np.array(y_train)
    #print("y train:",y_train)
    LR.fit(X_train, y_train)
    LR.predict_proba(X_test)

    LR.fit(X_train,DCE_vol_train)
    ABC.fit(X_train,DCE_vol_train)

    y_pred_LR = LR.predict(X_test)

    #for gausian NB
    y_pred_ABC = ABC.predict(X_test)

    LR_y_pred_proba = LR.predict_proba(X_test)
    ABC_y_pred_proba = ABC.predict_proba(X_test)

    y_pred_LRs.append(y_pred_LR)
    y_pred_ABCs.append(y_pred_ABC)

    LR_y_pred_probas[i,:]=LR_y_pred_proba
    ABC_y_pred_probas[i,:]=ABC_y_pred_proba

print(f"Number of LR mislabeled points out of a total {DCE_volumes.shape[0]} points : {np.sum(DCE_volumes != np.array(y_pred_LRs).reshape(1,-1)[0])}")
print(f"Number of ABC mislabeled points out of a total {DCE_volumes.shape[0]} points : {np.sum(DCE_volumes != np.array(y_pred_ABCs).reshape(1,-1)[0])}")

"""
Compute and plot an ROC curve
"""

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(DCE_volumes, LR_y_pred_probas[:,1])
fig,ax = plt.subplots(1,1)
ax.plot(fpr, tpr)
ax.plot([0,1],[0,1],c='r', lw=2, ls='--')
plt.title("ROC curve for Logistic Regression in DCE-MRI")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print("AUC for LR in DCE-MRI:",metrics.auc(fpr, tpr))
plt.savefig('/home/alicja/PET-LAB Code/PET-LAB/Radiomics/new_ROC_DCE_LR.png')

fpr, tpr, _ = roc_curve(DCE_volumes, ABC_y_pred_probas[:,1])
fig,ax = plt.subplots(1,1)
ax.plot(fpr, tpr)
ax.plot([0,1],[0,1],c='r', lw=2, ls='--')
plt.title("ROC curve for Gaussian Naive Bayes in DCE-MRI")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print("AUC for ABC in DCE-MRI:",metrics.auc(fpr, tpr))
plt.savefig('/home/alicja/PET-LAB Code/PET-LAB/Radiomics/new_ROC_DCE_ABC.png')

"""
Accuracy, sensitivity, specificity
"""

from sklearn.metrics import confusion_matrix

#Confusion matrix, Accuracy, sensitivity and specificity

cm1_LR = confusion_matrix(DCE_volumes,y_pred_LRs)
print('Confusion Matrix LR: \n', cm1_LR)

total1=sum(sum(cm1_LR))
##from confusion matrix calculate accuracy
accuracy1=(cm1_LR[0,0]+cm1_LR[1,1])/total1
print ('Accuracy LR : ', accuracy1)

sensitivity1 = cm1_LR[0,0]/(cm1_LR[0,0]+cm1_LR[0,1])
print('Sensitivity LR : ', sensitivity1 )

specificity1 = cm1_LR[1,1]/(cm1_LR[1,0]+cm1_LR[1,1])
print('Specificity LR : ', specificity1)

cm1_ABC = confusion_matrix(DCE_volumes,y_pred_ABCs)
print('Confusion Matrix ABC: \n', cm1_ABC)

total1=sum(sum(cm1_ABC))
##from confusion matrix calculate accuracy
accuracy1=(cm1_ABC[0,0]+cm1_ABC[1,1])/total1
print ('Accuracy ABC: ', accuracy1)

sensitivity1 = cm1_ABC[0,0]/(cm1_ABC[0,0]+cm1_ABC[0,1])
print('Sensitivity ABC: ', sensitivity1 )

specificity1 = cm1_ABC[1,1]/(cm1_ABC[1,0]+cm1_ABC[1,1])
print('Specificity ABC: ', specificity1)