
""""
ElasticNet regression model

"""
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler 

LR = LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5,max_iter=10000)
GNB = GaussianNB()

cross_validator = RepeatedKFold(n_splits=3, n_repeats=1, random_state=42)

PET_relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/PET_relevant_df.csv")
#PET_volumes=[-0.04193548,  0.41010401,  0.7903525,   0.23210634,  0.54054054, 0.786372007366482, 0.72744015,  0.65995976, 0.558282208588957, 0.59474091,  0.8634436,   0.64009112,  0.61558442]
PET_volumes=np.array([0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1])
#DCE_volumes=[0.38054322,   0.12242212,   0.45979818,   0.13357694,  -0.04038771, 0.62797084,   0.87002433,   0.12339629,   0.87001655,   0.70248771,   0.66075925,   0.71046931,  -0.07199637,   0.69350888,   0.95480148, -14.12608326]
DCE_volumes=np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0])
DCE_relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/DCE_relevant_df.csv")

test_splits=[[i] for i in range(13)] #[[0,1,2],[3,4,5],[6,7,8],[9,10,11,12]]
train_splits=[[i for i in range(13) if i not in j] for j in test_splits]

LR_predictions=[]
GNB_predictions=[]
LR_y_pred_probas=np.empty((13,2),float)
GNB_y_pred_probas=np.empty((13,2),float)

#for train_index, test_index in zip(train_splits,test_splits):

for i in range(13):
    train_indices = list(range(13))
    train_indices.remove(i)
    df_train = PET_relevant_df.iloc[train_indices]
    PET_vol_train=PET_volumes[train_indices]
    df_test = PET_relevant_df.iloc[i]
    PET_vol_test=PET_volumes[i]

    ss = StandardScaler()
    ss.fit(df_train)
    X_train = ss.transform(df_train)
    X_test = ss.transform([df_test])
    y_train = np.array([0,1,1,0,0,1,1,1,0,0,1,0,1])
    y_test=np.array(y_train[i])
    y_train=list(y_train)
    y_train.pop(i)
    y_train=np.array(y_train)
    print("y train:",y_train)
    #len(y_train)
    #y_train = [0,1,1,0,0,1,1,1,0,0,1,0]
    LR.fit(X_train, y_train)
    LR.predict_proba(X_test)

    #PET_train, PET_test = X[train_index], X[test_index]
    #PET_vol_train, PET_vol_test = PET_volumes[train_index], PET_volumes[test_index]

    LR.fit(X_train,PET_vol_train)
    GNB.fit(X_train,PET_vol_train)

    #LR_scores = cross_val_score(LR, X, PET_volumes, scoring="neg_mean_absolute_error", cv=cross_validator)
    #print("LR:", LR_scores.mean(), "+/-", LR_scores.std())

    #GNB_scores = cross_val_score(GNB, X, PET_volumes, scoring="neg_mean_absolute_error", cv=cross_validator)
    #print("GNB:", GNB_scores.mean(), "+/-", GNB_scores.std())

    #for LR
    importance=LR.coef_
    print("Importance for LR PET: ",importance)

    y_pred_LR = LR.fit(X_train, PET_vol_train).predict(X_test)
    print("LR predicted PET vols: ",y_pred_LR)
    print("Correct PET vols: ",PET_vol_test)
    print(f"Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], np.sum(PET_vol_test != y_pred_LR)))

    #for gausian NB
    y_pred_GNB = GNB.fit(X_train, PET_vol_train).predict(X_test)
    print("GNB predicted PET vols: ",y_pred_GNB)
    print("Correct PET vols: ",PET_vol_test)
    print(f"Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], np.sum(PET_vol_test != y_pred_GNB)))

    #for i,v in enumerate(importance):
    #    print('Feature: %0d, Score: %.5f' % (i,v))

    #LR_y_pred = LR.predict(PET_test)
    #print(LR_y_pred)

    LR_y_pred_proba = LR.predict_proba(X_test)
    GNB_y_pred_proba = GNB.predict_proba(X_test)

    LR_predictions.append(y_pred_LR)
    GNB_predictions.append(y_pred_GNB)

    LR_y_pred_probas[i,:]=LR_y_pred_proba
    GNB_y_pred_probas[i,:]=GNB_y_pred_proba

print(LR_predictions)
print(GNB_predictions)
print(PET_volumes)

print(LR_y_pred_probas)
print(GNB_y_pred_probas)

"""
Compute and plot an ROC curve
"""

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(PET_volumes, LR_y_pred_probas[:,1])
fig,ax = plt.subplots(1,1)
ax.plot(fpr, tpr)
ax.plot([0,1],[0,1],c='r', lw=2, ls='--')
plt.title("ROC curve for Logistic Regression in PET")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print("AUC for LR in PET:",metrics.auc(fpr, tpr))
plt.savefig('/home/alicja/PET-LAB Code/PET-LAB/Radiomics/new_ROC_PET_LR.png')

fpr, tpr, _ = roc_curve(PET_volumes, GNB_y_pred_probas[:,1])
fig,ax = plt.subplots(1,1)
ax.plot(fpr, tpr)
ax.plot([0,1],[0,1],c='r', lw=2, ls='--')
plt.title("ROC curve for Gaussian Naive Bayes in PET")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print("AUC for GNB in PET:",metrics.auc(fpr, tpr))
plt.savefig('/home/alicja/PET-LAB Code/PET-LAB/Radiomics/new_ROC_PET_GNB.png')

"""
Accuracy, sensitivity, specificity
"""

from sklearn.metrics import confusion_matrix

#Confusion matrix, Accuracy, sensitivity and specificity

cm1_LR = confusion_matrix(PET_volumes,LR_predictions)
print('Confusion Matrix LR: \n', cm1_LR)

total1=sum(sum(cm1_LR))
##from confusion matrix calculate accuracy
accuracy1=(cm1_LR[0,0]+cm1_LR[1,1])/total1
print ('Accuracy LR : ', accuracy1)

sensitivity1 = cm1_LR[0,0]/(cm1_LR[0,0]+cm1_LR[0,1])
print('Sensitivity LR : ', sensitivity1 )

specificity1 = cm1_LR[1,1]/(cm1_LR[1,0]+cm1_LR[1,1])
print('Specificity LR : ', specificity1)

cm1_GNB = confusion_matrix(PET_volumes,GNB_predictions)
print('Confusion Matrix GNB: \n', cm1_GNB)

total1=sum(sum(cm1_GNB))
##from confusion matrix calculate accuracy
accuracy1=(cm1_GNB[0,0]+cm1_GNB[1,1])/total1
print ('Accuracy GNB: ', accuracy1)

sensitivity1 = cm1_GNB[0,0]/(cm1_GNB[0,0]+cm1_GNB[0,1])
print('Sensitivity GNB: ', sensitivity1 )

specificity1 = cm1_GNB[1,1]/(cm1_GNB[1,0]+cm1_GNB[1,1])
print('Specificity GNB: ', specificity1)

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
#plt.figure()
#plot_confusion_matrix(LR, PET_test, PET_vol_test)
#plt.title("Confusion matrix for PET Logistic Regression model")
#plt.savefig("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/confusion_matrix_PET_LR.png")

#plt.figure()
#plot_confusion_matrix(GNB, PET_test, PET_vol_test)  
#plt.title("Confusion matrix for PET Gaussian Naive Bayes model")
#plt.savefig("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/confusion_matrix_PET_GNB.png")


# fpr, tpr, _ = roc_curve(PET_vol_test, LR_y_pred_proba[:,1])
# fig,ax = plt.subplots(1,1)
# ax.plot(fpr, tpr)
# ax.plot([0,1],[0,1],c='r', lw=2, ls='--')
# plt.title("ROC curve for Logistic Regression in PET")
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# print("AUC for LR in PET:",metrics.auc(fpr, tpr))
# #plt.savefig('/home/alicja/PET-LAB Code/PET-LAB/Radiomics/ROC_PET_LR.png')

# fpr, tpr, _ = roc_curve(PET_vol_test, GNB_y_pred_proba[:,1])
# fig,ax = plt.subplots(1,1)
# ax.plot(fpr, tpr)
# ax.plot([0,1],[0,1],c='r', lw=2, ls='--')
# plt.title("ROC curve for Gaussian Naive Bayes in PET")
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# print("AUC for GNB in PET:",metrics.auc(fpr, tpr))
# #plt.savefig('/home/alicja/PET-LAB Code/PET-LAB/Radiomics/ROC_PET_GNB.png')

#     #plt.savefig('/home/alicja/PET-LAB Code/PET-LAB/Radiomics/AUC_DCE_MRI_GNB.png')
#     #plt.savefig('/home/alicja/PET-LAB Code/PET-LAB/Radiomics/AUC_DCE_MRI_LR.png')
#     #plt.savefig('/home/alicja/PET-LAB Code/PET-LAB/Radiomics/AUC_PET_LR.png')
#     #plt.savefig('/home/alicja/PET-LAB Code/PET-LAB/Radiomics/AUC_PET_GNB.png')

#     """
#     Compute and plot an ROC curve
#     """
#     """
#     Accuracy, sensitivity, specificity
#     """

#from sklearn.metrics import confusion_matrix

#     #Confusion matrix, Accuracy, sensitivity and specificity

# cm1_LR = confusion_matrix(PET_vol_test,y_pred_LR)
# print('Confusion Matrix LR: \n', cm1_LR)

# total1=sum(sum(cm1_LR))
# ##from confusion matrix calculate accuracy
# accuracy1=(cm1_LR[0,0]+cm1_LR[1,1])/total1
# print ('Accuracy LR : ', accuracy1)

# sensitivity1 = cm1_LR[0,0]/(cm1_LR[0,0]+cm1_LR[0,1])
# print('Sensitivity LR : ', sensitivity1 )

# specificity1 = cm1_LR[1,1]/(cm1_LR[1,0]+cm1_LR[1,1])
# print('Specificity LR : ', specificity1)

# cm1_GNB = confusion_matrix(PET_vol_test,y_pred_GNB)
# print('Confusion Matrix GNB: \n', cm1_GNB)

# total1=sum(sum(cm1_GNB))
# ##from confusion matrix calculate accuracy
# accuracy1=(cm1_GNB[0,0]+cm1_GNB[1,1])/total1
# print ('Accuracy GNB: ', accuracy1)

# sensitivity1 = cm1_GNB[0,0]/(cm1_GNB[0,0]+cm1_GNB[0,1])
# print('Sensitivity GNB: ', sensitivity1 )

# specificity1 = cm1_GNB[1,1]/(cm1_GNB[1,0]+cm1_GNB[1,1])
# print('Specificity GNB: ', specificity1)

# # import matplotlib.pyplot as plt
# # from sklearn.metrics import plot_confusion_matrix
# # plt.figure()
# # plot_confusion_matrix(LR, PET_test, PET_vol_test)
# # plt.title("Confusion matrix for PET Logistic Regression model")
# # #plt.savefig("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/confusion_matrix_PET_LR.png")

# # plt.figure()
# # plot_confusion_matrix(GNB, PET_test, PET_vol_test)  
# # plt.title("Confusion matrix for PET Gaussian Naive Bayes model")
# # #plt.savefig("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/confusion_matrix_PET_GNB.png")