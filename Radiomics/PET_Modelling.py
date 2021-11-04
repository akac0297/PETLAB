from sklearn.model_selection import RepeatedKFold
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler

LR = LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5,max_iter=10000)
GNB = GaussianNB()

cross_validator = RepeatedKFold(n_splits=3, n_repeats=1, random_state=42)

PET_relevant_df=pd.read_csv("/home/alicja/PET-LAB Code/PET-LAB/Radiomics/PET_relevant_df.csv",index_col=0)
#PET_volumes=[-0.04193548,  0.41010401,  0.7903525,   0.23210634,  0.54054054, 0.786372007366482, 0.72744015,  0.65995976, 0.558282208588957, 0.59474091,  0.8634436,   0.64009112,  0.61558442]
PET_volumes=np.array([0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1])

test_splits=[[i] for i in range(13)] #[[0,1,2],[3,4,5],[6,7,8],[9,10,11,12]]
train_splits=[[i for i in range(13) if i not in j] for j in test_splits]

LR_predictions=[]
GNB_predictions=[]
LR_y_pred_probas=np.empty((13,2),float)
GNB_y_pred_probas=np.empty((13,2),float)

for i in range(13):
    train_indices = list(range(13))
    train_indices.remove(i)
    df_train = PET_relevant_df.iloc[train_indices]
    print("df train",df_train)
    PET_vol_train=PET_volumes[train_indices]
    df_test = PET_relevant_df.iloc[i]
    PET_vol_test=PET_volumes[i]

    ss = StandardScaler() #MinMaxScaler()
    ss.fit(PET_relevant_df)
    X_train = ss.transform(df_train)
    print("X train",X_train)
    X_test = ss.transform([df_test])
    print("X test",X_test)
    print("PET vol train",PET_vol_train)
    # y_train = np.array([0,1,1,0,0,1,1,1,0,0,1,0,1])
    # y_train=PET_volumes
    # y_test=np.array(y_train[i])
    # y_train=list(y_train)
    # y_train.pop(i)
    # y_train=np.array(y_train)
    # print("y train:",y_train)
    # #LR.fit(X_train, y_train)
    # #LR.predict_proba(X_test)

    LR.fit(X_train,PET_vol_train)
    GNB.fit(X_train,PET_vol_train)

    #for LR
    importance=LR.coef_
    print("Importance for LR PET: ",importance)

    y_pred_LR = LR.predict(X_test)
    print("LR predicted PET vols: ",y_pred_LR)
    print("Correct PET vols: ",PET_vol_test)
    print(f"Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], np.sum(PET_vol_test != y_pred_LR)))

    #for gausian NB
    y_pred_GNB = GNB.predict(X_test)
    print("GNB predicted PET vols: ",y_pred_GNB)
    print("Correct PET vols: ",PET_vol_test)
    print(f"Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], np.sum(PET_vol_test != y_pred_GNB)))

    LR_y_pred_proba = LR.predict_proba(X_test)
    GNB_y_pred_proba = GNB.predict_proba(X_test)
    print("LR y pred proba",LR_y_pred_proba)
    print("predict training set",LR.predict(X_train))

    LR_predictions.append(y_pred_LR)
    GNB_predictions.append(y_pred_GNB)

    LR_y_pred_probas[i,:]=LR_y_pred_proba
    GNB_y_pred_probas[i,:]=GNB_y_pred_proba

print(LR_predictions)
print(GNB_predictions)
print(PET_volumes)

print(LR_y_pred_probas)
print(GNB_y_pred_probas)

print(f"Number of LR mislabeled points out of a total {PET_volumes.shape[0]} points : {np.sum(PET_volumes != np.array(LR_predictions).reshape(1,-1)[0])}")
print(f"Number of GNB mislabeled points out of a total {PET_volumes.shape[0]} points : {np.sum(PET_volumes != np.array(GNB_predictions).reshape(1,-1)[0])}")

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