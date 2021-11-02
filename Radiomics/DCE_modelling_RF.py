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
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

LR = LogisticRegression(
    penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=10000
)
GNB = GaussianNB()

DCE_volumes = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0])
DCE_relevant_df = pd.read_csv("./DCE_relevant_df.csv", index_col=0)

LRaccuracies = []
GNBaccuracies = []
LR_AUCs = []
GNB_AUCs = []

y_pred_LRs = []
y_pred_GNBs = []

LR_y_pred_probas = np.empty((16, 2), float)
GNB_y_pred_probas = np.empty((16, 2), float)
LR_fprs = []
LR_tprs = []
GNB_fprs = []
GNB_tprs = []

LR_sensitivities = []
LR_specificities = []
GNB_sensitivities = []
GNB_specificities = []

# let's perform leave-one-out analysis
for i in range(16):
    test_index = i
    train_index = list(range(16))
    train_index.pop(i)

    # print(train_index, test_index)

    DCE_train, DCE_test = (
        DCE_relevant_df.iloc[train_index].values,
        DCE_relevant_df.iloc[test_index].values,
    )
    # we need to reshape the test input
    DCE_test = [DCE_test]

    DCE_vol_train, DCE_vol_test = DCE_volumes[train_index], DCE_volumes[test_index]

    ss = StandardScaler()
    ss.fit(DCE_train)
    X_train = ss.transform(DCE_train)
    X_test = ss.transform(DCE_test)

    y_train = DCE_volumes[train_index]
    y_test = DCE_volumes[test_index]
    # print("y train:", y_train)

    LR.fit(DCE_train, DCE_vol_train)
    GNB.fit(DCE_train, DCE_vol_train)

    y_pred_LR = LR.predict(DCE_test)
    y_pred_LRs.append(y_pred_LR)

    y_pred_GNB = GNB.predict(DCE_test)
    y_pred_GNBs.append(y_pred_GNB)

    # probas
    LR_y_pred_proba = LR.predict_proba(DCE_test)
    GNB_y_pred_proba = GNB.predict_proba(DCE_test)

    LR_y_pred_probas[i, :] = LR_y_pred_proba
    GNB_y_pred_probas[i, :] = GNB_y_pred_proba

"""
Compute and plot an ROC curve
"""

LR_fpr, LR_tpr, _ = roc_curve(DCE_volumes, LR_y_pred_probas[:, 1])

fig, ax = plt.subplots(1, 1)
ax.plot(LR_fpr, LR_tpr)
ax.plot([0, 1], [0, 1], c="r", lw=2, ls="--")
plt.title("ROC curve for Logistic Regression in DCE-MRI")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
print("AUC for LR in DCE:", metrics.auc(LR_fpr, LR_tpr))

GNB_fpr, GNB_tpr, _ = roc_curve(DCE_volumes, GNB_y_pred_probas[:, 1])

fig, ax = plt.subplots(1, 1)
ax.plot(GNB_fpr, GNB_tpr)
ax.plot([0, 1], [0, 1], c="r", lw=2, ls="--")
plt.title("ROC curve for Gaussian Naive Bayes in DCE-MRI")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
print("AUC for GNB in DCE:", metrics.auc(GNB_fpr, GNB_tpr))
GNB_AUCs.append(metrics.auc(GNB_fpr, GNB_tpr))

"""
other metrics
"""

print(
    f"Number of LR mislabeled points out of a total %d points : %d"
    % (DCE_volumes.shape[0], np.sum(DCE_volumes != y_pred_LRs))
)
print(
    f"Number of GNB mislabeled points out of a total %d points : %d"
    % (DCE_volumes.shape[0], np.sum(DCE_volumes != y_pred_GNBs))
)

cm1_LR = confusion_matrix(DCE_volumes, y_pred_LRs)
print("Confusion Matrix LR: \n", cm1_LR)

total1 = sum(sum(cm1_LR))
##from confusion matrix calculate accuracy
accuracy1 = (cm1_LR[0, 0] + cm1_LR[1, 1]) / total1
print("Accuracy LR : ", accuracy1)

cm1_GNB = confusion_matrix(DCE_volumes, y_pred_GNBs)
print("Confusion Matrix GNB: \n", cm1_GNB)

total1 = sum(sum(cm1_GNB))
##from confusion matrix calculate accuracy
accuracy1 = (cm1_GNB[0, 0] + cm1_GNB[1, 1]) / total1
print("Accuracy GNB : ", accuracy1)
