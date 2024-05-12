import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def plot_confusion_matrix(matrix):
    plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    classes = ["Negative", "Positive"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, format(matrix[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Prediction")
    plt.show()

def curve_ROC(proba, y_trues):
    list_umbrals = np.linspace(0,1,100)
    list_TPR = []
    list_FPR = []
    for umbral in list_umbrals:
        y_pred_binary = [1 if pred >= umbral else 0 for pred in proba]
        TP, FP, TN, FN = compare_result_final(y_pred_binary, y_trues)
        TPR, FPR = calc_TPR_FPR(TP, FP, TN, FN)
        list_TPR.append(TPR)
        list_FPR.append(FPR)

    return [list_TPR, list_FPR]

def reed_data(name_file):
    data = pd.read_csv(name_file)
    V1 = data["V1"]
    V2 = data["V2"]
    V3 = data["V3"]
    V4 = data["V4"]
    V5 = data["V5"]
    V6 = data["V6"]
    V7 = data["V7"]
    V8 = data["V8"]
    V9 = data["V9"]
    V10 = data["V10"]
    V11 = data["V11"]
    V12 = data["V12"]
    V13 = data["V13"]
    V14 = data["V14"]
    V15 = data["V15"]
    V16 = data["V16"]
    V17 = data["V17"]
    V18 = data["V18"]
    V19 = data["V19"]
    V20 = data["V20"]
    V21 = data["V21"]
    V22 = data["V22"]
    V23 = data["V23"]
    V24 = data["V24"]
    V25 = data["V25"]
    V26 = data["V26"]
    V27 = data["V27"]
    V28 = data["V28"]
    Class = data["Class"]
    Log_Amount = data["Log Amount"]
    size = len(Log_Amount)
    return np.array([V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Log_Amount, Class]).T, size

def plot_roc_curve(list_TPR, list_FPR, auc):
    plt.figure()
    plt.plot(list_FPR, list_TPR, color="darkorange", lw=2)
    plt.plot([0, 1], [0, 1], color="navy", label=f"AUC = {auc}", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.show()

def plot_PRC_curve(list_recall, list_precision):
    plt.figure()
    plt.plot(list_recall, list_precision, linestyle="-")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("PRC Curve")
    plt.show()

def max_min_data(data, initial_type, final_type):
    list_max = []
    list_min = []
    for index in range(initial_type, final_type):
        list_max.append(np.max(data[index]))
        list_min.append(np.min(data[index]))
    return list_max, list_min

def norm_data(X, index,  list_max, list_min, initial_type, final_type):
    for index in range(initial_type, final_type):
        X[index] = (X[index] - list_min[index]) / (list_max[index] - list_min[index])
    return X

def norm_dataset(dataset, size, initial_type, final_type):
    list_max, list_min = max_min_data(dataset.T, initial_type, final_type)

    for index in range(size):
        dataset[index] = norm_data(dataset[index], index, list_max, list_min, initial_type, final_type)

    return dataset

def div_x_y(list_data):
    list_x = []
    list_y = []
    for data in list_data:
        list_x.append(data[0:29])
        list_y.append(data[29])
    return np.array(list_x), np.array(list_y)

def build_set_data(data, indexs_train, indexs_val):
    set_train = [data[index] for index in indexs_train]
    set_val = [data[index] for index in indexs_val]

    return set_train, set_val 

def func_sigmoid(data):
    return 1/(1+np.exp(-data))

def derivate_func_sigmoid(data):
    sig = lambda data: 1/(1+np.exp(-data))
    return sig(data)*(1-sig(data))

def compare_result_final(predics, real_values):
    TP = sum(1 for p, t in zip(predics, real_values) if p == 1 and t == 1)
    FP = sum(1 for p, t in zip(predics, real_values) if p == 1 and t == 0)
    TN = sum(1 for p, t in zip(predics, real_values) if p == 0 and t == 0)
    FN = sum(1 for p, t in zip(predics, real_values) if p == 0 and t == 1)
    return TP, FP, TN, FN

def auc_roc(lista_A, list_B, sorted_true):
    if sorted_true == True:
        points = sorted(zip(list_B, lista_A), key=lambda x: x[0])
    else:
        points = zip(list_B, lista_A)

    auc = 0
    prev_fpr, prev_tpr = 0, 0

    for value_A, value_B in points:

        auc += (value_A - prev_fpr) * (value_B + prev_tpr) / 2
        prev_fpr, prev_tpr = value_A, value_B

    return auc

def accuracy(cant_successes, cant_predict):
    return cant_successes / cant_predict

def presicion(TP, FP):
    sum_ = TP + FP
    if sum_ != 0:
        return TP / sum_
    else:
        return 0

def recall(TP, FN):
    return TP / (TP + FN)

def calc_TPR_FPR(TP, FP, TN, FN):
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return TPR, FPR  