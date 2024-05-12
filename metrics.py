import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(matrix):
    """
    Plots the confusion matrix.

    Args:
        matrix (array_like): The confusion matrix to be plotted.
    """
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
    """
    Calculates the True Positive Rate (TPR) and False Positive Rate (FPR) for various threshold values.

    Args:
        proba (array_like): Predicted probabilities for positive class.
        y_trues (array_like): True labels.

    Returns:
        list: Lists of True Positive Rates (TPRs) and False Positive Rates (FPRs) for different threshold values.
    """
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

def plot_roc_curve(list_TPR, list_FPR, auc):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        list_TPR (array_like): List of True Positive Rates.
        list_FPR (array_like): List of False Positive Rates.
        auc (float): Area Under the Curve (AUC) value.
    """
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
    """
    Plots the Precision-Recall Curve (PRC).

    Args:
        list_recall (array_like): List of recall values.
        list_precision (array_like): List of precision values.
    """
    plt.figure()
    plt.plot(list_recall, list_precision, linestyle="-")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("PRC Curve")
    plt.show()

def compare_result_final(predics, real_values):
    """
    Compares predicted values with actual values and computes TP, FP, TN, FN.

    Args:
        predics (array_like): Predicted values.
        real_values (array_like): Actual values.

    Returns:
        tuple: TP, FP, TN, FN.
    """
    TP = sum(1 for p, t in zip(predics, real_values) if p == 1 and t == 1)
    FP = sum(1 for p, t in zip(predics, real_values) if p == 1 and t == 0)
    TN = sum(1 for p, t in zip(predics, real_values) if p == 0 and t == 0)
    FN = sum(1 for p, t in zip(predics, real_values) if p == 0 and t == 1)
    return TP, FP, TN, FN

def auc_roc(lista_A, list_B, sorted_true):
    """
    Calculates the Area Under the Curve (AUC) for the ROC curve.

    Args:
        lista_A (array_like): List of values for the first axis (e.g., False Positive Rate).
        list_B (array_like): List of values for the second axis (e.g., True Positive Rate).
        sorted_true (bool): Indicates whether the points are sorted.

    Returns:
        float: The AUC value.
    """
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
    """
    Computes the accuracy.

    Args:
        cant_successes (int): Number of successful predictions.
        cant_predict (int): Total number of predictions.

    Returns:
        float: Accuracy.
    """
    return cant_successes / cant_predict

def presicion(TP, FP):
    """
    Computes the precision.

    Args:
        TP (int): True positives.
        FP (int): False positives.

    Returns:
        float: Precision.
    """
    sum_ = TP + FP
    if sum_ != 0:
        return TP / sum_
    else:
        return 0

def recall(TP, FN):
    """
    Computes the recall.

    Args:
        TP (int): True positives.
        FN (int): False negatives.

    Returns:
        float: Recall.
    """
    return TP / (TP + FN)

def calc_TPR_FPR(TP, FP, TN, FN):
    """
    Calculates True Positive Rate (TPR) and False Positive Rate (FPR).

    Args:
        TP (int): True positives.
        FP (int): False positives.
        TN (int): True negatives.
        FN (int): False negatives.

    Returns:
        tuple: TPR and F
    """
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return TPR, FPR  