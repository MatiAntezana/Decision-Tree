import numpy as np
import pandas as pd

def div_x_y(list_data):
    """
    Separates feature vectors and labels.

    Args:
        list_data (array_like): List of data points.

    Returns:
        tuple: Tuple containing feature vectors and labels.
    """
    list_x = []
    list_y = []
    for data in list_data:
        list_x.append(data[0:29])
        list_y.append(data[29])
    return np.array(list_x), np.array(list_y)

def build_set_data(data, indexs_train, indexs_val):
    """
    Builds training and validation sets from the dataset.

    Args:
        data (array_like): The dataset.
        indexs_train (array_like): Indices of the training set.
        indexs_val (array_like): Indices of the validation set.

    Returns:
        tuple: Training and validation sets.
    """
    set_train = [data[index] for index in indexs_train]
    set_val = [data[index] for index in indexs_val]

    return set_train, set_val 

def func_sigmoid(data):
    """
    Computes the sigmoid function.

    Args:
        data (array_like): Input data.

    Returns:
        array_like: Result of the sigmoid function.
    """
    return 1/(1+np.exp(-data))

def derivate_func_sigmoid(data):
    """
    Computes the derivative of the sigmoid function.

    Args:
        data (array_like): Input data.

    Returns:
        array_like: Result of the derivative of the sigmoid function.
    """
    sig = lambda data: 1/(1+np.exp(-data))
    return sig(data)*(1-sig(data))

def reed_data(name_file):
    """
    Reads data from a CSV file.

    Args:
        name_file (str): Name of the CSV file.

    Returns:
        tuple: Tuple containing the data and the size of the data.
    """
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
    return np.array([V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, 
                     V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, 
                     Log_Amount, Class]).T, size