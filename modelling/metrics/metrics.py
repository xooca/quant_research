from collections import Counter
import sklearn.metrics as mt
import numpy as np


def calculate_confusion_matrix(y_true, y_pred):
    classes = list(set(y_true))
    n_classes = len(classes)
    
    tp = [0] * n_classes
    fn = [0] * n_classes
    fp = [0] * n_classes
    tn = [0] * n_classes
    
    for i, cls in enumerate(classes):
        for j in range(len(y_true)):
            if y_true[j] == cls and y_pred[j] == cls:
                tp[i] += 1
            elif y_true[j] == cls and y_pred[j] != cls:
                fn[i] += 1
            elif y_true[j] != cls and y_pred[j] == cls:
                fp[i] += 1
            else:
                tn[i] += 1
    
    return tp, fn, fp, tn

def calculate_confusion_values(y_true, y_pred):
    classes = list(set(y_true))
    n_classes = len(classes)
    
    tp = [0] * n_classes
    fn = [0] * n_classes
    fp = [0] * n_classes
    tn = [0] * n_classes
    
    for i, cls in enumerate(classes):
        for j in range(len(y_true)):
            if y_true[j] == cls and y_pred[j] == cls:
                tp[i] += 1
            elif y_true[j] == cls and y_pred[j] != cls:
                fn[i] += 1
            elif y_true[j] != cls and y_pred[j] == cls:
                fp[i] += 1
            else:
                tn[i] += 1
    
    return np.sum(tp), np.sum(fn), np.sum(fp), np.sum(tn)

def get_minority_class_data(y_true, y_pred):
    unique, counts = np.unique(y_true, return_counts=True)
    majority_class = unique[np.argmax(counts)]
    mask = y_true != majority_class
    if unique.size == 2:
        cat_type="binary"
    elif unique.size > 2:
        cat_type="multiclass"
    else:
        cat_type="unknown"
    y_true_minority = y_true[mask]
    y_pred_minority = y_pred[mask]
    return y_true_minority,y_pred_minority,cat_type
   
def calculate_f1score(y_true, y_pred,metric_type='minority'):
    if metric_type == 'minority':
        y_true,y_pred,cat_type = get_minority_class_data(y_true, y_pred)
    else:
        unique, counts = np.unique(y_true, return_counts=True)  
        if unique.size == 2:
            cat_type="binary"
        elif unique.size > 2:
            cat_type="multiclass"
        else:
            cat_type="unknown"
    if cat_type == 'multiclass':
        average="weighted"
    else:
        average="binary"
    ret_metric = mt.f1_score(y_true,y_pred,average=average)
    return ret_metric

def calculate_tp_fn_fp_tn(y_true, y_pred,metric_type='minority'):
    if metric_type == 'minority':
        y_true,y_pred,cat_type = get_minority_class_data(y_true, y_pred)
    tp, fn, fp, tn = calculate_confusion_values(y_true, y_pred)
    return tp, fn, fp, tn

def calculate_accuracy(y_true, y_pred,metric_type='minority'):
    if metric_type == 'minority':
        y_true,y_pred,cat_type = get_minority_class_data(y_true, y_pred)
    ret_metric = mt.accuracy_score(y_true,y_pred)
    return ret_metric

def calculate_precision(y_true, y_pred,metric_type='minority'):
    if metric_type == 'minority':
        y_true,y_pred,cat_type = get_minority_class_data(y_true, y_pred)
    ret_metric = mt.precision_score(y_true,y_pred)
    return ret_metric

def calculate_recall(y_true, y_pred,metric_type='minority'):
    if metric_type == 'minority':
        y_true,y_pred,cat_type = get_minority_class_data(y_true, y_pred)
    ret_metric = mt.recall_score(y_true,y_pred)
    return ret_metric

def calculate_metrices_all(y_true, y_pred,metric_type='minority'):
    if metric_type == 'minority':
        y_true,y_pred,cat_type = get_minority_class_data(y_true, y_pred)
    else:
        unique, counts = np.unique(y_true, return_counts=True)  
        if unique.size <= 2:
            cat_type="binary"
        elif unique.size > 2:
            cat_type="multiclass"
        else:
            cat_type="unknown"
    if cat_type == 'multiclass':
        average="weighted"
    else:
        average="binary"
    f1 = mt.f1_score(y_true,y_pred,average=average)
    accuracy = mt.accuracy_score(y_true,y_pred)
    precision = mt.precision_score(y_true,y_pred,average=average)
    recall = mt.recall_score(y_true,y_pred,average=average)
    return f1,accuracy,precision,recall

def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)

def calculate_profit_cost_for_minority_class(y_true, y_pred,tp_w=400,fn_w=200,fp_w=100,tn_w=0):
    tp, fn, fp, tn = calculate_tp_fn_fp_tn(y_true, y_pred,metric_type='minority')
    loss = tp_w*tp + tn_w*tn - fn_w*fn - fp_w*fp
    return loss

def calculate_profit_cost_for_all_class(y_true, y_pred,tp_w=400,fn_w=200,fp_w=100,tn_w=0):
    tp, fn, fp, tn = calculate_tp_fn_fp_tn(y_true, y_pred,metric_type='majority')
    loss = tp_w*tp + tn_w*tn - fn_w*fn - fp_w*fp
    return loss

def calculate_gini_for_minority_class(y_true, y_pred,normalized=True):
    y_true,y_pred,cat_type = get_minority_class_data(y_true, y_pred)
    if normalized:
        gini_val = gini_normalized(y_true, y_pred)
    else:
        gini_val = gini(y_true, y_pred)
    return gini_val

def calculate_gini_for_all_class(y_true, y_pred,normalized=True):
    if normalized:
        gini_val = gini_normalized(y_true, y_pred)
    else:
        gini_val = gini(y_true, y_pred)
    return gini_val