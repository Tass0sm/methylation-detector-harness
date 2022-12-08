import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

from sklearn import model_selection, metrics

###############################################################################
#                              plotting all rocs                              #
###############################################################################

lw = 2

def plot_model_roc_for_site(ax, decision_function, X_test, y_test):
    y_hat = decision_function(X_test)

    fpr_arr, tpr_arr, thresholds = metrics.roc_curve(y_test, y_hat)
    roc_auc = metrics.auc(fpr_arr, tpr_arr)

    ax.plot(
        fpr_arr,
        tpr_arr,
        lw=lw,
        label=f"ROC curve (area = %0.2f)" % roc_auc,
    )

def plot_roc_for_site_by_y_hat(ax, y_hat, y_test, label=None):
    fpr_arr, tpr_arr, thresholds = metrics.roc_curve(y_test, y_hat)
    roc_auc = metrics.auc(fpr_arr, tpr_arr)

    if label is not None:
        label = label + f" (AUC = %0.2f)" % roc_auc
    else:
        label = f"AUC = %0.2f" % roc_auc

    ax.plot(
        fpr_arr,
        tpr_arr,
        lw=lw,
        label=label,
    )

def plot_model_prc_for_site(ax, decision_function, X_test, y_test):
    y_hat = decision_function(X_test)

    precision_arr, recall_arr, thresholds = metrics.precision_recall_curve(y_test, y_hat)
    avg_prec = metrics.average_precision_score(y_test, y_hat)

    ax.plot(
        recall_arr,
        precision_arr,
        lw=lw,
        label=f"AP = %0.2f" % avg_prec,
    )

def plot_prc_for_site_by_y_hat(ax, y_hat, y_test, label=None):
    precision_arr, recall_arr, thresholds = metrics.precision_recall_curve(y_test, y_hat)
    avg_prec = metrics.average_precision_score(y_test, y_hat)

    if label is not None:
        label = label + f" (AP = %0.2f)" % avg_prec
    else:
        label = f"AP = %0.2f" % avg_prec

    ax.plot(
        recall_arr,
        precision_arr,
        lw=lw,
        label=label,
    )

def plot_rocs_for_site(ax, decision_functions, X_test, y_test):
    for df in decision_functions:
        plot_model_roc_for_site(ax, df, X_test, y_test)

    ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    # ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")

def plot_prcs_for_site(ax, decision_functions, X_test, y_test):
    for df in decision_functions:
        plot_model_prc_for_site(ax, df, X_test, y_test)

    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    # ax.set_title("Precision-Recall Curve")
