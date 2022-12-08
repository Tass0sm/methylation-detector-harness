import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

from sklearn import model_selection, metrics

###############################################################################
#                                     data                                    #
###############################################################################

LABELLED_DATA_LIST = {
    8078: {'positive': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/machine_learning/model_basedir/8079pos_newF1F2GL_fishers0.csv',
           'negative': '/fs/project/PAS1405/General/tassosm/ROC_PR_curve/ctrl9kb_newF1F2GL_msc0.csv'},
    8974: {'positive': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/machine_learning/model_basedir/8975pos_newF1F2GL_fishers0.csv',
           'negative': '/fs/project/PAS1405/General/tassosm/ROC_PR_curve/ctrl9kb_newF1F2GL_msc0.csv'},
    8988: {'positive': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/machine_learning/model_basedir/8989pos_newF1F2GL_fishers0.csv',
           'negative': '/fs/project/PAS1405/General/tassosm/ROC_PR_curve/ctrl9kb_newF1F2GL_msc0.csv'},
}

RANGE_OF_BASES_TO_INCLUDE = (-4, 1) # inclusive

def load_csv(filepath):
    '''Load per-read stats from a CSV file into a Pandas DataFrame'''
    retval = pd.read_csv(filepath, header=0, index_col=0).rename_axis('pos_0b', axis=1)
    retval.columns = retval.columns.astypes(int)
    retval.index = [x[2:-1] if x[0:2] == "b'" and x[-1] == "'" else x for x in retval.index]
    retval = retval.rename_axis('read_id')
    return retval

def longify(df):
    '''Convert dataframe output of load_csv to a long format'''
    return df.stack().rename('pval').reset_index()

def prepare_labelled_data(site):
    """The Kim Model makes a prediction about a modified site based on the Tombo
    MSC values surrounding that site in the read."""

    data_file_pair = LABELLED_DATA_LIST[site]
    to_concat = []
    for filepath, positive in [(data_file_pair['positive'], True),
                               (data_file_pair['negative'], False)]:
        # read data and create useful columns
        df = ( # pylint: disable=invalid-name
            longify(load_csv(filepath))
            .assign(
                positive = positive,
                site_0b = site
            ).assign(
                delta = lambda x: x['pos_0b'] - x['site_0b']
            )
        )
        # remove unnecessary positions
        df = df.loc[
            (RANGE_OF_BASES_TO_INCLUDE[0] <= df['delta'])
            & (df['delta'] <= RANGE_OF_BASES_TO_INCLUDE[1])
        ]
        to_concat.append(df)
        del df

    labelled_df = pd.concat(to_concat)
    pivoted_labelled_df = labelled_df.pivot(
        index=['positive', 'read_id'],
        columns='delta',
        values='pval'
    ).dropna()

    return pivoted_labelled_df

def get_randomized_data(site):
    pivoted_labelled_df = prepare_labelled_data(site)
    Xy_df = pivoted_labelled_df.reset_index(level=0).astype("float64")
    Xy_df = Xy_df[[*Xy_df.columns[1:], Xy_df.columns[0]]]

    rs = RandomState(MT19937(SeedSequence(123456789)))
    Xy_df = Xy_df.sample(frac = 1, random_state=rs)

    return Xy_df

def test_train_split(Xy_df):
    half_point = Xy_df.shape[0] // 2
    col_point = Xy_df.shape[1] - 1

    Xy_train = Xy_df.iloc[0:half_point, :].values
    X_train = Xy_train[:, 0:col_point]
    y_train = Xy_train[:, col_point]

    Xy_test = Xy_df.iloc[0:half_point, :].values
    X_test = Xy_test[:, 0:col_point]
    y_test = Xy_test[:, col_point]

    return X_train, y_train, X_test, y_test

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
    ax.legend(loc="lower right")

###############################################################################
#                                   testing                                   #
###############################################################################

# import kim.interface

# model_site = 8974
# test_site = 8974

# X_train, y_train, nop, nop = get_split_labelled_data(model_site)
# nop, nop, X_test, y_test = get_split_labelled_data(test_site)

# df = kim.interface.train(X_train, y_train)
# dfs = [df]

# fig, ax = plt.subplots(1, 1)

# plot_prcs_for_site(ax, dfs, X_test, y_test)

# fig.savefig("test1.png")

###############################################################################
#                                  main code                                  #
###############################################################################

roc_fig, roc_axes = plt.subplots(1, 3, figsize=(15, 5))
prc_fig, prc_axes = plt.subplots(1, 3, figsize=(15, 5))

for fig_col, site in enumerate([8078, 8974, 8988]):
    roc_ax = roc_axes[fig_col]
    prc_ax = prc_axes[fig_col]
            
    for model in ["kim", "m6anet", "nanom6a"]:
        with open(f"./{model}/{site}-predictions.pickle", "rb") as pred_f, open(f"./{model}/{site}-test.pickle", "rb") as test_f:
            y_pred = pickle.load(pred_f)
            y_test = pickle.load(test_f)

        if y_pred.dtype == np.object:
            y_pred = y_pred.astype(np.float64)
            y_test = y_test.astype(np.float64)
            
        y_pred = np.nan_to_num(y_pred)
        y_test = np.nan_to_num(y_test)

        if model == "kim":
            model = f"model{site}"

        plot_roc_for_site_by_y_hat(roc_ax, y_pred, y_test, label=model)
        plot_prc_for_site_by_y_hat(prc_ax, y_pred, y_test, label=model)
        print(f"finished with {model}")

    roc_ax.legend(loc='lower right')
    prc_ax.legend(loc='lower left')

        
roc_fig.suptitle("Receiver Operating Characteristic Curves")
roc_fig.savefig(f"multi_model_rocs.png")

prc_fig.suptitle("Precision-Recall Curves")
prc_fig.savefig(f"multi_model_prcs.png")
