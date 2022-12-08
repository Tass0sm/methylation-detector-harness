import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

from sklearn import model_selection, metrics

from plotting import *

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
    retval.columns = retval.columns.astype(int)
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

import kim.interface

other_models = ["m6anet", "nanom6a"]
            
def main():
    train_cache = {}
    test_cache = {}

    for site in [8078, 8974, 8988]:
        Xy_df = get_randomized_data(site)
        X_train, y_train, X_test, y_test = test_train_split(Xy_df)
        train_cache[site] = (X_train.copy(), y_train.copy())
        test_cache[site] = (X_test.copy(), y_test.copy())
        print(f"saved data for {site}")

    fig, axes = plt.subplots(3, 3, figsize=(13, 13))

    for i, model_site in enumerate([8078, 8974, 8988]):
        X_train, y_train = train_cache[model_site]
        df = kim.interface.train(X_train, y_train)
        dfs = [df]
        for j, test_site in enumerate([8078, 8974, 8988]):
            X_test, y_test = test_cache[test_site]
            plot_rocs_for_site(axes[i, j], dfs, X_test, y_test)

            for model in other_models:
                with open(f"./{model}/{test_site}-predictions.pickle", "rb") as pred_f, open(f"./{model}/{test_site}-test.pickle", "rb") as test_f:
                    model_y_pred_at_current_test_site = pickle.load(pred_f)
                    model_y_test_at_current_test_site = pickle.load(test_f)

                    if model_y_pred_at_current_test_site.dtype == np.object:
                        model_y_pred_at_current_test_site = model_y_pred_at_current_test_site.astype(np.float64)
                        model_y_test_at_current_test_site = model_y_test_at_current_test_site.astype(np.float64)

                    model_y_pred_at_current_test_site = np.nan_to_num(model_y_pred_at_current_test_site)
                    model_y_test_at_current_test_site = np.nan_to_num(model_y_test_at_current_test_site)

                    plot_roc_for_site_by_y_hat(axes[i, j],
                                               model_y_pred_at_current_test_site,
                                               model_y_test_at_current_test_site,
                                               label=model)

            axes[i, j].legend(loc="lower right")

    pad = 5 # in points

    cols = ['At Site {}'.format(site) for site in [8078, 8974, 8988]]
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    rows = ['With model{}'.format(site) for site in [8078, 8974, 8988]]
    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    fig.suptitle("Receiver Operating Characteristic Curves")
    fig.savefig(f"all_rocs.png")

    fig, axes = plt.subplots(3, 3, figsize=(13, 13))

    for i, model_site in enumerate([8078, 8974, 8988]):
        X_train, y_train = train_cache[model_site]
        df = kim.interface.train(X_train, y_train)
        dfs = [df]
        for j, test_site in enumerate([8078, 8974, 8988]):
            X_test, y_test = test_cache[test_site]
            plot_prcs_for_site(axes[i, j], dfs, X_test, y_test)

            for model in other_models:
                with open(f"./{model}/{test_site}-predictions.pickle", "rb") as pred_f, open(f"./{model}/{test_site}-test.pickle", "rb") as test_f:
                    model_y_pred_at_current_test_site = pickle.load(pred_f)
                    model_y_test_at_current_test_site = pickle.load(test_f)

                    if model_y_pred_at_current_test_site.dtype == np.object:
                        model_y_pred_at_current_test_site = model_y_pred_at_current_test_site.astype(np.float64)
                        model_y_test_at_current_test_site = model_y_test_at_current_test_site.astype(np.float64)

                    model_y_pred_at_current_test_site = np.nan_to_num(model_y_pred_at_current_test_site)
                    model_y_test_at_current_test_site = np.nan_to_num(model_y_test_at_current_test_site)

                    plot_prc_for_site_by_y_hat(axes[i, j],
                                               model_y_pred_at_current_test_site,
                                               model_y_test_at_current_test_site,
                                               label=model)

            axes[i, j].legend(loc="lower left")

    pad = 5 # in points

    cols = ['At Site {}'.format(site) for site in [8078, 8974, 8988]]
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    rows = ['With model{}'.format(site) for site in [8078, 8974, 8988]]
    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    fig.suptitle("Precision Recall Curves")
    fig.savefig(f"all_prcs.png")


main()
