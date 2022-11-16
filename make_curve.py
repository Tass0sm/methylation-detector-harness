import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection, metrics

fpr = dict()
tpr = dict()
roc_auc = dict()

###############################################################################
#                                     data                                    #
###############################################################################

LABELLED_DATA_LIST = {
    8078: {'positive': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/machine_learning/model_basedir/8079pos_newF1F2GL_fishers0.csv',
           'negative': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/machine_learning/model_basedir/8079neg_newF1F2GL_fishers0.csv'},
    8974: {'positive': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/machine_learning/model_basedir/8975pos_newF1F2GL_fishers0.csv',
           'negative': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/machine_learning/model_basedir/8975neg_newF1F2GL_fishers0.csv'},
    8988: {'positive': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/machine_learning/model_basedir/8989pos_newF1F2GL_fishers0.csv',
           'negative': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/machine_learning/model_basedir/8975neg_newF1F2GL_fishers0.csv'}
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
        index=['positive', 'read_id', 'site_0b'],
        columns='delta',
        values='pval'
    ).dropna()

    return pivoted_labelled_df

def get_split_labelled_data(site):
    pivoted_labelled_df = prepare_labelled_data(site)
    X = pivoted_labelled_df.values
    y = pivoted_labelled_df.index.get_level_values("positive").values

    # TODO: this is wrong, will fix later:
    sss = model_selection.StratifiedShuffleSplit(n_splits=2, test_size=0.5, random_state=0)
    g = sss.split(X, y)
    X_train_idx, y_train_idx = next(g)
    X_test_idx, y_test_idx = next(g)

    X_train = X[X_train_idx]
    y_train = y[y_train_idx].astype(np.float32)
    X_test = X[X_test_idx]
    y_test = y[y_test_idx].astype(np.float32)

    return X_train, y_train, X_test, y_test

###############################################################################
#                                training data                                #
###############################################################################

site = 8078
X_train, y_train, X_test, y_test = get_split_labelled_data(site)

#                                  kim model                                  #
import kim.interface
kim_decision_function = kim.interface.train(X_train, y_train)

#                                  tombo-msc                                  #
import tombo_msc.interface
tombo_msc_decision_function = tombo_msc.interface.train(X_train, y_train)

#                                    xpore                                    #
import xpore.interface
xpore_decision_function = xpore.interface.train(X_train, y_train)

#                                   nanom6a                                   #
import nanom6a.interface
nanom6a_decision_function = nanom6a.interface.train(X_train, y_train)

#                                    m6anet                                   #
import m6anet.interface
m6anet_decision_function = m6anet.interface.train(X_train, y_train)

#                                   nanoRMS                                   #
import nanoRMS.interface
nanoRMS_decision_function = nanoRMS.interface.train(X_train, y_train)

###############################################################################
#                               making all rocs                               #
###############################################################################

lw = 2

def plot_model_roc_for_site(ax, decision_function, site):
    plt.figure()

    y_hat = decision_function(X_test)

    fpr_arr, tpr_arr, thresholds = metrics.roc_curve(y_test, y_hat)
    roc_auc = metrics.auc(fpr_arr, tpr_arr)

    ax.plot(
        fpr_arr,
        tpr_arr,
        lw=lw,
        label=f"ROC curve (area = %0.2f)" % roc_auc,
    )

def plot_rocs_for_site(ax, decision_functions, site):
    for df in decision_functions:
        plot_model_roc_for_site(ax, df, site)

def make_plot():
    fig, ax = plt.subplots(1, 1)

    plot_rocs_for_site(ax, [kim_decision_function], site)

    ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")

    fig.savefig(f"roc_{site}.png")
