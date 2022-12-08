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
           'negative': '/fs/project/PAS1405/General/tassosm/ROC_PR_curve/ctrl9kb_newF1F2GL_msc0.csv',
           'read_dirs': ["/fs/project/PAS1405/General/tassosm/ROC_PR_curve/8079pos_single_fast5",
                         "/fs/project/PAS1405/General/HIV_RNA_modification_dataset/ctrl9kb/ctrl9kb_single_fast5"]},
    8974: {'positive': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/machine_learning/model_basedir/8975pos_newF1F2GL_fishers0.csv',
           'negative': '/fs/project/PAS1405/General/tassosm/ROC_PR_curve/ctrl9kb_newF1F2GL_msc0.csv',
           'read_dirs': ["/fs/project/PAS1405/General/tassosm/ROC_PR_curve/8975pos_single_fast5",
                         "/fs/project/PAS1405/General/HIV_RNA_modification_dataset/ctrl9kb/ctrl9kb_single_fast5"]},
    8988: {'positive': '/fs/project/PAS1405/GabbyLee/project/m6A_modif/machine_learning/model_basedir/8989pos_newF1F2GL_fishers0.csv',
           'negative': '/fs/project/PAS1405/General/tassosm/ROC_PR_curve/ctrl9kb_newF1F2GL_msc0.csv',
           'read_dirs': ["/fs/project/PAS1405/General/tassosm/ROC_PR_curve/8989pos_single_fast5",
                         "/fs/project/PAS1405/General/HIV_RNA_modification_dataset/ctrl9kb/ctrl9kb_single_fast5"]},
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

    Xy_train = Xy_df.iloc[0:half_point, :]
    X_train = Xy_train.iloc[:, 0:col_point]
    y_train = Xy_train.iloc[:, col_point]

    Xy_test = Xy_df.iloc[0:half_point, :]
    X_test = Xy_test.iloc[:, 0:col_point]
    y_test = Xy_test.iloc[:, col_point]

    return X_train, y_train, X_test, y_test

###############################################################################
#                                     main                                    #
###############################################################################


# Get data
for site in [8078, 8974, 8988]:
    read_dirs = LABELLED_DATA_LIST[site]["read_dirs"]
    Xy_df = get_randomized_data(site)
    X_train, y_train, X_test, y_test = test_train_split(Xy_df)

    # prepare(*read_dirs, X_test, f"{site}")

    # import kim.interface
    # decision_function = kim.interface.train(X_train, y_train)
    # y_pred = decision_function(X_test.values)

    # with open(f"./kim/{site}-predictions.pickle", "wb") as pred_f, open(f"./kim/{site}-test.pickle", "wb") as test_f:
    #     pickle.dump(y_pred, pred_f)
    #     pickle.dump(y_test.values, test_f)

    # import m6anet.interface
    # prob_read_name_df = m6anet.interface.get_y_pred(f"/users/PAS1405/tassosm/Work/m6anet-pipeline/results/{site}/data.indiv_proba.csv.gz",
    #                                                 f"/users/PAS1405/tassosm/Work/m6anet-pipeline/results/{site}/alignment-summary.txt",
    #                                                 site, y_test)
    # y_pred_and_test = prob_read_name_df.join(y_test)

    # y_pred = y_pred_and_test["probability_modified"].values
    # y_test_values = y_pred_and_test["positive"].values

    # with open(f"./m6anet/{site}-predictions.pickle", "wb") as pred_f, open(f"./m6anet/{site}-test.pickle", "wb") as test_f:
    #     pickle.dump(y_pred, pred_f)
    #     pickle.dump(y_test.values, test_f)

    import nanom6a.interface
    read_name_prob_df = nanom6a.interface.get_y_pred(f"/users/PAS1405/tassosm/Work/nanom6A_pipeline/{site}_results/prediction_results/extract.reference.bed12",
                                                     f"/users/PAS1405/tassosm/Work/nanom6A_pipeline/{site}_results/prediction_results/sam_parse2.txt",
                                                     f"/users/PAS1405/tassosm/Work/nanom6A_pipeline/{site}_results/prediction_results/total_mod.tsv",
                                                    site)

    y_pred_and_test = read_name_prob_df.join(y_test)
    y_pred = y_pred_and_test["probability"].values
    y_test_vals = y_pred_and_test["positive"].values

    with open(f"./nanom6a/{site}-predictions.pickle", "wb") as pred_f, open(f"./nanom6a/{site}-test.pickle", "wb") as test_f:
        pickle.dump(y_pred, pred_f)
        pickle.dump(y_test_vals, test_f)
