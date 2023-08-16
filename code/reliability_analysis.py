"""
reliability_analysis.py

Code to train/test models
and plot results for reliability
analyses.
"""

# Imports
from sklearn.ensemble import RandomForestClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, \
    f1_score, recall_score, precision_score, confusion_matrix
import statsmodels.api as sm
import itertools
import pingouin as pg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
from scipy.stats import ttest_rel, ttest_ind, mannwhitneyu, bootstrap
import reliability_data_prep
from sklearn.calibration import calibration_curve, CalibratedClassifierCV


# Matplotlib style
plt.style.use(['seaborn-v0_8-white'])


def percentile(n):
    """
    Get percentiles. 
    Creates percentiles function

    :param n: <int>, the percentile

    :return percentile_(x): the percentile function
    """
    def percentile_(x):
        """
        Get a percentile of x

        :param x: <array>, to get percentile
        :return percentile(x, n): the percentile at n
        """
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


def specificity_score(y_true, y_pred):
    """
    Specificity score

    :param y_true: np.array, the true values
    :param y_pred: np.array, the predicted values
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def pos_background_auroc(group, bgrd):
    """
    Positive background AUROC
    Also called BNSP

    :param group: np.array, the group values
    :param bgrd: np.array, \
        the values for the remainder of the population

    :return: <float>, the BNSP AUC
    """
    # Get positive example for group
    group_pos = group.loc[group.y_binary == 1, :]
    bgrd_neg = bgrd.loc[bgrd.y_binary == 0, :]
    concat = pd.concat([
        group_pos[['prob', 'y_binary']], 
        bgrd_neg[['prob', 'y_binary']]
    ]).reset_index(drop=True)
    return roc_auc_score(concat['y_binary'], concat['prob'])


def neg_background_auroc(group, bgrd):
    """
    Negative background AUROC
    Also called BPSN

    :param group: np.array, the group values
    :param bgrd: np.array, \
        the values for the remainder of the population

    :return: <float>, the BPSN AUC
    """
    # Get positive example for group
    group_neg = group.loc[group.y_binary == 0, :]
    bgrd_pos = bgrd.loc[bgrd.y_binary == 1, :]
    concat = pd.concat([
        group_neg[['prob', 'y_binary']], 
        bgrd_pos[['prob', 'y_binary']]
    ]).reset_index(drop=True)
    return roc_auc_score(concat['y_binary'], concat['prob'])


def get_model(model_type, params):
    """
    Get model to train.

    :param model_type: <str>, the type of model
    :param params: <dict>, the model parameters

    :return: sklearn ML model
    """
    if model_type == 'rf':
        return RandomForestClassifier(**params)
    elif model_type == 'gb':
        return GradientBoostingClassifier(**params)
    elif model_type == 'sv':
        return SVC(**params)
    elif model_type == 'lr':
        return LogisticRegression(**params)


def get_param_combinations(param_dict):
    """
    Get parameter combinations from a dictionary

    :param param_dict: dict<s:list>, the dictionary where parameters 
                                     are each specified in lists
    :return list<dict>, a list of all the parameter combination dicts
    """
    # If model does not have parameters
    if param_dict is None:
        return [None]

    # Get combinations
    param_combinations = list(itertools.product(*param_dict.values()))
    dict_keys = list(param_dict.keys())

    # Iterate through combinations
    return [dict(zip(dict_keys, v)) for v in param_combinations]


def ml_pipeline(
    model_type, params,
    train, test, features
):
    """
    Build and train an ML model

    :param model_type: <str>, the type of model
    :param params: <dict>, the params
    :param train: pd.DataFrame, training data
    :param test: pd.DataFrame, testing data
    :param features: list<str>, the feature names

    :return train_prob: np.array, risk prob on train data
    :return test_prob: np.array, risk prob on test data
    :return train_pred: np.array, predictions on train data
    :return test_pred: np.array, predictions on test data
    :return m: model, the trained model
    :return train: pd.DataFrame, imputed/normed training data
    :return test: pd.DataFrame, imputed/normed testing data
    """

    # Reset index
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # Train and test model
    m = get_model(
        model_type=model_type,
        params=params, 
    )
    # Calibrate if not LR
    if model_type != 'lr':
        m = CalibratedClassifierCV(
            m, method='sigmoid', cv=5
        )

    # Check if two values
    if train.y_binary.nunique() == 2:
        # Fit
        m.fit(train[features], train['y_binary'])

        # Predict
        train_prob = m.predict_proba(train[features])[:, 1]
        test_prob = m.predict_proba(test[features])[:, 1]
        train_pred = m.predict(train[features])
        test_pred = m.predict(test[features])
    else:
        y = train.y_binary.unique()[0]
        train_prob = [y] * train.shape[0]
        test_prob = [y] * test.shape[0]
        train_pred = [y] * train.shape[0]
        test_pred = [y] * test.shape[0]

    return train_prob, test_prob, train_pred, test_pred, \
        m, train, test


def run_cv(args):
    """
    Run a CV pipeline on the output index
    args are a tuple with the following values

    :param run: <str>, the model run
    :param dataset: <str>, the dataset name
    :param cv_type: <str>, the cv procedure
    :param c: <str>, the column group name (e.g., race)
    :param m: <str>, the model type
    :param p: dict, the model parameters
    :param train_df: pd.DataFrame, training data
    :param test_df: pd.DataFrame, validation data
    :param features: features<str>, the features used in analysis
    :param group_cols: list<str> the column names with subgroups

    :return res_df: <pd.DataFrame>, the predictions
    """
    # Get args
    run, dataset, cv_type, c, m, p, train_dfs, test_dfs, \
        features, group_cols = args

    # Iterate
    pred_dfs = []

    # Run CV
    for i in range(len(train_dfs)):
        # Get dfs
        train_df = train_dfs[i]
        test_df = test_dfs[i]

        # Run model and get results
        _, test_prob, _, test_pred, _, _, _ = ml_pipeline(
            model_type=m, params=p,
            train=train_df.copy(), test=test_df.copy(), 
            features=features
        )

        # Get results
        test_df.reset_index(drop=True, inplace=True)
        test_df['prob'] = test_prob
        test_df['pred'] = test_pred

        pred_dfs.append(test_df)

    # Concatenate
    pred_df = pd.concat(pred_dfs).reset_index(drop=True)

    # Add parameters
    pred_df['training_column'] = c
    pred_df['model_type'] = m
    pred_df['dataset'] = dataset
    pred_df['params'] = str(p)
    pred_df['cv_type'] = cv_type
    pred_df['run'] = run

    # Calculate results
    res_df = calc_sim_results(
        pred_df=pred_df, 
        dataset=dataset, 
        cv_type=cv_type,
        column=c,
        model_type=m, 
        params=p,
        run=run,
        group_cols=group_cols,
    )

    print(run,
        res_df.loc[res_df.column == 'Entire study', 
            'auroc'].round(2).iloc[0],
        res_df.loc[res_df.column == 'Entire study', 
            'sensitivity_mitchell'].round(3).iloc[0],
        res_df.loc[res_df.column == 'Entire study', 
            'specificity_mitchell'].round(3).iloc[0],
        dataset, cv_type, c, train_df.shape[0], test_df.shape[0],
        m, p
    )

    # Return
    return res_df, pred_df[[
        'study_id',
        'training_column', 'dataset',
        'outcome_index',
        'cv_type', 'fold',
        'model_type', 'params', 'run',
        'pred', 'prob', 'y_binary'
    ] + group_cols]


def calc_sim_results(
    pred_df, dataset, cv_type, column, 
    model_type, params,
    run, group_cols
):
    """
    Calculate results for a specific simulation setup

    :param pred_df: pd.DataFrame, prediction dataframe
    :param dataset: <str>, the dataset
    :param cv_type: <str>, the type of cv
    :param column: <str>, the column trained over
    :params model_type: <str>, the model type
    :params params: <str>, the model parameters in string form
    :params run: <float>, the run number
    :param group_cols: pd.DataFrame, columns to get results over
    
    :return res_df: <pd.DataFrame>, the df with added result values
    """
    # Sensitivity/specificity from Mitchell et al. paper
    specificity_threshold = 0.813

    res_dict = {
        'dataset': [],
        'cv_type': [],
        'training_column': [],
        'column': [],
        'value': [],
        'model_type': [],
        'params': [],
        'run': [],
        'num_samples': [],
        'num_ids': [],
        'pct_pos': [],
        'auroc': [],
        'pos_auroc': [],
        'neg_auroc': [],
        'auprc': [],
        'sensitivity': [],
        'specificity': [],
        'precision': [],
        'f1': [],
        'fpr': [],
        'fnr': [],
        'fdr': [],
        'sensitivity_mitchell': [],
        'specificity_mitchell': []
    }

    # Iterate over the group_cols
    for c in group_cols:

        # Get values
        for v in pred_df[c].unique():
        
            # Filter
            temp = pred_df.loc[(pred_df[c] == v), :]
            bgrd = pred_df.loc[(pred_df[c] != v), :]


            # Only record if 2 classes
            if (temp['y_binary'].nunique() < 2):
                continue

            # Get metrics
            auroc = roc_auc_score(temp['y_binary'], temp['prob'])
            if (c == 'Entire study') or (bgrd.y_binary.nunique() < 2):
                pos_auroc = None
                neg_auroc = None
            else:
                pos_auroc = pos_background_auroc(group=temp.copy(), bgrd=bgrd.copy())
                neg_auroc = neg_background_auroc(group=temp.copy(), bgrd=bgrd.copy())
            auprc = average_precision_score(
                temp['y_binary'], temp['prob'])
            sensitivity = recall_score(
                temp['y_binary'], temp['pred'])
            specificity = specificity_score(
                temp['y_binary'], temp['pred'])
            precision = precision_score(
                temp['y_binary'], temp['pred'], zero_division=0)
            f1 = f1_score(
                temp['y_binary'], temp['pred'])
            fnr = 1 - sensitivity
            fpr = 1 - specificity
            fdr = 1 - precision

            # Calculate Mitchell stats
            fpr, tpr, _ = roc_curve(temp['y_binary'], temp['prob'])
            auc_df = pd.DataFrame({
                'fpr': fpr,
                'tpr': tpr
            })
            auc_df['specificity'] = 1 - auc_df['fpr']
            # Get difference
            auc_df['diff'] = auc_df['specificity'] - specificity_threshold
            # Get positive value
            auc_df = auc_df.loc[auc_df['diff'] >= 0, :]
            if auc_df.shape[0] > 0:
                # Get highest sensitivity
                idxmax = auc_df['tpr'].idxmax()
                sensitivity_mitchell = auc_df.loc[idxmax, 'tpr']
                specificity_mitchell = auc_df.loc[idxmax, 'specificity']
            else:
                sensitivity_mitchell = 0
                specificity_mitchell = 0

            # Add to res dict
            res_dict['dataset'].append(dataset)
            res_dict['cv_type'].append(cv_type)
            res_dict['training_column'].append(column)
            res_dict['model_type'].append(model_type)
            res_dict['params'].append(str(params))
            res_dict['run'].append(run)
            res_dict['column'].append(c)
            res_dict['value'].append(v)
            res_dict['num_samples'].append(temp.shape[0])
            res_dict['num_ids'].append(temp.study_id.nunique())
            res_dict['pct_pos'].append(temp['y_binary'].mean())
            res_dict['auroc'].append(auroc)
            res_dict['pos_auroc'].append(pos_auroc)
            res_dict['neg_auroc'].append(neg_auroc)
            res_dict['auprc'].append(auprc)
            res_dict['sensitivity'].append(sensitivity)
            res_dict['specificity'].append(specificity)
            res_dict['precision'].append(precision)
            res_dict['f1'].append(f1)
            res_dict['fnr'].append(fnr)
            res_dict['fpr'].append(fpr)
            res_dict['fdr'].append(fdr)
            res_dict['sensitivity_mitchell'].append(sensitivity_mitchell)
            res_dict['specificity_mitchell'].append(specificity_mitchell)

    return pd.DataFrame(res_dict)


def get_best_models(
    res_df, cv_type, dataset,
    training_col
):
    """
    Get best models based upon the AUROC

    :param res_df: pd.DataFrame, df with results
    :param cv_type: <str>, forecast/loro
    :param dataset: <str>, the dataset name
    :param training_col: <str>, the column to use for training

    :return res_df: pd.DataFrame, res_df with only best model
    """
    # Copy
    res_df = res_df.copy()

    # Filter
    res_df = res_df.loc[
        (res_df.cv_type == cv_type) & \
            (res_df.dataset == dataset), :
    ]

    # If entire study, filter to that training_col
    res_df = res_df.loc[
        res_df.training_column == 'Entire study', :]

    # Get best models where the focus
    # is the entire study
    res_df_grouped = res_df.loc[
        res_df['column'] == 'Entire study', :
    ].groupby(
        ['training_column', 'model_type', 'params'], 
        as_index=False)['auroc'].mean()

    best_models = res_df_grouped.loc[
        res_df_grouped.groupby(['training_column'])['auroc'].idxmax(), 
        ['training_column', 'model_type', 'params']
    ]

    res_df = pd.merge(
        left=res_df, right=best_models
    )

    return res_df


def plot_auc_groups(
    res_df, pred_df, group_cols, group_map, order,
    cv_type='forecast',
    dataset='lifesense',
    training_col='Entire study',
):
    """
    Plot a metric broken down by a 
    specific column

    :param res_df: pd.DataFrame, df with results
    :param pred_df: pd.DataFrame, df with predictions
    :param group_cols: list<str>, group column
    :param order: dict, dict for plotting
    :param group_map: dict, the dict with group titles
    :param cv_type: <str>, forecast/loro
    :param dataset: <str>, the dataset name
    :param training_col: <str>, the column to use for training
    """
    fig = plt.figure(figsize=(15, 17))

    res_df = get_best_models(
        res_df=res_df, 
        cv_type=cv_type, 
        dataset=dataset, 
        training_col=training_col,
    )

    # Color palette
    colors = sns.color_palette(
            "colorblind", 
            n_colors=len(group_cols))

    ylabels = {
        'auroc': 'Subgroup AUC',
        'pos_auroc': 'BNSP AUC',
        'neg_auroc': 'BPSN AUC'
    }

    # Go through each group
    # and subplot
    curr = 1
    color_ind = 0
    for g in group_cols:
        for metric in ['auroc', 'pos_auroc', 'neg_auroc']:

            # Get group + entire study
            temp = res_df.loc[
                res_df['column'].isin(['Entire study', g])]

            # Get merge_df for order
            merge_df = pd.DataFrame(
                {'value': order[g]})
            temp = pd.merge(left=merge_df, right=temp, how='left')

            # Plot
            ax = plt.subplot(6, 3, curr)

            # Only display values with average >= 15 individuals
            num_ids = pred_df.groupby([g])['study_id'].nunique()
            display_val = num_ids.loc[num_ids >= 15].index
            temp = temp.loc[temp.value.isin(display_val), :]

            # Reformat
            temp['value'] = [
                t.replace(' American', '\nAmerican'
                    ).replace(' one race', '\none race')  \
                for t in temp['value']
            ]
            if g == 'demo_fam_income':
                temp['value'] = [
                    t.replace(' to ', ' to\n') for t \
                        in temp['value']
                ]
            num_ids = pred_df.groupby([g])['study_id'].nunique()
            display_val = num_ids.loc[num_ids >= 15].index

            # Get groups
            all_groups = temp['value'].unique()

            # Get median across groups
            if len(all_groups) > 2:
                # Get median across groups
                base = temp.loc[
                    (temp['value'] != 'Entire study'), :
                ].groupby(['run'], as_index=False
                    )[metric].median()
            else:
                # Find higher group
                base = temp.loc[
                    (temp['value'] != 'Entire study'), :
                ].groupby(['value'], as_index=False
                    )[metric].median()
                v = base.loc[base[metric].idxmax(), 'value']
                base = temp.loc[
                    (temp['value'] == v), ['run', metric]
                ]
            
            # Rename column
            base.rename(
                columns={metric: 'base'},
                inplace=True
            )

            # Plot background median/CI
            m = base['base'].median()
            low = base['base'].quantile(0.025)
            high = base['base'].quantile(0.975)

            x = [-0.5, len(all_groups) + 0.5]
            ax.fill_between(x, 
                low, high, color=colors[color_ind],
                alpha=0.25
            )
            ax.axhline(m, ls='--', color=colors[color_ind])

            # Plot results
            ax = sns.pointplot(
                data=temp,
                x='value',
                y=metric,
                estimator='median',
                color=colors[color_ind],
                errorbar=('pi', 95),
                join=True,
                capsize=0.2
            )

            print(metric)
            print(temp.groupby(
                ['value']
            )[metric].agg(
                ['median', percentile(2.5), percentile(97.5)])
            )

            if curr < 4:
                plt.title(ylabels[metric], size=16, pad=20)

            plt.xticks(rotation=90)
            plt.xlabel('')
            if (curr % 3) == 1:
                plt.ylabel('Value', size=16, labelpad=20)
            else:
                plt.ylabel('')
            plt.xticks(size=16)
            plt.yticks(size=16)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            curr += 1
        color_ind += 1

    sns.despine()
    plt.tight_layout()


def plot_auc_groups_by_model_type(
    res_df, group_cols, group_map, order,
    cv_type='forecast',
    dataset='lifesense',
    training_col='Entire study',
):
    """
    Plot a metric broken down by a 
    specific column

    :param res_df: pd.DataFrame, df with results
    :param group_cols: list<str>, group column
    :param order: dict, dict for plotting
    :param group_map: dict, the dict with group titles
    :param cv_type: <str>, forecast/loro
    :param dataset: <str>, the dataset name
    :param training_col: <str>, the column to use for training
    """
    fig = plt.figure(figsize=(15, 17))

    model_type_map = {
        'lr': 'Logistic\nRegression',
        'sv': 'Support Vector\nMachine',
        'rf': 'Random\nForrest',
        'gb': 'Gradient\nBoosting\nTree'
    }

    ylabels = {
        'auroc': 'Subgroup AUC',
        'pos_auroc': 'BNSP AUC',
        'neg_auroc': 'BPSN AUC'
    }

    res_dfs = []
    for m in ['lr', 'sv', 'rf', 'gb']:

        temp_res_df = get_best_models(
            res_df=res_df.loc[res_df.model_type == m, :], 
            cv_type=cv_type, 
            dataset=dataset, 
            training_col=training_col,
        )
        res_dfs.append(temp_res_df)

    res_df = pd.concat(res_dfs).reset_index(drop=True)
    res_df['model_type'] = [model_type_map[m] for m in res_df.model_type]

    # Go through each group
    # and subplot
    curr = 1
    for g in group_cols:
        for metric in ['auroc', 'pos_auroc', 'neg_auroc', 'legend']:

            if metric == 'legend':
                plt.subplot(6, 4, curr)
                leg = plt.legend(
                    handles=h,
                    labels=l,
                    loc=(-0.2, 0.0),
                    fontsize=12,
                )
                leg._legend_box.align = 'left'
                plt.axis('off')
                curr += 1
                continue

            # Get group + entire study
            temp = res_df.loc[
                res_df['column'].isin(['Entire study', g])]

            # Get merge_df for order
            merge_df = pd.DataFrame(
                {'value': order[g]})
            temp = pd.merge(left=merge_df, right=temp, how='left')

            # Plot
            plt.subplot(6, 4, curr)

            # Only display values with average >= 15 individuals
            temp_grouped = temp.groupby(['value'])['num_ids'].mean()
            display_val = temp_grouped.loc[temp_grouped >= 15].index

            # Color palette
            colors = sns.color_palette(
                "colorblind", 
                n_colors=len(display_val)
            )
            ax = sns.barplot(
                data=temp.loc[temp.value.isin(display_val), :],
                x='model_type',
                y=metric,
                hue='value',
                palette=colors,
                estimator='median',
                errorbar=('pi', 95),
                capsize=0.05,
                width=0.75
            )

            if curr < 4:
                plt.title(ylabels[metric], size=16, pad=20)

            h, l = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            plt.xticks(rotation=90)
            plt.xlabel('')
            plt.ylabel('')
            plt.xticks(size=16)
            plt.yticks(size=16)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            curr += 1

    sns.despine()
    plt.tight_layout()


def lr_auc_groups_by_ground_truth(
    pred_df,
    group_cols,
    group_map,
):
    """
    Risk LR for groups by ground truth

    :param pred_df: pd.DataFrame, df with probabilities
    :param group_cols: list<str>, group column
    :param group_map: <str>, the group mapping
    """

    # Run logistic regression
    pred_df['bias'] = 1
    # Run GEE for each class
    features = []
    coefs = []
    pvalues = []
    low = []
    high = []
    y_binary = []
    idx_dict = dict()
    num_cols = dict()
    for y in [0, 1]:
        lr_df = pred_df.loc[pred_df.y_binary == y, :].copy()
        # Group is sample, identified by study_id/outcome_index
        lr_df['group'] = [
            str(lr_df.loc[ind, 'study_id']) + \
                str(lr_df.loc[ind, 'outcome_index']) for ind in \
                    lr_df.index
        ]
        idx_dict[y] = dict()
        dummy_cols = []
        for c in group_cols:
            # Get groups to keep
            keep_groups = pred_df.groupby([c])['study_id'].nunique()
            keep_groups = keep_groups.loc[keep_groups >= 15].index
            # Get dummies
            dummies = pd.get_dummies(
                lr_df[c], prefix=c, prefix_sep='|')
            # Identify min/max group
            idx_df = lr_df.groupby([c]).agg({'prob': 'mean', 'study_id': 'nunique'})
            idx_df = idx_df.loc[idx_df.index.isin(keep_groups), :]
            if y == 0:
                idx = idx_df['prob'].idxmin()
            else:
                idx = idx_df['prob'].idxmax()
            idx_dict[y][c] = idx
            ind = [i for i in idx_df.index if i != idx]
            cols = [c + '|' + i for i in ind]
            dummy_cols += cols
            lr_df = pd.concat([lr_df, dummies[cols]], axis=1)
        cols = dummy_cols[:]
        # Run GEE
        lr = sm.GEE(
            endog=lr_df['prob'],
            exog=lr_df[
                cols + ['bias']
            ].astype(float),
            groups=lr_df['group'],
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()
        features += cols
        coefs += list(lr.params.loc[cols])
        pvalues += list(lr.pvalues.loc[cols])
        y_binary += [y] * len(cols)
        conf_int = lr.conf_int().loc[cols]
        low += [conf_int.loc[c, 0] for c in cols]
        high += [conf_int.loc[c, 1] for c in cols]
        num_cols[y] = len(cols)

    lr_df = pd.DataFrame({
        'PHQ-8': y_binary,
        'feature': features,
        'coef': coefs,
        'low': low,
        'high': high,
        'pvalue': pvalues,
    })
    lr_df['sig'] = 0
    lr_df.loc[lr_df.pvalue < 0.05, 'sig'] = 1
    # For Bonferroni
    alpha_bonf = 0.05 / len(cols)
    lr_df['bonf_sig'] = 0
    lr_df.loc[lr_df.pvalue < alpha_bonf, 'bonf_sig'] = 1


    lr_df['coef'] = lr_df['coef']
    lr_df['column'] = [g.split('|')[0] for g in lr_df.feature]
    lr_df['Comparison Group'] = [
        idx_dict[lr_df.loc[i, 'PHQ-8']][lr_df.loc[i, 'column']] \
            for i in lr_df.index
    ]
    lr_df['value'] = [g.split('|')[1].split(' x' )[0] for g in lr_df.feature]
    lr_df['Attribute'] = [
        group_map[g.split('|')[0]] for g in lr_df.feature]
    lr_df['Group'] = [
        g.split('|')[1] for g in lr_df.feature]
    lr_df['β (95% CI)'] = [
        str(np.round(lr_df.loc[i, 'coef'], 2)) + ' (' + \
            str(np.round(lr_df.loc[i, 'low'], 2)) + ' to ' + \
            str(np.round(lr_df.loc[i, 'high'], 2)) + ')' \
            for i in lr_df.index
    ]
    
    print(0, idx_dict[0])
    print(1, idx_dict[1])

    # Get num_ids
    lr_df['num_ids'] = 0
    for ind in lr_df.index:
        lr_df.loc[ind, 'num_ids'] = \
            pred_df.loc[
                pred_df[lr_df.loc[ind, 'column']] == \
                    lr_df.loc[ind, 'value'], 'study_id'
            ].nunique()
        
    return lr_df


def plot_auroc_w_prevalence(
    pred_df, res_df, group_cols, cv_type, 
    order, group_map
):
    """
    Plot the AUROC by each class
    with prevalence

    :param pred_df: pd.DataFrame, the average risk
    :param res_df: pd.DataFrame, the results with prevalences
    :param group_cols: list<str>, the group columns
    :param cv_type: <str>, the cv type
    :param order: dict<str>, order to plot
    :param group_map: dict, the group map
    """
    plt.figure(figsize=(15, 17))

    # Get best model
    best_res_df = get_best_models(
        res_df=res_df,
        cv_type=cv_type, 
        dataset='lifesense',
        training_col='Entire study'
    )
    # Filter to entire study columns
    group_res_df = best_res_df.loc[best_res_df.column.isin(group_cols), :]

    # Get groups
    group_res_df_grouped = group_res_df.groupby([
        'run', 'column', 'value'
    ], as_index=False)[
        ['pct_pos', 'num_samples', 'num_ids']].mean()
    group_res_df_grouped['% PHQ-8 ≥10'] = \
        (group_res_df_grouped['pct_pos'] * 100).round()
    group_res_df_grouped['Number of Participants'] = \
        (group_res_df_grouped['num_ids'])

    # Y labels
    ylabels = {
        'pos_auroc': 'BNSP AUC',
        'neg_auroc': 'BPSN AUC'
    }

    # Colors
    colors = sns.color_palette(
            "colorblind", 
            n_colors=len(group_cols))
    color_ind = 0

    curr = 1

    # Go through each group
    for g in group_cols:

        # Filter to groups with n>=15
        keep_groups = pred_df.groupby(
            [g])['study_id'].nunique()
        keep_groups = keep_groups.loc[
            keep_groups >= 15].index

        plt.subplot(6, 3, curr)

        temp = group_res_df_grouped.loc[
            group_res_df_grouped.column == g, :]
        temp = temp.loc[temp.value.isin(keep_groups), :]
        temp = pd.merge(
            left=pd.DataFrame({'value': order[g]}),
            right=temp
        ).copy()
        temp['value'] = [
            t.replace(' American', '\nAmerican'
                ).replace(' one race', '\none race')  \
            for t in temp['value']
        ]
        if g == 'demo_fam_income':
            temp['value'] = [
                t.replace(' to ', ' to\n') for t \
                    in temp['value']
            ]

        # Plot prevalence   
        sns.barplot(
            x=temp['value'],
            y=temp['% PHQ-8 ≥10'],
            color=colors[color_ind],
            capsize=0.25,
            errorbar=('pi', 95),
        )
        plt.xlabel('')
        plt.ylabel(group_map[g] + 
            '\n\nValue', size=16, labelpad=20)
        if curr == 1:
            plt.title('Group Prevalence (%)', 
                    size=16, pad=20)
        plt.xticks(size=16, rotation=90)
        plt.yticks(size=16)

        curr += 1

        # Plot BPSN/BNSP
        for metric in ['pos_auroc', 'neg_auroc']:

            # Get group + entire study
            temp = best_res_df.loc[
                best_res_df['column'].isin(['Entire study', g])]

            # Get merge_df for order
            merge_df = pd.DataFrame(
                {'value': order[g]})
            temp = pd.merge(left=merge_df, right=temp, how='left')

            # Plot
            ax = plt.subplot(6, 3, curr)

            # Only display values with average >= 15 individuals
            num_ids = pred_df.groupby([g])['study_id'].nunique()
            display_val = num_ids.loc[num_ids >= 15].index
            temp = temp.loc[temp.value.isin(display_val), :]

            # Reformat
            temp['value'] = [
                t.replace(' American', '\nAmerican'
                    ).replace(' one race', '\none race')  \
                for t in temp['value']
            ]
            if g == 'demo_fam_income':
                temp['value'] = [
                    t.replace(' to ', ' to\n') for t \
                        in temp['value']
                ]
            num_ids = pred_df.groupby([g])['study_id'].nunique()
            display_val = num_ids.loc[num_ids >= 15].index

            # Get groups
            all_groups = temp['value'].unique()

            # Get median across groups
            if len(all_groups) > 2:
                # Get median across groups
                base = temp.loc[
                    (temp['value'] != 'Entire study'), :
                ].groupby(['run'], as_index=False
                    )[metric].median()
            else:
                # Find higher group
                base = temp.loc[
                    (temp['value'] != 'Entire study'), :
                ].groupby(['value'], as_index=False
                    )[metric].median()
                v = base.loc[base[metric].idxmax(), 'value']
                base = temp.loc[
                    (temp['value'] == v), ['run', metric]
                ]
            
            # Rename column
            base.rename(
                columns={metric: 'base'},
                inplace=True
            )

            # Plot median/CI background
            m = base['base'].median()
            low = base['base'].quantile(0.025)
            high = base['base'].quantile(0.975)

            x = [-0.5, len(all_groups) + 0.5]
            ax.fill_between(x, 
                low, high, color=colors[color_ind],
                alpha=0.25
            )
            ax.axhline(m, ls='--', color=colors[color_ind])

            ax = sns.pointplot(
                data=temp,
                x='value',
                y=metric,
                estimator='median',
                color=colors[color_ind],
                errorbar=('pi', 95),
                join=True,
                capsize=0.2
            )

            if curr < 4:
                plt.title(ylabels[metric], size=16, pad=20)

            plt.xticks(rotation=90)
            plt.xlabel('')
            if (curr % 3) == 1:
                plt.ylabel('Value', size=16, labelpad=20)
            else:
                plt.ylabel('')
            plt.xticks(size=16)
            plt.yticks(size=16)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            curr += 1
        color_ind += 1

    sns.despine()
    plt.tight_layout()


def plot_auroc_w_samples(
    pred_df, res_df, cv_type, group_cols, order, group_map
):
    """
    Plot the number of samples and the AUROC

    :param pred_df: pd.DataFrame, the average risk
    :param res_df: pd.DataFrame, the results with prevalences
    :param cv_type: <str>, the cv_type
    :param group_cols: list<str>, the group columns
    :param order: dict<str>, order to plot
    :param group_map: dict, the group map
    """
    plt.figure(figsize=(15, 17))

    # Get best model
    best_res_df = get_best_models(
        res_df=res_df,
        cv_type=cv_type, 
        dataset='lifesense',
        training_col='Entire study'
    )
    # Filter to entire study columns
    group_res_df = best_res_df.loc[best_res_df.column.isin(group_cols), :]

    # Get groups
    group_res_df_grouped = group_res_df.groupby([
        'run', 'column', 'value'
    ], as_index=False)[
        ['pct_pos', 'num_samples', 'num_ids', 'auroc']].mean()
    group_res_df_grouped['% PHQ-8 ≥10'] = \
        (group_res_df_grouped['pct_pos'] * 100).round()
    group_res_df_grouped['Number of Participants'] = \
        (group_res_df_grouped['num_ids'])

    curr = 1

    # Colors
    colors = sns.color_palette(
            "colorblind", 
            n_colors=len(group_cols))
    color_ind = 0

    # Get each group
    for c in group_cols:

        keep_groups = pred_df.groupby(
            [c])['study_id'].nunique()
        keep_groups = keep_groups.loc[
            keep_groups >= 15].index

        plt.subplot(6, 2, curr)

        temp = group_res_df_grouped.loc[
            group_res_df_grouped.column == c, :]
        temp = pd.merge(
            left=pd.DataFrame({'value': order[c]}),
            right=temp
        ).copy()
        temp = temp.loc[temp.value.isin(keep_groups), :]
        # Only display values with average >= 15 individuals
        temp['value'] = [
            t.replace(' American', '\nAmerican'
                ).replace(' one race', '\none race')  \
            for t in temp['value']
        ]
        if c == 'demo_fam_income':
            temp['value'] = [
                t.replace(' to ', ' to\n') for t \
                    in temp['value']
            ]

        # Plot number of samples     
        sns.barplot(
            x=temp['value'],
            y=np.log(temp['num_samples']),
            color=colors[color_ind],
            capsize=0.25,
            estimator='median',
            errorbar=('pi', 95),
        )
        plt.xlabel('')
        plt.ylabel(group_map[c] + \
                '\n\nLog Number of\nSamples', 
                size=16, labelpad=20)
        plt.xticks(size=16, rotation=90)
        plt.yticks(size=16)

        curr += 1

        ax = plt.subplot(6, 2, curr)

        # Get median across groups
        all_groups = temp['value'].unique()
        if len(all_groups) > 2:
            # Get median across groups
            base = temp.loc[
                (temp['value'] != 'Entire study'), :
            ].groupby(['run'], as_index=False
                )['auroc'].median()
        else:
            # Find higher group
            base = temp.loc[
                (temp['value'] != 'Entire study'), :
            ].groupby(['value'], as_index=False
                )['auroc'].median()
            v = base.loc[base['auroc'].idxmax(), 'value']
            base = temp.loc[
                (temp['value'] == v), ['run', 'auroc']
            ]
        
        # Rename column
        base.rename(
            columns={'auroc': 'base'},
            inplace=True
        )

        # Plot median/CI background
        m = base['base'].median()
        low = base['base'].quantile(0.025)
        high = base['base'].quantile(0.975)

        x = [-0.5, len(all_groups) + 0.5]
        ax.fill_between(x, 
            low, high, color=colors[color_ind],
            alpha=0.25
        )
        ax.axhline(m, ls='--', color=colors[color_ind])

        # Plot pointplot
        ax = sns.pointplot(
            x=temp['value'],
            y=temp['auroc'],
            color=colors[color_ind],
            capsize=0.25,
            estimator='median',
            errorbar=('pi', 95),
        )
        plt.xlabel('')
        plt.ylabel('Subgroup\nAUC', 
                size=16, labelpad=20)
        plt.xticks(size=16, rotation=90)
        plt.yticks(size=16)

        curr += 1

        color_ind += 1

    sns.despine()
    plt.tight_layout()


def feature_lr_with_group(
    df, features, group_cols
):
    """
    Run logistic regression for features
    with group_cols as dummy variables

    :param df: pd.DataFrame, dataframe (entire study)
    :param features: list<str>, the features used for modeling
    :param group-cols: list<str>, columns to run analysis over

    :return lr_df: pd.DataFrame, the df with added lr values
    """
    lr_dict = {
        'column': [],
        'value': [],
        'majority': [],
        'num_samples': [],
        'feature': [],
        'lr_coef': [],
        'lr_pvalue': [],
        'lr_low': [],
        'lr_high': [],
        'median': [],
        'low_95': [],
        'high_95': [],
        'num_features': [],
    }

    # Impute and scale data
    df[features] = IterativeImputer(
        max_iter=1000).fit_transform(df[features])
    # Standardize
    df[features] = StandardScaler().fit_transform(df[features])

    for g in group_cols:
        # Filter to groups with >= 15 IDs
        study_ids_group = df[['study_id', g]].drop_duplicates()
        # Filter to groups with >=15 study IDs 
        # as per Seyyed-Kalantari et al. 2021
        val_counts = study_ids_group[g].value_counts()
        val_counts = val_counts.loc[val_counts >= 15]
        keep_groups = val_counts.index
        study_ids_group = study_ids_group.loc[
            study_ids_group[g].isin(keep_groups), 'study_id']

        # Get study IDs
        log_df = df.loc[
            df.study_id.isin(study_ids_group), :].copy()

        # Get averages
        med_df = log_df.groupby([g])[features].median()
        low_df = log_df.groupby([g])[features].quantile(0.025)
        high_df = log_df.groupby([g])[features].quantile(0.975)

        # Make dummy vars
        dummy_df = pd.get_dummies(log_df[g])
        # Save non-majority
        val_counts = log_df[g].value_counts().sort_values(ascending=False)
        dummy_df = dummy_df[val_counts.index[1:]]
        majority = val_counts.index[0]
        # Get values
        dummy_cols = list(dummy_df.columns)
        # Concatenate column wise
        log_df = pd.concat([log_df, dummy_df], axis=1)
        # Reset index
        log_df.reset_index(drop=True, inplace=True)
        # Bias
        log_df['bias'] = 1

        # Calculate logistic regression for each feature
        # and dummy cols
        for f in features:
            # Get dummy by feature cols
            dummy_by_features = []
            for d in dummy_cols:
                log_df[f + '|' + d] = log_df[f] * log_df[d]
                dummy_by_features.append(f + '|' + d)
            # Add to dict
            cols = [f] + dummy_cols + dummy_by_features
            keep_cols = [f] + dummy_by_features
            num_cols = len(keep_cols)
            lr_dict['column'] += [g] * num_cols
            lr_dict['feature'] += [f] * num_cols
            lr_dict['value'] += [majority] \
                + [d.split('|')[1] for d in dummy_by_features]
            lr_dict['majority'] += [1] + \
                    [0] * len(dummy_by_features)
            lr_dict['num_samples'] += [val_counts.loc[majority]] \
                + [val_counts.loc[d.split('|')[1]] \
                    for d in dummy_by_features]
            lr_dict['num_features'] += [num_cols] * num_cols
            lr_dict['median'] += [
                med_df.loc[v, f] for v in [majority] \
                    + [d.split('|')[1] for d in dummy_by_features]
            ]
            lr_dict['low_95'] += [
                low_df.loc[v, f] for v in [majority] \
                    + [d.split('|')[1] for d in dummy_by_features]
            ]
            lr_dict['high_95'] += [
                high_df.loc[v, f] for v in [majority] \
                    + [d.split('|')[1] for d in dummy_by_features]
            ]

            temp = log_df[
                ['study_id', 'bias', 'y_binary'] + \
                    cols
            ].dropna()
            # Logistic regression   
            lr = sm.Logit(
                endog=temp['y_binary'].astype(int),
                exog=temp[['bias'] + cols].astype(float),
            ).fit(disp=0)
            # Coefs
            coefs = list(lr.params.loc[keep_cols])
            pvalues = list(lr.pvalues.loc[keep_cols])
            conf_int = lr.conf_int().loc[keep_cols]
            lr_dict['lr_coef'] += \
                [coefs[0]] + [c + coefs[0] for c in coefs[1:]]
            lr_dict['lr_low'] += \
                [conf_int.iloc[0, 0]] + \
                    [coefs[0] + conf_int.loc[i, 0] \
                        for i in conf_int.index[1:]]
            lr_dict['lr_high'] += \
                [conf_int.iloc[0, 1]] + \
                    [coefs[0] + conf_int.loc[i, 1] \
                        for i in conf_int.index[1:]]
            lr_dict['lr_pvalue'] += pvalues
            # Intercepts
            intercept = list(lr.params.loc[['bias'] + dummy_cols])
            intercept_pvalues = list(lr.pvalues.loc[['bias'] + dummy_cols])
            intercept_conf_int = lr.conf_int().loc[['bias'] + dummy_cols]
            
    # Make lr df
    lr_df = pd.DataFrame(lr_dict)

    # Mark bonferroni corrected significant values
    lr_df['alpha'] = 0.05
    lr_df['alpha_bonf'] = lr_df['alpha'] / lr_df['num_features']
    lr_df['lr_sig'] = 0
    lr_df['lr_bonf_sig'] = 0
    lr_df.loc[lr_df['lr_pvalue'] < lr_df.alpha, 'lr_sig'] = 1
    lr_df.loc[lr_df['lr_pvalue'] < lr_df.alpha_bonf, 'lr_bonf_sig'] = 1
        
    return lr_df

def sensed_behavior_distributions(
    df, features, group_cols
):
    """
    Get distributions of sensed-behaviors (mean, standard deviation)

    :param df: pd.DataFrame, dataframe (entire study)
    :param features: list<str>, the features used for modeling
    :param group-cols: list<str>, columns to run analysis over

    :return lr_df: pd.DataFrame, the df with added lr values
    """
    dist_dict = dict()

    # Add entire study to group_cols
    group_cols = ['Entire study'] + group_cols

    # Impute and scale data
    df[features] = IterativeImputer(
        max_iter=1000).fit_transform(df[features])
    # Standardize
    df[features] = StandardScaler().fit_transform(df[features])

    for g in group_cols:
        # Filter to groups with >= 15 IDs
        study_ids_group = df[['study_id', g]].drop_duplicates()
        # Filter to groups with >=15 study IDs 
        # as per Seyyed-Kalantari et al. 2021
        val_counts = study_ids_group[g].value_counts()
        val_counts = val_counts.loc[val_counts >= 15]
        keep_groups = val_counts.index

        # Get study IDs to keep
        df_keep = df.loc[
            df[g].isin(keep_groups), :].copy()

        # Make df
        dist_df = pd.DataFrame(
            index=features, columns=keep_groups)

        # Go through each feature
        for f in features:
            # Go through each value
            for v in keep_groups:
                # Create empty string
                curr_str = ''
                # Filter
                temp = df_keep.loc[(df_keep[g] == v), :]
                # Get median
                curr_str += str(temp[f].median().round(2))
                curr_str += ' ('
                # Get 25th percentile
                curr_str += str(temp[f].quantile(0.25).round(2))
                curr_str += ' to '
                # Get 75th percentile
                curr_str += str(temp[f].quantile(0.75).round(2))
                curr_str += ')'

                # Set value
                dist_df.loc[f, v] = curr_str

        # Add to dict
        dist_dict[g] = dist_df

            
   # Return
    return dist_dict
