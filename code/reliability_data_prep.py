"""
reliability_data_prep.py

Prep data for reliability analysis
"""

# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import timedelta
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import jaccard
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns


def age_mapping(x):
    """
    Map age to categories

    :param x: <int>, the age

    :return <str>: the age category
    """
    if x < 25:
        return '18 to 25'
    elif x < 35:
        return '25 to 34'
    elif x < 45:
        return '35 to 44'
    elif x < 55:
        return '45 to 54'
    elif x < 65:
        return '55 to 64'
    elif x < 75:
        return '65 to 74'
    elif x < 85:
        return '75 to 84'
    elif x >= 85:
        return 'Greater than 85'
    else:
        return 'Not answered'


def upload_lifesense_data():
    """
    Upload the lifesense data

    :return train_df: pd.DataFrame, the training data
    :return test_df: pd.DataFrame, the testing data
    :return features: list<str>, the list of features
    """
    # Upload data
    feature_df = pd.read_csv('/home/dadler/lifesense_data/feature_df.csv')

    # Upload demographics
    path = '/home/dadler/lifesense_data/self_reports/'
    w1_initial = pd.read_excel(
        path + 'W1/Wave_1_Android_Initial_Deployment.xlsx', 
        sheet_name='Self_Report_ScreenBL'
    )

    w1_main = pd.read_excel(
        path + 'W1/Wave_1_Android_Main_Deployment.xlsx', 
        sheet_name='Self_Report_ScreenBL'
    )

    w2 = pd.read_excel(
        path + 'W2/Wave_2.xlsx', 
        sheet_name='Self_Report'
    )

    w3 = pd.read_excel(
        path + 'W3/Wave_3.xlsx', 
        sheet_name='Self_Report'
    )

    # For w2 and w3 filter to screener
    w2 = w2.loc[w2['redcap_event_name'] == 'scbl_arm_1', :].reset_index(drop=True) 
    w3 = w3.loc[w3['redcap_event_name'] == 'scbl_arm_1', :].reset_index(drop=True) 

    # Map IDs
    id_map = pd.read_excel('~/lifesense_data/lifesense_enrolled_app_ids.xlsx')

    id_map['wave'] = id_map['group'].map({
        'Wave_1_Android_Initial_Deployment': 'wave1_init',
        'Wave_1_Android_Main_Deployment': 'wave1_main',
        '1. LifeSense Wave 2 iOS Wave': 'wave2',
        '1. LifeSense Wave 2 Android, FPG Group 1': 'wave2',
        '0. LifeSense Wave 2 Android, FPG Group 2': 'wave2',
        'Wave 3_Android_Social_Media_Registry_Group_1': 'wave3',
        'Wave 3_Android_Social_Media_Registry_Group_2': 'wave3',
        'Wave 3_Android_Group_3': 'wave3'
    })

    # Columns to analyze
    cols = [
        'record_id', 'wave',
        'age', 'gender_id', 'demo_gender',
        'demo_race',
        'demo_hispanic',
        'demo_fam_income', 'demo_personal_income',
        'demo_health_insurance',
        'routine_slabels02',
        'slabels03a', # slabels03 not in W3
        'work_schd',
        'slabels03b', 'slabels03c', 'slabels03d',
        'slabels04', 'slabels04a'
    ]

    # Add wave
    w1_initial['wave'] = 'wave1_init'
    w1_main['wave'] = 'wave1_main'
    w2['wave'] = 'wave2'
    w3['wave'] = 'wave3'

    # Concat
    self_report_df = pd.concat(
        [w1_initial[cols], w1_main[cols], w2[cols], w3[cols]]
    ).reset_index(drop=True)

    # Merge
    self_report_df = pd.merge(
        left=self_report_df,
        right=id_map[['record_id', 'app_id', 'wave']],
        on=['record_id', 'wave']
    )

    # Map demographic values
    mapping_dict = {
        'gender_id': {
            0: 'Female',
            1: 'Male',
            2: 'Non-binary/third gender',
            3: 'Transgender',
            88: 'Prefer to self-describe',
            99: 'Prefer not to answer'
        },
        'demo_gender': {
            0: 'Female',
            1: 'Male',
        },
        'demo_race': {
            1: 'Black/African American',
            2: 'Other',
            3: 'Asian/Asian American',
            4: 'Other',
            5: 'White',
            6: 'More than one race',
            77: 'Decline to report',
            99: 'Prefer not to answer',
        },
        'demo_hispanic': {
            0: 'Non-Hispanic/Non-Latinx',
            1: 'Hispanic/Latinx',
            77: "Don't know",
            99: 'Prefer not to answer'
        },
        'demo_fam_income': {
            1: '<20,000',
            2: '<20,000',
            3: '20,000 to 39,999',
            4: '40,000 to 59,999',
            5: '60,000 to 99,999',
            6: '100,000+',
            7: "Don't know",
            99: "Prefer not to answer"
        },
        'demo_personal_income': {
            1: '<20,000',
            2: '<20,000',
            3: '20,000 to 39,999',
            4: '40,000 to 59,999',
            5: '60,000 to 99,999',
            6: '100,000+',
            7: "Don't know",
            99: "Prefer not to answer"
        },
        'demo_health_insurance': {
            0: "Uninsured",
            1: 'Insured',
            77: "Don't know",
            99: "Prefer not to answer",
        },
        'routine_slabels02': {
            1: 'Employed',
            2: 'Unemployed',
            3: 'Disability',
            4: 'Retired',
            88: 'Other',
            99: 'Prefer not to answer',
        }, 
        'work_schd': {
            1: 'Fixed work schedule',
            2: 'Varied work schedule',
            77: 'Left blank',
            99: 'Prefer not to answer'
        },
        'slabels03b': {
            1: 'Work outside of home',
            2: 'Work from home',
            3: 'Do not work from one place',
            77: 'Left blank',
            99: 'Prefer not to answer'
        },
        'slabels03c': {
            1: 'Do not work from home',
            2: '<1 day at home',
            3: '1 day at home',
            4: '2 days at home',
            5: '3 days at home',
            6: '4 days at home',
            7: '5+ days at home',
            77: 'Left blank',
            99: 'Prefer not to answer'
        },
        'slabels03d': {
            1: 'Do not travel',
            2: '<1 day traveled',
            3: '1 day traveled',
            4: '2 days traveled',
            5: '3 days traveled',
            6: '4 days traveled',
            7: '5+ days traveled',
            77: 'Left blank',
            99: 'Prefer not to answer'
        },
        'slabels04': {
            0: 'Not a student',
            1: 'Student',
            99: 'Prefer not to answer'
        },
        'slabels04a': {
            1: 'Full-time student',
            2: 'Part-time student',
            77: 'Left blank',
            99: 'Prefer not to answer'
        }
    }

    # Map
    for c in mapping_dict.keys():
        # Fill with 77 if na in employment column/student column
        if c in [
           'work_schd', 'slabels03b', 
            'slabels03c',  'slabels03d',
            'slabels04a'
        ]:
            self_report_df[c] = self_report_df[c].fillna(77)
        # Map
        self_report_df[c] = self_report_df[c].map(mapping_dict[c])

    # Map age
    self_report_df['age'] = self_report_df['age'].apply(age_mapping)

    # Merge onto feature df
    merge_cols = [
        'study_id',
        'age', 'gender_id', 'demo_gender',
        'demo_race',
        'demo_hispanic',
        'demo_fam_income',
        'demo_personal_income',
        'demo_health_insurance',
        'routine_slabels02',
        'work_schd', 'slabels03b', 
        'slabels03c',  'slabels03d',
        'slabels04', 'slabels04a'
    ]

    self_report_df['study_id'] = self_report_df['app_id'].copy()

    # Merge
    feature_df = pd.merge(
        left=feature_df,
        right=self_report_df[merge_cols],
        on=['study_id'],
        how='left'
    )

    # Merge seasonal information
    season_map = {
        1: 'Winter',
        2: 'Winter',
        3: 'Spring',
        4: 'Spring',
        5: 'Spring',
        6: 'Summer',
        7: 'Summer',
        8: 'Summer',
        9: 'Fall',
        10: 'Fall',
        11: 'Fall'
    }

    month_map = {
        1: 'January',
        2: 'February',
        3: 'March',
        4: 'April',
        5: 'May',
        6: 'June',
        7: 'July',
        8: 'August',
        9: 'September',
        10: 'October',
        11: 'November'
    }

    feature_df['month'] = [month_map[int(d.split('-')[1])] for d in feature_df.outcome_date]
    feature_df['season'] = [season_map[int(d.split('-')[1])] for d in feature_df.outcome_date]

    # Change
    feature_df['wave'] = feature_df['wave'].map({1: 'Wave 1', 2: 'Wave 2', 3: 'Wave 3'})

    # Prep for model training
    features = [
        'location_entropy',
        'location_norm_entropy',
        'location_log_var',
        'location_log_pgram',
        'location_log_num_clusters',
        'location_log_home_stay',
        'location_log_transition_time',
        'screen-state_sleep_onset_mean_daily',
        'screen-state_log_sleep_duration_mean_daily',
        'screen-state_log_sleep_duration_std_daily',
        'screen-state_log_unlock_duration_sum_mean_daily',
        'screen-state_log_unlock_duration_sum_mean_1',
        'screen-state_log_unlock_duration_sum_mean_2',
        'screen-state_log_unlock_duration_sum_mean_3',
        'screen-state_log_unlock_duration_sum_mean_4',
        'screen-state_log_unlock_duration_sum_std_daily',
        'screen-state_log_unlock_duration_sum_std_1',
        'screen-state_log_unlock_duration_sum_std_2',
        'screen-state_log_unlock_duration_sum_std_3',
        'screen-state_log_unlock_duration_sum_std_4',
        'screen-state_log_unlock_duration_count_mean_daily',
        'screen-state_log_unlock_duration_count_mean_1',
        'screen-state_log_unlock_duration_count_mean_2',
        'screen-state_log_unlock_duration_count_mean_3',
        'screen-state_log_unlock_duration_count_mean_4',
        'screen-state_log_unlock_duration_count_std_daily',
        'screen-state_log_unlock_duration_count_std_1',
        'screen-state_log_unlock_duration_count_std_2',
        'screen-state_log_unlock_duration_count_std_3',
        'screen-state_log_unlock_duration_count_std_4',
        'screen-state_log_unlock_duration_count_count_daily',
        'screen-state_log_unlock_duration_count_count_1',
        'screen-state_log_unlock_duration_count_count_2',
        'screen-state_log_unlock_duration_count_count_3',
        'screen-state_log_unlock_duration_count_count_4',
    ]

    # Train a model on the feature set to predict
    ml_df = feature_df.copy()
    ml_df['y_binary'] = 0
    ml_df.loc[ml_df['outcome'] >= 10, 'y_binary'] = 1

    # For all clusters
    ml_df['all_cluster'] = 0
    ml_df['Entire study'] = 'Entire study'

    # Mark outcomes
    ml_df = ml_df.sort_values(by=['study_id', 'outcome_date']).reset_index(drop=True)
    ml_df['outcome_index'] = None
    for s in ml_df.study_id.unique():
        nrows = (ml_df.study_id == s).sum()
        ml_df.loc[ml_df.study_id == s, 'outcome_index'] = list(range(1, nrows + 1))

    # Filter to study_id's with 6 values
    max_index = ml_df.outcome_index.max()
    study_ids_max = ml_df.loc[ml_df.outcome_index == max_index, 'study_id'].unique()
    print('Total IDs', ml_df.study_id.nunique())
    ml_df = ml_df.loc[ml_df.study_id.isin(study_ids_max), :].reset_index(drop=True)
    print('IDs kept', ml_df.study_id.nunique())

    # Return train, test, features
    return ml_df, features


def get_data_stats(df, demo_cols):
    """
    Get data statistics

    :param df: pd.DataFrame, the data
    :param demo_cols: list<str>, demographic columns
    """
    # Train/test information
    demo_summary_df = pd.DataFrame(
        columns=['num_samples', 'num_ids', '%'])
    demo_summary_df.loc['Entire study', :] = [
        df.shape[0], df.study_id.nunique(), None]
    demo_summary_df.loc['Class Balance (%)', :] = [
        None, None, int(np.round(df['y_binary'].mean() * 100))]

    for c in demo_cols:
        demo_cat = list(df[c].unique())
        demo_cat.sort()
        for d in demo_cat:
            demo_summary_df.loc[c + ': ' + str(d), :] = [
                int(np.round((df[c] == d).sum())),
                int(df.loc[df[c] == d, 'study_id'].nunique()),
                int(np.round((df[c] == d).mean() * 100))
            ]

    return demo_summary_df
