"""
Description: Process data to train model unsing approach developed
by Yahav Bechavod et. al. (2017). Here, column names are parsed and json file
is modified with selected column names to be used.
"""
import os, sys, json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def parse_JUDGE(df):
    df['JUDGE'] = [j.replace(',', '') for j in df['JUDGE']]
    return df

def parse_RACE(df):
    """
    Convert race to int to prevent one-hot encoding downstream
    """
    race_encoded = []
    for race in df['RACE']:
        if race == 'Black':
            race_encoded.append(0)
        elif race == 'White':
            race_encoded.append(1)
    df['RACE'] = race_encoded
    return df

def to_dummy(df):
    df = pd.get_dummies(df) # Detect columns to one-hot encode.
    for character in [' ', '-', "'"]:
        df.columns = df.columns.str.replace(character, '') # Parse col names
    return df

def min_max_scale(df, cols_to_scale):
    # min-max scale specified columns.
    for col in cols_to_scale:
        min_val, max_val = min(df[col]), max(df[col])
        df[col] = [(i-min_val)/(max_val-min_val) for i in df[col]]
    return df

def mod_json(options_file, header, num_folds, csv_filename):
    """
    Modify json options file for fairness-penalizer method.
    """
    with open(options_file, 'r') as infile:
        data = json.load(infile)
    data['data_headers'] = header # Features names to use. Exclude target var
    data['num_of_folds'] = num_folds
    data['file'] = csv_filename
    with open(options_file, 'w') as outfile:
        json.dump(data, outfile)

def scale_minmax(df, col_name, mn, mx):
    df[col_name] = [(i-mn)/(mx-mn) for i in df[col_name]]
    return df

def preprocess(df, to_keep):
    df = pd.DataFrame(df[to_keep])
    df['AOIC'] = ['a'+str(aioc) for aioc in df['AOIC']] #Convert int to string to ensure one-hot encoding downstream.
    df = parse_JUDGE(df)
    df = parse_RACE(df)
    df = to_dummy(df)
    df = scale_minmax(df, 'AGE_AT_INCIDENT', 17, 100) # min and max ages
    df = scale_minmax(df, 'CLASS.INITIATIONS', 1, 4) # min and max classes
    return df

if __name__ == "__main__":
    ##### INPUTS ####
    train_df = pd.read_csv(sys.argv[1])
    test_df = pd.read_csv(sys.argv[2])

    train_ids = train_df['CASE_PARTICIPANT_ID']
    test_ids = test_df['CASE_PARTICIPANT_ID']

    full_df = pd.concat([train_df, test_df], axis=0)
    full_df = full_df.set_index('CASE_PARTICIPANT_ID')
    full_df = full_df.drop('CASE_PARTICIPANT_ID')

    #### CALLS ####
    to_keep = ['LAW_ENFORCEMENT_AGENCY', 'INCIDENT_CITY', 'CLASS.INITIATIONS', \
            'AOIC',  'AGE_AT_INCIDENT', 'GENDER', 'RACE', 'UPDATED_OFFENSE_CATEGORY', \
            'JUDGE', 'COURT_NAME', 'COURT_FACILITY', 'CHARGE_REDUCTION']

    # Parse Train Data
    full_df = preprocess(full_df, to_keep)
    full_df.head(n=train_df.shape[0]).to_csv('train_processed.csv', index=False)
    full_df.tail(n=test_df.shape[0]).to_csv('test_processed.csv', index=False)

    # Write features to use in json options file.
    header = ','.join(list(full_df.drop(['CHARGE_REDUCTION'], axis=1)))
    options_file = 'chicago.json'
    folds = 5 # k-fold to use in options file for hyperparameter selection.
    mod_json(options_file, header, folds, 'train_processed.csv')
