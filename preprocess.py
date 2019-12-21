"""
Description: The fair logistic regression method requires feature column names in a json file
Here, column names are parsed and json file is modified with the addition of selected column names to be used.
"""
import sys, json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def keep_cols(df, to_keep):
    df = pd.DataFrame(df[to_keep])
    df['AOIC'] = ['a'+str(aioc) for aioc in df['AOIC']] # Int's to string to force one-hot encoding downstream
    return df

def parse_JUDGE(df):
    # Remove commas from judge feature
    df['JUDGE'] = [j.replace(',', '') for j in df['JUDGE']]
    return df

def parse_RACE(df):
    # Convert race here, to prevent one-hot encoding downstream
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
    #// Modify json options file for fairness-penalizer method.
    with open(options_file, 'r') as infile:
        data = json.load(infile)

    data['data_headers'] = header # hold features names to use and exclude target var
    data['num_of_folds'] = num_folds
    data['file'] = csv_filename

    with open(options_file, 'w') as outfile:
        json.dump(data, outfile)

def scale_minmax(df, col_name, mn, mx):
    df[col_name] = [(i-mn)/(mx-mn) for i in df[col_name]]
    return df

def preprocess(df, to_keep):
    df = keep_cols(df, to_keep)
    df = parse_JUDGE(df)
    df = parse_RACE(df)
    df = to_dummy(df)
    df = scale_minmax(df, 'AGE_AT_INCIDENT', 17, 100) # min and max ages
    df = scale_minmax(df, 'CLASS.INITIATIONS', 1, 4) # min and max classes
    return df

##### INPUTS ####

train_df = pd.read_csv(sys.argv[1])
test_df = pd.read_csv(sys.argv[2])


#### CALLS ####
to_keep = ['LAW_ENFORCEMENT_AGENCY', 'INCIDENT_CITY', 'CLASS.INITIATIONS', 'AOIC', 'AGE_AT_INCIDENT', 'GENDER', 'RACE', 'UPDATED_OFFENSE_CATEGORY', 'JUDGE', 'COURT_NAME', 'COURT_FACILITY', 'CHARGE_REDUCTION']

# Parse Train Data

train_ids = train_df['CASE_PARTICIPANT_ID']
train_df = preprocess(train_df, to_keep)
train_df['CASE_PARTICIPANT_ID'] = train_ids # Prevent IDs from being one-hot encoded
train_df.to_csv('train_processed.csv', index=False)

test_ids = test_df['CASE_PARTICIPANT_ID']
test_df = preprocess(test_df, to_keep)
test_df['CASE_PARTICIPANT_ID'] = test_ids # Prevent IDs from being one-hot encoded
test_df.to_csv('test_processed.csv', index=False)


# Write JSON file
header = ','.join(list(train_df.drop(['CHARGE_REDUCTION', 'CASE_PARTICIPANT_ID'], axis=1))) # Write features to use in json options file.
options_file = 'chicago.json'
folds = 5 # k-fold to use in options file for hyperparameter selection.
mod_options(options_file, header, folds, train_parsed_fname)
