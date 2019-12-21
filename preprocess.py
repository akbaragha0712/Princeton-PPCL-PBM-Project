"""
Description: The fair logistic regression method requires feature column names in a jason file
Here, column names are parsed and json file is modified with chosen columns.
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


##### INPUTS ####

train_df = pd.read_csv(sys.argv[1])
test_df = pd.read_csv(sys.argv[2])


test_case_ids = test_df['CASE_PARTICIPANT_ID']
train_case_ids = train_df['CASE_PARTICIPANT_ID']

train_df = train_df.drop('CASE_PARTICIPANT_ID', axis=1)
test_df = test_df.drop('CASE_PARTICIPANT_ID', axis=1)

folds = 5
train_parsed_fname = 'train_parsed.csv'
test_parsed_fname = 'test_parsed.csv'


#### CALLS ####
to_keep = ['LAW_ENFORCEMENT_AGENCY', 'INCIDENT_CITY', 'CLASS.INITIATIONS', 'AOIC', 'AGE_AT_INCIDENT', 'GENDER', 'RACE', 'UPDATED_OFFENSE_CATEGORY', 'JUDGE', 'COURT_NAME', 'COURT_FACILITY', 'CHARGE_REDUCTION']

# Parse Train Data
train_df = keep_cols(train_df, to_keep)
print(train_df)
train_df = parse_JUDGE(train_df)
train_df = parse_RACE(train_df)
train_df = to_dummy(train_df)
train_df = scale_minmax(train_df, 'AGE_AT_INCIDENT', 17, 100) # min and max ages
train_df = scale_minmax(train_df, 'CLASS.INITIATIONS', 1, 4) # min and max classes


train_df['CASE_PARTICIPANT_ID'] = participant_ids
#train_out = train_df[-train_df['CASE_PARTICIPANT_ID'].isin(test_case_ids)]
train_out.to_csv(train_parsed_fname.replace('.csv', '_IDs.csv'), index=False)
train_out= train_out.drop('CASE_PARTICIPANT_ID', axis=1)
train_out.to_csv(train_parsed_fname, index=False)

test_out = train_df[train_df['CASE_PARTICIPANT_ID'].isin(test_case_ids)]
test_out1= test_out.drop('CASE_PARTICIPANT_ID', axis=1)
test_out1.to_csv(test_parsed_fname, index=False)
test_out.to_csv(test_parsed_fname.replace('.csv', '_IDs.csv'), index=False)


# Write JSON file
header = ','.join(list(train_out.drop('CHARGE_REDUCTION', axis=1)))
options_file = 'chicago.json'
mod_options(options_file, header, folds, train_parsed_fname)
