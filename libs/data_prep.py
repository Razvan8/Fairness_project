from imported_libraries import *

###Store data###
def store_data(X_train_with_A, X_val_with_A, X_test_with_A, y_train, y_val, y_test, age=None, gender=None, education=None,dataset_name='', sufix_name=''):
    ''' Function that takes as input dataframes for predictors, values to be predicted, sensitive features, and stores them in csv file
        Storing them allows someone else to get fairness related results without loosing time on running the preprocessing steps.

    '''
    if education != None :
        assert len(education) == 3, "There should be a list of train, val test data"
        education_train=education[0]
        education_val=education[1]
        education_test=education[2]
    if age != None :
        assert len(age) == 3, "There should be a list of train, val test data"
        age_train=age[0]
        age_val=age[1]
        age_test=age[2]
    if gender != None :
        assert len(gender) == 3, "There should be a list of train, val test data"
        gender_train=gender[0]
        gender_val=gender[1]
        gender_test=gender[2]



    # Define the directory path for saving dataframes
    dataframes_directory = os.path.join('..', 'Dataframes', dataset_name)

    # Create the 'dataframes' directory if it doesn't exist
    if not os.path.exists(dataframes_directory):
        os.makedirs(dataframes_directory)

    # Save the modified dataframes to CSV files
    X_train_with_A.to_csv(os.path.join(dataframes_directory, f'X_train_with_A{sufix_name}.csv'), index=False)
    X_val_with_A.to_csv(os.path.join(dataframes_directory, f'X_val_with_A{sufix_name}.csv'), index=False)
    X_test_with_A.to_csv(os.path.join(dataframes_directory, f'X_test_with_A{sufix_name}.csv'), index=False)

    y_train.to_csv(os.path.join(dataframes_directory, f'y_train{sufix_name}.csv'), index=False)
    y_val.to_csv(os.path.join(dataframes_directory, f'y_val{sufix_name}.csv'), index=False)
    y_test.to_csv(os.path.join(dataframes_directory, f'y_test{sufix_name}.csv'), index=False)


    # Save the gender and age and education Series to CSV files
    if gender!= None:
        gender_train.to_csv(os.path.join(dataframes_directory, f'gender_train{sufix_name}.csv'), index=False)
        gender_val.to_csv(os.path.join(dataframes_directory, f'gender_val{sufix_name}.csv'), index=False)
        gender_test.to_csv(os.path.join(dataframes_directory, f'gender_test{sufix_name}.csv'), index=False)

    if age != None:
        age_train.to_csv(os.path.join(dataframes_directory, f'age_train{sufix_name}.csv'), index=False)
        age_val.to_csv(os.path.join(dataframes_directory, f'age_val{sufix_name}.csv'), index=False)
        age_test.to_csv(os.path.join(dataframes_directory, f'age_test{sufix_name}.csv'), index=False)

    if education != None:
        education_train.to_csv(os.path.join(dataframes_directory, f'education_train{sufix_name}.csv'), index=False)
        education_val.to_csv(os.path.join(dataframes_directory, f'education_val{sufix_name}.csv'), index=False)
        education_test.to_csv(os.path.join(dataframes_directory, f'education_test{sufix_name}.csv'), index=False)




    print("Dataframes saved in their directory from 'Dataframes' directory.")


def load_stored_data( age=None, gender=None, education=None,dataset_name='', scale=True, sufix_name=''):
    '''
    Function that loads the stored data
    Directly loading them allows someone else to get fairness related results without loosing time on running the preprocessing steps
    '''
    # Define the directory path for the saved dataframes
    dataframes_directory = os.path.join('..', 'Dataframes', dataset_name)

    # Load the dataframes from the CSV files
    X_train_with_A = pd.read_csv(os.path.join(dataframes_directory, f'X_train_with_A{sufix_name}.csv'))
    X_val_with_A = pd.read_csv(os.path.join(dataframes_directory, f'X_val_with_A{sufix_name}.csv'))
    X_test_with_A = pd.read_csv(os.path.join(dataframes_directory, f'X_test_with_A{sufix_name}.csv'))

    age_train,age_val,age_test,gender_train, gender_val, gender_test, ed_train, ed_val,ed_test = None, None, None, None, None, None, None, None, None #initialize variables

    if gender != None:
        gender_train = pd.read_csv(os.path.join(dataframes_directory, f'gender_train{sufix_name}.csv')).values.reshape(-1)
        gender_val = pd.read_csv(os.path.join(dataframes_directory, f'gender_val{sufix_name}.csv')).values.reshape(-1)
        gender_test = pd.read_csv(os.path.join(dataframes_directory, f'gender_test{sufix_name}.csv')).values.reshape(-1)
    if age != None:
        age_train = pd.read_csv(os.path.join(dataframes_directory, f'age_train{sufix_name}.csv')).values.reshape(-1)
        age_val = pd.read_csv(os.path.join(dataframes_directory, f'age_val{sufix_name}.csv')).values.reshape(-1)
        age_test = pd.read_csv(os.path.join(dataframes_directory, f'age_test{sufix_name}.csv')).values.reshape(-1)

    if education != None:
        ed_train = pd.read_csv(os.path.join(dataframes_directory, f'education_train{sufix_name}.csv')).values.reshape(-1)
        ed_val = pd.read_csv(os.path.join(dataframes_directory, f'education_val{sufix_name}.csv')).values.reshape(-1)
        ed_test = pd.read_csv(os.path.join(dataframes_directory, f'education_test{sufix_name}.csv')).values.reshape(-1)

    if scale == True:
        X_train_with_A, X_val_with_A, X_test_with_A= scale_dataframes(
            [X_train_with_A, X_val_with_A, X_test_with_A]) ###scale all dfs ##Take care to use MinMaxScaler




    # Load target variables (y_train, y_test, y_val)
    y_train = pd.read_csv(os.path.join(dataframes_directory, f'y_train{sufix_name}.csv'))
    y_val = pd.read_csv(os.path.join(dataframes_directory, f'y_val{sufix_name}.csv'))
    y_test = pd.read_csv(os.path.join(dataframes_directory, f'y_test{sufix_name}.csv'))


    return X_train_with_A, X_val_with_A, X_test_with_A, y_train, y_val, y_test, age_train,age_val,age_test,gender_train, gender_val, gender_test, ed_train, ed_val, ed_test




def load_German_dataset(verbose=False):
    '''
    Function that loads the initial data.
    Return X and y
    '''

    # fetch dataset
    statlog_german_credit_data = fetch_ucirepo(id=144)

    # data (as pandas dataframes)
    X = statlog_german_credit_data.data.features
    y = statlog_german_credit_data.data.targets
    assert X.shape[0]==1000, "Smth went wrong X should have 1000 rows"
    assert y.shape[0]==1000, 'Smth went wrong y should have 1000 rows'
    print("Data loaded successfully")

    #Store data such that reviewer can access it without internet connection
    # Define the folder path


    # Ensure the folder exists, create it if not
    os.makedirs(folder_path, exist_ok=True)

    if verbose == True:
        print(f"Variables : {statlog_german_credit_data.variables}")
        print("print X.head()")
        print(X.head(3))
        print()
        print('y.head()')
        print(y.head(3))
    return X, y

def load_German_dataset_offline(verbose=True):
    dataframes_directory = os.path.join('..', 'Dataframes', 'German_credit')
    X=pd.read_csv(os.path.join(dataframes_directory, 'X_German.csv'))
    y=pd.read_csv(os.path.join(dataframes_directory, 'y_German.csv'))
    assert X.shape[0] == 1000, "Smth went wrong X should have 1000 rows"
    assert y.shape[0] == 1000, 'Smth went wrong y should have 1000 rows'
    print("Data loaded sucessfully")
    return X,y




def replace_values_with_binary(df, column_name, values_list):
    '''Function that checks all values from a column. It replaces the values with 1 if they are in a specified list, else 0 '''

    # Check if the column exists in the dataframe
    assert column_name  in df.columns, "Column name is not correct. It should be in df.column"

    df[column_name] = df[column_name].apply(lambda x: 1 if x in values_list else 0)

    return df


## Here I checked the function functionality
# Sample DataFrame
data = {'A': [1, 2, 3, 4, 5],
        'B': ['apple', 'banana', 'cherry', 'apple', 'date']}
df = pd.DataFrame(data)
# Define a list of values to replace
values_list = ['apple', 'banana']
# Test the function
df = replace_values_with_binary(df, 'B', values_list)
# Check if the 'B' column has been modified as expected
assert df['B'].equals(pd.Series([1, 1, 0, 1, 0])), "Function replace_values_with_binary does not work properly"







def apply_function_to_column(df, column_name, test_function,new_name):
    """
    Apply the test function to the specified column in the DataFrame.
    Returns:
    pd.DataFrame: The modified DataFrame.
    """
    assert column_name in df.columns, "Column name is not correct. It should be in df.columns"

    df[new_name] = df[column_name].apply(test_function)

    return df


def scale_dataframes(list_of_dfs):
    '''
    Function that scales a list of dataframes with MinMaxScaler
    '''
    scaled_dfs = []  # List to store the scaled DataFrames

    for df in list_of_dfs:
        scaler = MinMaxScaler()   ###Scale the data to 0,1 as authors do in paper
        scaled_values = scaler.fit_transform(df.values)  # Fit the scaler and transform the values

        # Create a new DataFrame with the scaled values and the same columns and index
        scaled_df = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)

        scaled_dfs.append(scaled_df)  # Append the scaled DataFrame to the list

    return scaled_dfs



