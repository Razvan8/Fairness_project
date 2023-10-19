###IMPORTS

from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt
import numpy as np

###
def load_data(verbose=False):
    '''
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

    if verbose == True:
        print(f"Variables : {statlog_german_credit_data.variables}")
        print("print X.head()")
        print(X.head(3))
        print()
        print('y.head()')
        print(y.head(3))
    return X, y


####
def replace_values_with_binary(df, column_name, values_list):
    # Check if the column exists in the dataframe
    assert column_name  in df.columns, "Column name is not correct. It should be in df.column"

    df[column_name] = df[column_name].apply(lambda x: 1 if x in values_list else 0)

    return df

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

# The 'B' column should contain [1, 1, 0, 1, 0] after applying the function

###

import pandas as pd

def apply_function_to_column(df, column_name, test_function,new_name):
    """
    Apply the test function to the specified column in the DataFrame.
    Parameters:
    df (pd.DataFrame): The DataFrame.
    column_name (str): The name of the column to modify.
    test_function (function): The function to apply to each value in the specified column.

    Returns:
    pd.DataFrame: The modified DataFrame.
    """
    assert column_name in df.columns, "Column name is not correct. It should be in df.columns"

    df[new_name] = df[column_name].apply(test_function)

    return df



def find_best_model(model, param_grid, X_train, y_train,verbosee=True):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    grid_search_accuracy = grid_search.best_score_
    if verbosee== True:
        print(f"Accuracy for best model is grid_search_accuracy{grid_search_accuracy}")
    return best_model, best_params


