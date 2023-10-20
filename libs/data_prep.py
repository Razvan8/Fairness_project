###IMPORTS

from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

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
        print(f"Accuracy for best grid search {model} is : {grid_search_accuracy}")
    return best_model, best_params

from sklearn.metrics import confusion_matrix

def eq_op_dif(y_true,y_predicted, sensitive_attribute):
    """
    Compute Equal Opportunity fairness metric.

    Parameters:
    y_predicted (array-like): Predicted labels (0 or 1).
    y_true (array-like): True labels (0 or 1).
    sensitive_attribute (array-like): Binary sensitive attribute (0 or 1).

    Returns:
    float: Equal Opportunity score (0 to 1).
    """

    # Confusion matrices for different groups (privileged and unprivileged)
    cm_privileged = confusion_matrix(y_true[sensitive_attribute == 1], y_predicted[sensitive_attribute == 1])
    cm_unprivileged = confusion_matrix(y_true[sensitive_attribute == 0], y_predicted[sensitive_attribute == 0])

    # Calculate True Positive Rates (TPR)
    TPR_privileged = cm_privileged[1, 1] / (cm_privileged[1, 0] + cm_privileged[1, 1]) if cm_privileged[1, 1] + cm_privileged[1, 0] > 0 else 0
    TPR_unprivileged = cm_unprivileged[1, 1] / (cm_unprivileged[1, 0] + cm_unprivileged[1, 1]) if cm_unprivileged[1, 1] + cm_unprivileged[1, 0] > 0 else 0

    # Calculate Equal Opportunity score
    equal_opportunity_score = abs(TPR_privileged - TPR_unprivileged)

    return equal_opportunity_score



