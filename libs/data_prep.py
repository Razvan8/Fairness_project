###IMPORTS
import joblib
import os
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
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
from itertools import product
from scipy.stats import  spearmanr
from scipy.stats import norm
###


###Store data###
def store_data(X_train_with_A, X_train_without_A, X_val_with_A, X_val_without_A, X_test_with_A, X_test_without_A, y_train, y_val, y_test, age=None, gender=None, education=None,dataset_name='', sufix_name=''):

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
    X_train_without_A.to_csv(os.path.join(dataframes_directory, f'X_train_without_A{sufix_name}.csv'), index=False)

    X_test_with_A.to_csv(os.path.join(dataframes_directory, f'X_test_with_A{sufix_name}.csv'), index=False)
    X_test_without_A.to_csv(os.path.join(dataframes_directory, f'X_test_without_A{sufix_name}.csv'), index=False)

    X_val_with_A.to_csv(os.path.join(dataframes_directory, f'X_val_with_A{sufix_name}.csv'), index=False)
    X_val_without_A.to_csv(os.path.join(dataframes_directory, f'X_val_without_A{sufix_name}.csv'), index=False)

    y_train.to_csv(os.path.join(dataframes_directory, f'y_train{sufix_name}.csv'), index=False)
    y_test.to_csv(os.path.join(dataframes_directory, f'y_test{sufix_name}.csv'), index=False)
    y_val.to_csv(os.path.join(dataframes_directory, f'y_val{sufix_name}.csv'), index=False)

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


def load_stored_data( age=None, gender=None, education=None,dataset_name='', scale=True, without_A=True, sufix_name=''):
    # Define the directory path for the saved dataframes
    dataframes_directory = os.path.join('..', 'Dataframes', dataset_name)

    # Load the dataframes from the CSV files
    X_train_with_A = pd.read_csv(os.path.join(dataframes_directory, f'X_train_with_A{sufix_name}.csv'))
    if without_A== True:
        X_train_without_A = pd.read_csv(os.path.join(dataframes_directory, f'X_train_without_A{sufix_name}.csv'))

    X_test_with_A = pd.read_csv(os.path.join(dataframes_directory, f'X_test_with_A{sufix_name}.csv'))
    if without_A ==True:
        X_test_without_A = pd.read_csv(os.path.join(dataframes_directory, f'X_test_without_A{sufix_name}.csv'))

    X_val_with_A = pd.read_csv(os.path.join(dataframes_directory, f'X_val_with_A{sufix_name}.csv'))
    if without_A ==True:
        X_val_without_A = pd.read_csv(os.path.join(dataframes_directory, f'X_val_without_A{sufix_name}.csv'))
    age_train,age_val,age_test,gender_train, gender_val, gender_test, ed_train, ed_val,ed_test = None, None, None, None, None, None, None, None, None


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

    if scale == True: ############################## TAKE CARE NOT TO SCALE SENS ATTRIBUTES THAT ARE CONSIDERED CLASSES############################
        X_train_with_A, X_val_with_A, X_test_with_A= scale_dataframes(
            [X_train_with_A, X_val_with_A, X_test_with_A]) ###scale all dfs ##Take care scale keeps 0,1 true
        if without_A == True:
            X_test_without_A, X_val_without_A, X_test_without_A=scale_dataframes([X_test_without_A, X_val_without_A,X_test_without_A])




    #age_train_val = np.concatenate((age_train, age_val), axis=0)
    #gender_train_val = np.concatenate((gender_train, gender_val), axis=0)

    # Load target variables (y_train, y_test, y_val)
    y_train = pd.read_csv(os.path.join(dataframes_directory, f'y_train{sufix_name}.csv'))
    y_test = pd.read_csv(os.path.join(dataframes_directory, f'y_test{sufix_name}.csv'))
    y_val = pd.read_csv(os.path.join(dataframes_directory, f'y_val{sufix_name}.csv'))

    return X_train_with_A, X_val_with_A, X_test_with_A, y_train, y_val, y_test, age_train,age_val,age_test,gender_train, gender_val, gender_test, ed_train, ed_val, ed_test




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


import numpy as np

def merge_two_sets(X_train, X_val, y_train, y_val):
    """
    Merge the training and validation sets.

    Parameters:
    - X_train, X_val: Training and validation features
    - y_train, y_val: Training and validation labels

    Returns:
    - X_train_val, y_train_val: Merged training and validation sets
    """
    X_train_val = np.concatenate((X_train, X_val), axis=0)
    y_train_val = np.concatenate((y_train, y_val), axis=0)

    return X_train_val, y_train_val

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




from itertools import product

def find_best_model(model, param_grid, X_train, y_train, X_val, y_val):
    """
    Find the best model based on performance on the validation set.

    Parameters:
    - model: The machine learning model (e.g., RandomForestClassifier, SVC, etc.)
    - param_grid: Dictionary with hyperparameter names as keys and lists of hyperparameter values to try
    - X_train, y_train: Training data and labels
    - X_val, y_val: Validation data and labels

    Returns:
    - best_model: The best model trained on the entire training set with the best hyperparameters
    """
    best_score = 0
    best_params = None
    best_model = None

    # Generate all combinations of hyperparameters
    param_combinations = list(product(*param_grid.values()))

    for params in param_combinations:
        # Create a dictionary of hyperparameters
        param_dict = dict(zip(param_grid.keys(), params))

        # Set the hyperparameters
        model.set_params(**param_dict)

        # Train the model on the entire training set
        model.fit(X_train, y_train)

        # Evaluate the model on the validation set
        val_score = model.score(X_val, y_val)

        # Update the best model if the current model has a higher validation score
        if val_score > best_score:
            best_score = val_score
            best_params = param_dict
            best_model = model

    print("Best Model Hyperparameters:", best_params)
    print("Validation Accuracy:", best_score)

    return best_model



def find_best_model_old(model, param_grid, X_train, y_train,verbosee=True):

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    grid_search_accuracy = grid_search.best_score_
    if verbosee== True:
        print(f"Accuracy for best grid search {model} is : {grid_search_accuracy}")
    return best_model, best_params



def eq_op_dif(y_true,y_predicted, sensitive_attribute, no_abs = False):
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
    cm_privileged = confusion_matrix(y_true[sensitive_attribute == 1], y_predicted[sensitive_attribute == 1]) ##confusion matrix class 1
    cm_unprivileged = confusion_matrix(y_true[sensitive_attribute == 0], y_predicted[sensitive_attribute == 0]) ## confusion matrix class 0

    # Calculate True Positive Rates (TPR)
    TPR_privileged = cm_privileged[1, 1] / (cm_privileged[1, 0] + cm_privileged[1, 1]) if cm_privileged[1, 1] + cm_privileged[1, 0] > 0 else 0
    TPR_unprivileged = cm_unprivileged[1, 1] / (cm_unprivileged[1, 0] + cm_unprivileged[1, 1]) if cm_unprivileged[1, 1] + cm_unprivileged[1, 0] > 0 else 0

    # Calculate Equal Opportunity score
    equal_opportunity_score = abs(TPR_privileged - TPR_unprivileged)
    if no_abs == True :
        equal_opportunity_score = TPR_privileged - TPR_unprivileged


    return equal_opportunity_score


###Scale the data to 0,1 as in paper

from sklearn.preprocessing import MinMaxScaler


def scale_dataframes(list_of_dfs):
    scaled_dfs = []  # List to store the scaled DataFrames

    for df in list_of_dfs:
        scaler = MinMaxScaler()  # Create a MinMaxScaler object
        scaled_values = scaler.fit_transform(df.values)  # Fit the scaler and transform the values

        # Create a new DataFrame with the scaled values and the same columns and index
        scaled_df = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)

        scaled_dfs.append(scaled_df)  # Append the scaled DataFrame to the list

    return scaled_dfs



####### Use fairness optimizer #########


def fairness_optimizer_results(threshold_optimizer, X_fit, y_fit, X_obs, y_obs, y_train, y_val, sensitive_1_fit,
                           sensitive_2_fit,
                           sensitive_1_obs, sensitive_2_obs, name_1, name_2, fitted=False, name_dataset1="train",
                           name_dataset2="val"):
    '''treshold optimiizer should have prefit= True
     sensitive_1 = feature we do optimiziation w.r.t
      sensitive_2 = feature that we see how was the fairness affected
      y_train, y_val = true y
      X_fit, y_fit= like train for fit
      X_obs, y_obs = like validation for optimizer
      y_fit, y_obs= predicted y's before optimizer '''

    if fitted == False:
        threshold_optimizer.predict_method = 'auto'
        threshold_optimizer.fit(X_fit, y_train, sensitive_features=sensitive_1_fit)

    adjusted_sensitive_train = threshold_optimizer.predict(X_fit, sensitive_features=sensitive_1_fit)
    adjusted_sensitive_val = threshold_optimizer.predict(X_obs, sensitive_features=sensitive_1_obs)

    #print(f"--------- SCORES AFTER OPTIMIZING FOR {name_1} ---------")
    #print()
    #print("----- accuracy scores -----")

    #print(
        #f' acc score {name_dataset1} got from : {accuracy_score(y_fit, y_train)} to {accuracy_score(adjusted_sensitive_train, y_train)}')
    #print(
        #f" acc score {name_dataset2} from : {accuracy_score(y_obs, y_val)} to {accuracy_score(adjusted_sensitive_val, y_val)}")

    #print()
    #print("----- Scores for fariness -----")

    #print(
        #f'{name_1} {name_dataset1} eq op went from: {eq_op_dif(y_train, y_fit, sensitive_attribute=sensitive_1_fit)} to {eq_op_dif(y_train, adjusted_sensitive_train, sensitive_attribute=sensitive_1_fit)}')
    #print(
        #f"{name_1} {name_dataset2} eq op went from :  {eq_op_dif(y_val, y_obs, sensitive_attribute=sensitive_1_obs)} to {eq_op_dif(y_val, adjusted_sensitive_val, sensitive_attribute=sensitive_1_obs)}")

    #print(
        #f'{name_2} {name_dataset1} eq op went from: {eq_op_dif(y_train, y_fit, sensitive_attribute=sensitive_2_fit, no_abs=True)} to {eq_op_dif(y_train, adjusted_sensitive_train, sensitive_attribute=sensitive_2_fit, no_abs=True)}')
    #print(
        #f"{name_2} {name_dataset2} eq op went from :  {eq_op_dif(y_val, y_obs, sensitive_attribute=sensitive_2_obs)} to {eq_op_dif(y_val, adjusted_sensitive_val, sensitive_attribute=sensitive_2_obs)}")


    initial_acc_train, initial_acc_test, after_acc_train,after_acc_test = accuracy_score(y_fit, y_train), accuracy_score(y_obs, y_val), accuracy_score(adjusted_sensitive_train, y_train), accuracy_score(adjusted_sensitive_val, y_val)
    initial_fair_train_1, initial_fair_test_1, after_fair_train_1, after_fair_test_1 = eq_op_dif(y_train, y_fit, sensitive_attribute=sensitive_1_fit), eq_op_dif(y_val, y_obs, sensitive_attribute=sensitive_1_obs), eq_op_dif(y_train, adjusted_sensitive_train, sensitive_attribute=sensitive_1_fit), eq_op_dif(y_val, adjusted_sensitive_val, sensitive_attribute=sensitive_1_obs)

    initial_fair_train_2, initial_fair_test_2, after_fair_train_2, after_fair_test_2 = eq_op_dif(y_train, y_fit,
                                                                                                 sensitive_attribute=sensitive_2_fit), eq_op_dif(
        y_val, y_obs, sensitive_attribute=sensitive_2_obs), eq_op_dif(y_train, adjusted_sensitive_train,
                                                                      sensitive_attribute=sensitive_2_fit), eq_op_dif(
        y_val, adjusted_sensitive_val, sensitive_attribute=sensitive_2_obs)
    return initial_acc_train, initial_acc_test, after_acc_train,after_acc_test, initial_fair_train_1, initial_fair_test_1, after_fair_train_1, after_fair_test_1, initial_fair_train_2, initial_fair_test_2, after_fair_train_2, after_fair_test_2



def use_fairness_optimizer(threshold_optimizer,X_fit, y_fit, X_obs, y_obs, y_train,y_val, sensitive_1_fit, sensitive_2_fit,
                           sensitive_1_obs, sensitive_2_obs, name_1, name_2, fitted=False, name_dataset1 = "train", name_dataset2="val"):

    '''treshold optimiizer should have prefit= True
     sensitive_1 = feature we do optimiziation w.r.t
      sensitive_2 = feature that we see how was the fairness affected
      y_train, y_val = true y
      X_fit, y_fit= like train for fit
      X_obs, y_obs = like validation for optimizer
      y_fit, y_obs= predicted y's before optimizer '''
    
    if fitted == False:
        threshold_optimizer.predict_method = 'auto'
        threshold_optimizer.fit(X_fit, y_train, sensitive_features=sensitive_1_fit)

    adjusted_sensitive_train = threshold_optimizer.predict(X_fit, sensitive_features=sensitive_1_fit)
    adjusted_sensitive_val = threshold_optimizer.predict(X_obs, sensitive_features=sensitive_1_obs)

    print(f"--------- SCORES AFTER OPTIMIZING FOR {name_1} ---------")
    print()
    print("----- accuracy scores -----")

    print(
        f' acc score {name_dataset1} got from : {accuracy_score(y_fit, y_train)} to {accuracy_score(adjusted_sensitive_train, y_train)}')
    print(
        f" acc score {name_dataset2} from : {accuracy_score(y_obs, y_val)} to {accuracy_score(adjusted_sensitive_val, y_val)}")

    print()
    print("----- Scores for fariness -----")

    print(
        f'{name_1} {name_dataset1} eq op went from: {eq_op_dif(y_train, y_fit, sensitive_attribute=sensitive_1_fit)} to {eq_op_dif(y_train, adjusted_sensitive_train, sensitive_attribute=sensitive_1_fit)}')
    print(
        f"{name_1} {name_dataset2} eq op went from :  {eq_op_dif(y_val, y_obs, sensitive_attribute=sensitive_1_obs)} to {eq_op_dif(y_val, adjusted_sensitive_val, sensitive_attribute=sensitive_1_obs)}")

    print(
        f'{name_2} {name_dataset1} eq op went from: {eq_op_dif(y_train, y_fit, sensitive_attribute = sensitive_2_fit, no_abs=True)} to {eq_op_dif(y_train, adjusted_sensitive_train, sensitive_attribute=sensitive_2_fit, no_abs=True)}')
    print(
        f"{name_2} {name_dataset2} eq op went from :  {eq_op_dif(y_val, y_obs, sensitive_attribute= sensitive_2_obs)} to {eq_op_dif(y_val, adjusted_sensitive_val, sensitive_attribute=sensitive_2_obs)}")


from sklearn.metrics import accuracy_score

def find_best_threshold( y, y_pred_proba, verbose= True):
    y_scores = y_pred_proba[:, 1]  # Assuming the probability for class 1 is at index 1

    thresholds = np.arange(0, 1, 0.05)  # Threshold values to test
    accuracies = []

    for threshold in thresholds:
        y_pred = (y_scores > threshold).astype(int)
        accuracy = accuracy_score(y, y_pred)
        accuracies.append(accuracy)

    best_threshold = thresholds[np.argmax(accuracies)]

    if verbose == True:
        plt.plot(thresholds, accuracies, marker='o')
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy at Different Thresholds')
        plt.show()
        print(f"Best treshold is {best_threshold} and best score is {np.max(accuracies)}")

    return best_threshold





################################## functions to add bias ###################
import pandas as pd
import numpy as np


def add_bias(X, y, unprivileged_class_name, unprivileged_class_value, p, verbose=True):
    ''' X = predictors dataframe
        y = values to predict
        unprivileged_class_name = str of unprivileged class
        unprivileged class values = int value for unprivileged (e.g. 0 or 1)
        p=probability to keep '''
    # Identify samples with the unprivileged class
    unprivileged_samples = X[X[unprivileged_class_name] == unprivileged_class_value]  # samples to drop form
    y_indexes = y[y["class"] == 1]
    unprivileged_samples_drop = unprivileged_samples[unprivileged_samples.index.isin(y_indexes.index)]
    if verbose == True:
        print(
            f"Initial positive percentage of samples from unprivileged class is {unprivileged_samples_drop.shape[0] / X.shape[0] * 100} % .")

    # Generate a mask indicating whether each row should be dropped based on the probability p
    drop_mask = np.random.rand(unprivileged_samples_drop.shape[0]) > p

    # Extract the indices of the rows to drop
    samples_to_drop = unprivileged_samples_drop.index[drop_mask].tolist()
    assert len(samples_to_drop) == drop_mask.sum(), "There is an error in process data function"

    # Drop rows from both DataFrames based on the generated mask
    X = X.drop(samples_to_drop)
    y = y.drop(samples_to_drop)
    assert X.shape[0] == y.shape[0], "X and y should have the same number of samples"

    if verbose == True:
        print(
            f"Current percentage of samples from positive unprivileged class is {(len(drop_mask) - len(samples_to_drop)) / X.shape[0] * 100}% . ")
        print(
            f"No of positive samples of unprivileged class kept is {len(drop_mask) - len(samples_to_drop)}, this means {(len(drop_mask) - len(samples_to_drop)) / len(drop_mask) * 100} % . It should be close to {p * len(drop_mask)}")

    return X, y




def deviation_for_CI (size1, size2, p1, p2, alpha=0.05):
    ''' Returns the std det vor difference'''
    assert 0 <= p1 and p1<=1, 'p1 must be a probability (TPR) '
    assert 0 <= p2 and p2<=1, 'p1 must be a probability (TPR) '
    z_critical = norm.ppf(1 - alpha / 2)
    s1=np.sqrt(p1*(1-p1)/size1) ##Since we have mean of Bernoulis
    s2=np.sqrt(size2*p2*(1-p2)/size2)
    sp=np.sqrt(   ((size1-1)*s1**2 + (size2-1)*s2**2)/  (size1+size2-2)  )
    dev= z_critical * np.sqrt(sp**2/size1 +sp**2/size2)
    return dev



import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_values(x_values, y_values, plot_name, x_axis_name, y_axis_name):
    """
    Plot a line chart given lists of x and y values.

    Parameters:
    - x_values (list): List of x-axis values.
    - y_values (list): List of y-axis values.
    - plot_name (str): Name of the plot.
    - x_axis_name (str): Label for the x-axis.
    - y_axis_name (str): Label for the y-axis.

    Returns:
    - None (displays the plot).
    """
    # Plot the values
    plt.plot(x_values, y_values, marker='o', linestyle='-')

    # Add labels and title
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.title(plot_name)

    # Show the plot
    plt.show()


import matplotlib.pyplot as plt

def plot_values_with_legend(x_values, y_values1, y_values2, plot_name, x_axis_name, y_axis_name):
    """
    Plot two lines with different colors and add a legend.

    Parameters:
    - x_values (list): List of x-axis values.
    - y_values1 (list): List of y-axis values for the first line.
    - y_values2 (list): List of y-axis values for the second line.
    - plot_name (str): Name of the plot.
    - x_axis_name (str): Label for the x-axis.
    - y_axis_name (str): Label for the y-axis.

    Returns:
    - None (displays the plot).
    """
    # Plot the first line with red color and label "before optimization"
    plt.plot(x_values, y_values1, color='red', linestyle='-', marker='o', label='before optimization')

    # Plot the second line with blue color and label "after optimization"
    plt.plot(x_values, y_values2, color='blue', linestyle='-', marker='o', label='after optimization')

    # Add labels, title, and legend
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.title(plot_name)
    plt.legend()

    # Show the plot
    plt.show()


def find_trend(data_fairness_age, data_fairness_gender, data_accuracy, correlation_test, ls_p_vary, vary_age=True):
    '''
    data_fairness_age, gender, acc should be from the same set (i.e. all training/test)
    first position in tuple is for age, second for gender
    '''
    if vary_age == True:
        p_fixed = 1  # Fix no drop in positive samples w.r.t gender
        vals = []
        for p_vary in ls_p_vary:
            vals.append(data_fairness_age[(p_vary, p_fixed)])  ## Fairness age

        plt.plot(ls_p_vary, vals)
        plt.ylabel("Eq op dif")
        plt.xlabel("probability of keeping positive samples w.r.t age")
        plt.title("Correlation analysis")
        plt.show()
        print(f" Age correlation: {correlation_test(ls_p_vary, vals)}")

        p_fixed = 1  # Fix no drop in positive samples w.r.t age  (keep it fixed)
        vals = []
        for p_vary in ls_p_vary:
            vals.append(data_fairness_gender[(p_vary, p_fixed)])  ## Fairness gender

        plt.plot(ls_p_vary, vals)
        plt.ylabel("Eq op dif")
        plt.xlabel("probability of keeping positive samples w.r.t age")
        plt.title("Correlation analysis")
        plt.show()
        print(f" Gender correlation: {spearmanr(ls_p_vary, vals)}") #gender

        p2 = 1  # Fix no drop in positive samples w.r.t age  (keep it fixed)
        vals = []
        for p_vary in ls_p_vary:
            vals.append(data_accuracy[(p_vary, p_fixed)])  ## Fairness age

        plt.plot(ls_p_vary, vals)
        plt.ylabel("Accuracy")
        plt.xlabel("probability of keeping positive samples w.r.t age")
        plt.title("Correlation analysis")
        plt.show()
        print(f"Accuracy correlation: {spearmanr(ls_p_vary, vals)}" )
        print()
        print()
        

    else:  # vary gender

        p_fixed = 1  # Fix no drop in positive samples w.r.t gender
        vals = []
        for p_vary in ls_p_vary:
            vals.append(data_fairness_age[(p_fixed, p_vary)])  ## Fairness age

        plt.plot(ls_p_vary, vals)
        plt.ylabel("Eq op dif")
        plt.xlabel("probability of keeping positive samples w.r.t gender")
        plt.title("Correlation analysis")
        plt.show()
        print(f"Age correlation: {correlation_test(ls_p_vary, vals)}")

        p_fixed = 1  # Fix no drop in positive samples w.r.t age  (keep it fixed)
        vals = []
        for p_vary in ls_p_vary:
            vals.append(data_fairness_gender[(p_fixed, p_vary)])  ## Fairness gender

        plt.plot(ls_p_vary, vals)
        plt.ylabel("Eq op dif")
        plt.xlabel("probability of keeping positive samples w.r.t gender")
        plt.title("Correlation analysis")
        plt.show()
        print(f" Gender correlation: {correlation_test(ls_p_vary, vals)}")

        p_fixed = 1  # Fix no drop in positive samples w.r.t age  (keep it fixed)
        vals = []
        for p_vary in ls_p_vary:
            vals.append(data_accuracy[(p_fixed, p_vary)])  ## Accuracy

        plt.plot(ls_p_vary, vals)
        plt.ylabel("Accuracy")
        plt.xlabel("probability of keeping positive samples w.r.t gender")
        plt.title("Correlation analysis")
        plt.show()
        print(f"Accuracy correlation: {correlation_test(ls_p_vary, vals)}")
        print()
        print()


def find_trend_optimization(before_fairness_age, before_fairness_gender, before_accuracy, after_fairness_age, after_fairness_gender, after_accuracy, correlation_test, ls_p_vary, vary_age=True, std = True, size1=0, size2=0):
    '''
    data_fairness_age, gender, acc should be from the same set (i.e. all training/test)
    first position in tuple is for age, second for gender
    Take care! Optimization is done w.r.t what we vary. I.e, vary age fairness, optimize w.r.t age
    '''

    if std == True:
        assert size1 >0 and size2>0 , "Both sizes must be positive"
    if vary_age == True:
        print("Vary age and optimize w.r.t age")
        print()

        p_fixed = 1  # Fix no drop in positive samples w.r.t gender
        vals_age = []
        margin_of_error=[]
        for p_vary in ls_p_vary:
            vals_age.append(after_fairness_age[(p_vary, p_fixed)] - before_fairness_age[(p_vary, p_fixed)] )  ## Fairness age
            if std == True:
                margin_of_error.append(deviation_for_CI(size1=size1, size2=size2, p1=before_fairness_age[(p_vary, p_fixed)],p2=after_fairness_age[(p_vary, p_fixed)], alpha=0.05))

        plt.plot(ls_p_vary, vals_age)
        if std == True:
            plt.fill_between(ls_p_vary, np.array(vals_age) - np.array(margin_of_error), np.array(vals_age) + np.array(margin_of_error), alpha=0.2,label='CI')
        plt.ylabel("Eq op dif")
        plt.xlabel("probability of keeping positive samples w.r.t age")
        plt.title("Correlation analysis")
        plt.show()
        print(f" Correlation between probability of keeping positive samples w.r.t age and age fairness increase: {correlation_test(ls_p_vary, vals_age)}")



        p_fixed = 1  # Fix no drop in positive samples w.r.t age  (keep it fixed)
        vals_gender = []

        margin_of_error=[]

        for p_vary in ls_p_vary:
            vals_gender.append(after_fairness_gender[(p_vary, p_fixed)] - before_fairness_gender[(p_vary, p_fixed)])  ## Fairness gender
            if std ==True:
                margin_of_error.append( deviation_for_CI(size1=size1, size2=size2, p1= before_fairness_gender[(p_vary, p_fixed)], p2=after_fairness_gender[(p_vary, p_fixed)], alpha=0.05))


        plt.plot(ls_p_vary, vals_gender)
        if std == True:
            plt.fill_between(ls_p_vary, np.array(vals_gender) - np.array(margin_of_error), np.array(vals_gender) + np.array(margin_of_error), alpha=0.2,label='CI')

        plt.ylabel("Eq op dif")
        plt.xlabel("probability of keeping positive samples w.r.t age")
        plt.title("Correlation analysis")
        plt.show()
        print(f"  Correlation between probability of keeping positive samples w.r.t age and gender fairness increase: {correlation_test(ls_p_vary, vals_gender)}")  # gender



        p_fixed = 1  # Fix no drop in positive samples w.r.t age  (keep it fixed)
        vals_acc = []
        margin_of_error=[]
        for p_vary in ls_p_vary:
            vals_acc.append(after_accuracy[(p_vary, p_fixed)] - before_accuracy[(p_vary, p_fixed)])  ## Fairness age
            if std ==True:
                margin_of_error.append( deviation_for_CI(size1=size1, size2=size2, p1= before_accuracy[(p_vary, p_fixed)], p2=after_accuracy[(p_vary, p_fixed)], alpha=0.05))


        plt.plot(ls_p_vary, vals_acc)
        if std == True:
            plt.fill_between(ls_p_vary, np.array(vals_acc) - np.array(margin_of_error), np.array(vals_acc) + np.array(margin_of_error), alpha=0.2,label='CI')

        plt.ylabel("Accuracy")
        plt.xlabel("probability of keeping positive samples w.r.t age")
        plt.title("Correlation analysis")
        plt.show()
        print(f" Correlation between probability of keeping positive samples w.r.t age and accuracy increase: {correlation_test(ls_p_vary, vals_acc)}")
        print()

        print(f"-- RESULTS ABOUT CORRELATION BETWEEN FAIRNESS AND ACC --")
        print(f"Coorelation between age fairness improvement and gender fairness : {correlation_test(vals_age, vals_gender)}")
        print(f"Coorelation between age fairness improvement and accuracy : {correlation_test(vals_age, vals_acc)}") # we do only gender and acc bc we optimize w.r.t gender
        print()
        print()



    else:  # vary gender

        print("Vary gender and optimize w.r.t gender")
        print()

        p_fixed = 1  # Fix no drop in positive samples w.r.t gender
        vals_age = []
        margin_of_error=[]
        for p_vary in ls_p_vary:
            if std ==True:
                margin_of_error.append( deviation_for_CI(size1=size1, size2=size2, p1= before_fairness_age[(p_fixed, p_vary)], p2=after_fairness_age[(p_fixed, p_vary)], alpha=0.05))
            vals_age.append(after_fairness_age[(p_fixed, p_vary)] - before_fairness_age[(p_fixed, p_vary)])  ## Fairness age

        plt.plot(ls_p_vary, vals_age)
        if std == True:
            plt.fill_between(ls_p_vary, np.array(vals_age) - np.array(margin_of_error),np.array(vals_age) + np.array(margin_of_error), alpha=0.2, label='CI')
        plt.ylabel("Eq op dif")
        plt.xlabel("probability of keeping positive samples w.r.t gender")
        plt.title("Correlation analysis")
        plt.show()
        print(f" Correlation between probability of keeping positive samples w.r.t gender and age fairness increase: {correlation_test(ls_p_vary, vals_age)}")



        p_fixed = 1  # Fix no drop in positive samples w.r.t age  (keep it fixed)
        vals_gender = []
        margin_of_error=[]
        for p_vary in ls_p_vary:
            if std ==True:
                margin_of_error.append( deviation_for_CI(size1=size1, size2=size2, p1= before_fairness_gender[(p_fixed, p_vary)], p2=after_fairness_gender[(p_fixed, p_vary)], alpha=0.05))
            vals_gender.append(after_fairness_gender[(p_fixed, p_vary)] - before_fairness_gender[(p_fixed, p_vary)])  ## Fairness gender

        plt.plot(ls_p_vary, vals_gender)
        if std == True:
            plt.fill_between(ls_p_vary, np.array(vals_gender) - np.array(margin_of_error), np.array(vals_gender) + np.array(margin_of_error), alpha=0.2,label='CI')
        plt.ylabel("Eq op dif")
        plt.xlabel("probability of keeping positive samples w.r.t gender")
        plt.title("Correlation analysis")
        plt.show()
        print(f" Correlation between probability of keeping positive samples w.r.t gender and gender fairness increase: {correlation_test(ls_p_vary, vals_gender)}")




        p_fixed = 1  # Fix no drop in positive samples w.r.t age  (keep it fixed)
        vals_acc = []
        margin_of_error=[]
        for p_vary in ls_p_vary:
            if std ==True:
                margin_of_error.append( deviation_for_CI(size1=size1, size2=size2, p1= before_accuracy[(p_fixed, p_vary)], p2=after_accuracy[(p_fixed, p_vary)], alpha=0.05))
            vals_acc.append(after_accuracy[(p_fixed, p_vary)] - before_accuracy[(p_fixed, p_vary)])  ## Accuracy

        plt.plot(ls_p_vary, vals_acc)
        if std == True:
            plt.fill_between(ls_p_vary, np.array(vals_acc) - np.array(margin_of_error), np.array(vals_acc) + np.array(margin_of_error), alpha=0.2,label='CI')
        plt.ylabel("Accuracy")
        plt.xlabel("probability of keeping positive samples w.r.t gender")
        plt.title("Correlation analysis")
        plt.show()
        print(f"Correlation between probability of keeping positive samples w.r.t gender and accuracy increase:: {correlation_test(ls_p_vary, vals_acc)}")
        print()

        print(f"-- RESULTS ABOUT CORRELATION BETWEEN FAIRNESS AND ACC --")
        print(f"Coorelation between gender fairness improvement and age fairness : {correlation_test(vals_gender, vals_age)}")
        print(f"Coorelation between gender fairness improvement and accuracy : {correlation_test(vals_gender, vals_acc)}") # we do only gender and acc bc we optimize w.r.t gender
        print()
        print()






####Accuracy fairness correlation backup

def find_trend_optimization_backup(before_fairness_age, before_fairness_gender, before_accuracy, after_fairness_age, after_fairness_gender, after_accuracy, correlation_test, ls_p_vary, vary_age=True):
    '''
    data_fairness_age, gender, acc should be from the same set (i.e. all training/test)
    first position in tuple is for age, second for gender
    '''
    if vary_age == True:
        p_fixed = 1  # Fix no drop in positive samples w.r.t gender
        vals = []
        for p_vary in ls_p_vary:
            vals.append(after_fairness_age[(p_vary, p_fixed)] - before_fairness_age[(p_vary, p_fixed)] )  ## Fairness age

        plt.plot(ls_p_vary, vals)
        plt.ylabel("Eq op dif")
        plt.xlabel("probability of keeping positive samples w.r.t age")
        plt.title("Correlation analysis")
        plt.show()
        print(f" Age correlation: {correlation_test(ls_p_vary, vals)}")

        p_fixed = 1  # Fix no drop in positive samples w.r.t age  (keep it fixed)
        vals = []
        for p_vary in ls_p_vary:
            vals.append(after_fairness_gender[(p_vary, p_fixed)] - before_fairness_gender[(p_vary, p_fixed)])  ## Fairness gender

        plt.plot(ls_p_vary, vals)
        plt.ylabel("Eq op dif")
        plt.xlabel("probability of keeping positive samples w.r.t age")
        plt.title("Correlation analysis")
        plt.show()
        print(f" Gender correlation: {spearmanr(ls_p_vary, vals)}")  # gender

        p2 = 1  # Fix no drop in positive samples w.r.t age  (keep it fixed)
        vals = []
        for p_vary in ls_p_vary:
            vals.append(after_accuracy[(p_vary, p_fixed)] - before_accuracy[(p_vary, p_fixed)])  ## Fairness age

        plt.plot(ls_p_vary, vals)
        plt.ylabel("Accuracy")
        plt.xlabel("probability of keeping positive samples w.r.t age")
        plt.title("Correlation analysis")
        plt.show()
        print(f"Accuracy correlation: {spearmanr(ls_p_vary, vals)}")
        print()
        print()


    else:  # vary gender

        p_fixed = 1  # Fix no drop in positive samples w.r.t gender
        vals = []
        for p_vary in ls_p_vary:
            vals.append(after_fairness_age[(p_fixed, p_vary)] - before_fairness_age[(p_fixed, p_vary)])  ## Fairness age

        plt.plot(ls_p_vary, vals)
        plt.ylabel("Eq op dif")
        plt.xlabel("probability of keeping positive samples w.r.t gender")
        plt.title("Correlation analysis")
        plt.show()
        print(f"Age correlation: {correlation_test(ls_p_vary, vals)}")

        p_fixed = 1  # Fix no drop in positive samples w.r.t age  (keep it fixed)
        vals = []
        for p_vary in ls_p_vary:
            vals.append(after_fairness_gender[(p_fixed, p_vary)] - before_fairness_gender[(p_fixed, p_vary)])  ## Fairness gender

        plt.plot(ls_p_vary, vals)
        plt.ylabel("Eq op dif")
        plt.xlabel("probability of keeping positive samples w.r.t gender")
        plt.title("Correlation analysis")
        plt.show()
        print(f" Gender correlation: {correlation_test(ls_p_vary, vals)}")

        p_fixed = 1  # Fix no drop in positive samples w.r.t age  (keep it fixed)
        vals = []
        for p_vary in ls_p_vary:
            vals.append(after_accuracy[(p_fixed, p_vary)] - before_accuracy[(p_fixed, p_vary)])  ## Accuracy

        plt.plot(ls_p_vary, vals)
        plt.ylabel("Accuracy")
        plt.xlabel("probability of keeping positive samples w.r.t gender")
        plt.title("Correlation analysis")
        plt.show()
        print(f"Accuracy correlation: {correlation_test(ls_p_vary, vals)}")
        print()
        print()






