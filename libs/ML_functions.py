
###### ML functions#########################################################################################################
from imported_libraries import *
from data_prep import *

def find_best_model(model, param_grid, X_train, y_train, X_val, y_val, verbose= True):
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

    if verbose == True :
        print("Best Model Hyperparameters:", best_params)
        print("Validation Accuracy:", best_score)

    return best_model



def find_best_model_old(model, param_grid, X_train, y_train,verbosee=True): ### probabbly delete this

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    grid_search_accuracy = grid_search.best_score_
    if verbosee== True:
        print(f"Accuracy for best grid search {model} is : {grid_search_accuracy}")
    return best_model, best_params


from sklearn.metrics import accuracy_score

def find_best_threshold( y, y_pred_proba, verbose= True):
    '''Function that finds best threshold to convert probabilities in 0 or 1'''
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

def create_save_iterative_german_models(p_range1=[0.2,0.5,0.8], p_range2=[0.25], dataset_name="German_credit_biased",verbose=True):
    '''Function that stores the best ML model for every biased dataset created'''
    for p1 in p_range1:
        for p2 in p_range2:
            X_train_with_A, X_val_with_A, X_test_with_A, y_train, y_val, y_test, age_train, age_val, age_test,gender_train, \
                gender_val, gender_test, ed_train, ed_val, ed_test = load_stored_data(age=None, gender=None, education=None, dataset_name=dataset_name,
                                                                                    scale=True,sufix_name=f'_{p1}_{p2}')
            #########MODEL

            ############# find best model

            #param_grid_rf = {
                #'n_estimators': [10, 50, 100],
                #'max_depth': [5, 10],
                #'min_samples_leaf': [8, 16]
            #}

            # param_grid_svc = {
            # 'C': [0.1, 1, 10],
            # 'kernel': ['linear']
            # }

            #param_grid_knn = {
                #'n_neighbors': [3, 5, 10, 20]
            #}

            param_grid_lr = {
                'random_state' :[1], ###random state is fixed. I dont optimize over it but I wanted to fix it for reproducibility
                'C': [0.001, 0.01, 0.1, 1,10,20,50],
                'penalty': ['l2',
                            ],
                'max_iter' : [1000] ##dont optimize over this. Just make sure lr converges in general!!

            }

            #model_rf = RandomForestClassifier()

            #model_knn = KNeighborsClassifier()
            model_lr = LogisticRegression()  # The solver 'liblinear' is suitable for small datasets.


            best_lr_A = find_best_model(model_lr, param_grid_lr, X_train_with_A, y_train.values.ravel(), X_val_with_A,
                                        y_val.values.ravel())
            #best_rf_A = find_best_model(model_rf, param_grid_rf, X_train_with_A, y_train.values.ravel(), X_val_with_A,
                                        #y_val.values.ravel())
            #best_knn_A = find_best_model(model_knn, param_grid_knn, X_train_with_A, y_train.values.ravel(),
                                         #X_val_with_A, y_val.values.ravel())



            models_directory = os.path.join('..', 'ML_models', dataset_name)  # '..' moves one directory up
            if not os.path.exists(models_directory):
                os.makedirs(models_directory)

            # Save the best models to separate files
            #joblib.dump(best_rf_A, os.path.join(models_directory, 'best_random_forest_A_model.pkl'))
            #joblib.dump(best_knn_A, os.path.join(models_directory, 'best_knn_A_model.pkl'))
            joblib.dump(best_lr_A, os.path.join(models_directory, f'best_logistic_regression_A_model_{p1}_{p2}.pkl'))


    print("Best models saved in 'German_credit_biased' directory from 'ML_models' directory.")


