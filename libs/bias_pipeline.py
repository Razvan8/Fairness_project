from data_prep import *
import joblib
import os


def create_iterative_german_bias (X, y,  unprivileged_class_name1="Age_group", unprivileged_class_name2= "Attribute9", unprivileged_class_value1=0, unprivileged_class_value2=0,
                            p_range1=[0.2,0.5,0.8], p_range2=[0.2,0.5,0.8], verbose=True, dataset_name =  'German_credit_biased'):
    
    for p1 in p_range1:
        for p2 in p_range2:
            Xc, yc = add_bias(X=X, unprivileged_class_name=unprivileged_class_name1, unprivileged_class_value= unprivileged_class_value1, y=y, p=p1,
                            verbose=True)  # Age bias
            Xc, yc = add_bias(X=Xc, unprivileged_class_name = unprivileged_class_name2, unprivileged_class_value = unprivileged_class_value2, y=yc, p=p2,
                            verbose=True)  # Gender bias

            num_features = ["Attribute2", "Attribute5", "Attribute8", "Attribute11", "Attribute13", "Attribute16",
                            "Attribute18"]
            cat_features = [col_name for col_name in X.columns if col_name not in num_features]
            Xc = pd.get_dummies(Xc, columns=cat_features, drop_first=True)

            X_train, X_test, y_train, y_test = train_test_split(Xc, yc, test_size=0.4, random_state=123)

            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)  ##this make 0.2 for both val and test

            ## Save sensitive attributes

            gender_train = X_train["Attribute9_1"]
            age_train = X_train["Age_group_1"]

            gender_test = X_test["Attribute9_1"]
            age_test = X_test["Age_group_1"]

            gender_val = X_val["Attribute9_1"]
            age_val = X_val["Age_group_1"]

            X_train_with_A = X_train.copy()  # X with sensitive_attributes
            X_train_with_A.drop("Age_group_1", axis=1, inplace=True)
            X_train_without_A = X_train.drop(["Age_group_1", "Attribute9_1"], axis=1)

            X_test_with_A = X_test.copy()  # X with sensitive_attributes
            X_test_with_A.drop("Age_group_1", axis=1, inplace=True)
            X_test_without_A = X_test.drop(["Age_group_1", "Attribute9_1"], axis=1)

            X_val_with_A = X_val.copy()  # X with sensitive_attributes
            X_val_with_A.drop("Age_group_1", axis=1, inplace=True)
            X_val_without_A = X_val.drop(["Age_group_1", "Attribute9_1"], axis=1)

            store_data(dataset_name=dataset_name, X_train_with_A=X_train_with_A,
                       X_train_without_A=X_train_without_A, X_val_with_A=X_val_with_A,
                       X_val_without_A=X_val_without_A, X_test_with_A=X_test_with_A, X_test_without_A=X_test_without_A,
                       y_train=y_train,
                       y_val=y_val, y_test=y_test, age=[age_train, age_val, age_test],
                       gender=[gender_train, gender_val, gender_test], education=None,sufix_name=f"_{p1}_{p2}")





def create_save_iterative_german_models(p_range1=[0.2,0.5,0.8], p_range2=[0.2,0.5,0.8], dataset_name="German_credit_biased",verbose=True):
    for p1 in p_range1:
        for p2 in p_range2:
            X_train_with_A, X_val_with_A, X_test_with_A, y_train, y_val, y_test, age_train, age_val, age_test,gender_train, \
                gender_val, gender_test, ed_train, ed_val, ed_test=load_stored_data(age=None, gender=None, education=None, dataset_name=dataset_name, scale=True, without_A=False,
                                                                                    sufix_name=f'_{p1}_{p2}')
            #########MODEL

            ############# find best model

            param_grid_rf = {
                'n_estimators': [10, 50, 100],
                'max_depth': [5, 10],
                'min_samples_leaf': [8, 16]
            }

            # param_grid_svc = {
            # 'C': [0.1, 1, 10],
            # 'kernel': ['linear']
            # }

            param_grid_knn = {
                'n_neighbors': [3, 5, 10, 20]
            }

            param_grid_lr = {
                'C': [0.001, 0.01, 0.1, 1],
                'penalty': ['l2',
                            ]
            }

            model_rf = RandomForestClassifier()

            model_knn = KNeighborsClassifier()
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

            X_train_val_with_A, y_train_val = merge_two_sets(X_train_with_A, X_val_with_A, y_train, y_val)

            # Create the 'models' directory if it doesn't exist
            if not os.path.exists(models_directory):
                os.makedirs(models_directory)

            #best_rf_A.fit(X_train_val_with_A, y_train_val.ravel())
            best_lr_A.fit(X_train_val_with_A, y_train_val.ravel())
            #best_knn_A.fit(X_train_val_with_A, y_train_val.ravel())

            # Save the best models to separate files
            #joblib.dump(best_rf_A, os.path.join(models_directory, 'best_random_forest_big_A_model.pkl'))
            #joblib.dump(best_knn_A, os.path.join(models_directory, 'best_knn_big_A_model.pkl'))
            joblib.dump(best_lr_A, os.path.join(models_directory, f'best_logistic_regression_big_A_model_{p1}_{p2}.pkl'))

    print("Best models saved in the 'models' directory.")


















            