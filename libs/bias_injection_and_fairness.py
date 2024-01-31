from imported_libraries import *
from data_prep import *
from basic_visualizations import *

np.random.seed(123)

###BIAS AND FAIRNESS##############################################################################################
def eq_op_dif(y_true, y_predicted, sensitive_attribute, no_abs=False):
    """
    Compute Equal Opportunity fairness metric.

    Parameters:
    y_predicted (array-like): Predicted labels (0 or 1). MUST BE BINARY!
    y_true (array-like): True labels (0 or 1). MUST BE BINARY!
    sensitive_attribute (array-like): Binary sensitive attribute (0 or 1).

    Returns:
    float: Equal Opportunity score (0 to 1).
    """


    cm_privileged = confusion_matrix(y_true[sensitive_attribute == 1],
                                     y_predicted[sensitive_attribute == 1])  ##confusion matrix class 1
    cm_unprivileged = confusion_matrix(y_true[sensitive_attribute == 0],
                                       y_predicted[sensitive_attribute == 0])  ## confusion matrix class 0


    # Calculate True Positive Rates (TPR)
    TPR_privileged = cm_privileged[1, 1] / (cm_privileged[1, 0] + cm_privileged[1, 1]) if cm_privileged[1, 1] + \
                                                                                          cm_privileged[1, 0] > 0 else 0
    TPR_unprivileged = cm_unprivileged[1, 1] / (cm_unprivileged[1, 0] + cm_unprivileged[1, 1]) if cm_unprivileged[
                                                                                                      1, 1] + \
                                                                                                  cm_unprivileged[
                                                                                                      1, 0] > 0 else 0

    # Calculate Equal Opportunity score
    equal_opportunity_score = abs(TPR_privileged - TPR_unprivileged)
    if no_abs == True:
        equal_opportunity_score = TPR_privileged - TPR_unprivileged

    return equal_opportunity_score

# Test the function
result = eq_op_dif( pd.DataFrame(np.array([1,0,0,1]), columns=['class']), np.array([1,0,0,1]), np.array([True, True, False, False])  )

# Assert statement
assert result<=0.000001, "Equal Opportunity difference should be between 0 in this case"
## I put 0.000001 instead of 0 just in case of small numerical intsability. Probably works with 0 too.








def fairness_optimizer_results(threshold_optimizer, X_fit, y_fit, X_obs, y_obs, y_train, y_val, sensitive_1_fit,
                               sensitive_2_fit,
                               sensitive_1_obs, sensitive_2_obs, name_1, name_2, fitted=False, name_dataset1="train",
                               name_dataset2="val"):
    '''treshold optimiizer should have prefit= True
     sensitive_1 = feature we do optimiziation w.r.t
      sensitive_2 = the other feature that we see how was the fairness affected
      y_train, y_val = true y
      X_fit, y_fit= the data used as train for fitting fairness optimizer
      X_obs, y_obs = validation for fairness optimizer
      y_fit, y_obs= predicted y's before optimizer '''

    if fitted == False:
        threshold_optimizer.predict_method = 'auto'
        threshold_optimizer.fit(X_fit, y_train, sensitive_features=sensitive_1_fit)

    adjusted_sensitive_train = threshold_optimizer.predict(X_fit, sensitive_features=sensitive_1_fit, random_state=123)
    adjusted_sensitive_val = threshold_optimizer.predict(X_obs, sensitive_features=sensitive_1_obs, random_state=123)

    initial_acc_train, initial_acc_test, after_acc_train, after_acc_test = accuracy_score(y_fit,y_train), accuracy_score(y_obs, y_val),\
        accuracy_score(adjusted_sensitive_train, y_train), accuracy_score(adjusted_sensitive_val, y_val)

    initial_fair_train_1, initial_fair_test_1, after_fair_train_1, after_fair_test_1 = eq_op_dif(y_train, y_fit,sensitive_attribute=sensitive_1_fit),\
        eq_op_dif( y_val, y_obs, sensitive_attribute=sensitive_1_obs), eq_op_dif(y_train, adjusted_sensitive_train,sensitive_attribute=sensitive_1_fit),\
        eq_op_dif(y_val, adjusted_sensitive_val, sensitive_attribute=sensitive_1_obs)

    initial_fair_train_2, initial_fair_test_2, after_fair_train_2, after_fair_test_2 = eq_op_dif(y_train, y_fit,sensitive_attribute=sensitive_2_fit), \
        eq_op_dif(y_val, y_obs, sensitive_attribute=sensitive_2_obs), eq_op_dif(y_train, adjusted_sensitive_train,sensitive_attribute=sensitive_2_fit),\
        eq_op_dif(y_val, adjusted_sensitive_val, sensitive_attribute=sensitive_2_obs)
    
    return initial_acc_train, initial_acc_test, after_acc_train, after_acc_test, initial_fair_train_1, initial_fair_test_1, after_fair_train_1, after_fair_test_1, initial_fair_train_2, initial_fair_test_2, after_fair_train_2, after_fair_test_2


def use_fairness_optimizer(threshold_optimizer, X_fit, y_fit, X_obs, y_obs, y_train, y_val, sensitive_1_fit,
                           sensitive_2_fit,
                           sensitive_1_obs, sensitive_2_obs, name_1, name_2, fitted=False, name_dataset1="dataset used for fariness fitting",
                           name_dataset2="test"):
    '''treshold optimiizer should have prefit= True
     sensitive_1 = feature we do optimiziation w.r.t
      sensitive_2 = the other feature that we see how was the fairness affected
      y_train, y_val = true y
      X_fit, y_fit= the data used as train for fitting fairness optimizer
      X_obs, y_obs = validation for fairness optimizer
      y_fit, y_obs= predicted y's before optimizer '''

    if fitted == False:
        threshold_optimizer.predict_method = 'auto'
        threshold_optimizer.fit(X_fit, y_train, sensitive_features=sensitive_1_fit)

    adjusted_sensitive_train = threshold_optimizer.predict(X_fit, sensitive_features=sensitive_1_fit, random_state=123)
    adjusted_sensitive_val = threshold_optimizer.predict(X_obs, sensitive_features=sensitive_1_obs, random_state=123)

    print(f"--------- SCORES AFTER OPTIMIZING FOR {name_1} ---------")
    print()
    print("----- accuracy scores -----")

    print(
        f' acc score on {name_dataset1} got from : {np.round(accuracy_score(y_fit, y_train),3)} to {np.round(accuracy_score(adjusted_sensitive_train, y_train),3)}')
    print(
        f" acc score on {name_dataset2} from : {np.round(accuracy_score(y_obs, y_val),3)} to {np.round(accuracy_score(adjusted_sensitive_val, y_val),3)}")

    print()
    print("----- Scores for fariness -----")

    print(
        f'{name_1} on {name_dataset1} eq op difference went from: {np.round(eq_op_dif(y_train, y_fit, sensitive_attribute=sensitive_1_fit),3)} to '
        f'{np.round(eq_op_dif(y_train, adjusted_sensitive_train, sensitive_attribute=sensitive_1_fit),3)}')

    print(
        f"{name_1} on {name_dataset2} eq op difference went from :  {np.round(eq_op_dif(y_val, y_obs, sensitive_attribute=sensitive_1_obs),3)} to"
        f" {np.round(eq_op_dif(y_val, adjusted_sensitive_val, sensitive_attribute=sensitive_1_obs),3)}")

    print(
        f'{name_2} on {name_dataset1} eq op difference went from: {np.round(eq_op_dif(y_train, y_fit, sensitive_attribute=sensitive_2_fit),3)} to '
        f'{np.round(eq_op_dif(y_train, adjusted_sensitive_train, sensitive_attribute=sensitive_2_fit),3)}')

    print(
        f"{name_2} on {name_dataset2} eq op difference went from :  {np.round(eq_op_dif(y_val, y_obs, sensitive_attribute=sensitive_2_obs),3)} to "
        f"{np.round(eq_op_dif(y_val, adjusted_sensitive_val, sensitive_attribute=sensitive_2_obs),3)}")

    print()
    print()

################################## functions to add bias


def add_bias(X, y, unprivileged_class_name, unprivileged_class_value, p, verbose=True):
    ''' X = predictors dataframe
        y = values to predict
        unprivileged_class_name = str of unprivileged class
        unprivileged class values = int value for unprivileged (e.g. 0 or 1)
        p=probability to keep
         returns the biased data (X,y)'''
    np.random.seed(123)
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


def find_trend(data_fairness_age, data_fairness_gender, data_accuracy, correlation_test, ls_p_vary, vary_age=True, p_fixed=0.25):
    '''
    data_fairness_age, gender, acc should be from the same set (i.e. all training/test)
    first position in tuple is for age, second for gender
    It shows how fairness is influenced by the parameter of the bias addition to the data (used more as a sanity check for the method that biases the data)
    '''
    if vary_age == True:
        vals = []
        for p_vary in ls_p_vary:
            vals.append(data_fairness_age[(p_vary, p_fixed)])  ## Fairness age

        plt.plot(ls_p_vary, vals)
        plt.ylabel("Eq op dif")
        plt.xlabel("probability of keeping positive samples w.r.t age")
        plt.title("Correlation analysis")
        plt.show()
        print(f" Age correlation: {correlation_test(ls_p_vary, vals)}")

        vals = []
        for p_vary in ls_p_vary:
            vals.append(data_fairness_gender[(p_vary, p_fixed)])  ## Fairness gender

        plt.plot(ls_p_vary, vals)
        plt.ylabel("Eq op dif")
        plt.xlabel("probability of keeping positive samples w.r.t age")
        plt.title("Correlation analysis")
        plt.show()
        print(f" Gender correlation: {spearmanr(ls_p_vary, vals)}")  # gender

        vals = []
        for p_vary in ls_p_vary:
            vals.append(data_accuracy[(p_vary, p_fixed)])  ## Fairness age

        plt.plot(ls_p_vary, vals)
        plt.ylabel("Accuracy")
        plt.xlabel("probability of keeping positive samples w.r.t age")
        plt.title("Correlation analysis")
        plt.show()
        print(f"Accuracy correlation: {spearmanr(ls_p_vary, vals)}")
        print()
        print()


    else:  # vary gender

        vals = []
        for p_vary in ls_p_vary:
            vals.append(data_fairness_age[(p_fixed, p_vary)])  ## Fairness age

        plt.plot(ls_p_vary, vals)
        plt.ylabel("Eq op dif")
        plt.xlabel("probability of keeping positive samples w.r.t gender")
        plt.title("Correlation analysis")
        plt.show()
        print(f"Age correlation: {correlation_test(ls_p_vary, vals)}")

        vals = []
        for p_vary in ls_p_vary:
            vals.append(data_fairness_gender[(p_fixed, p_vary)])  ## Fairness gender

        plt.plot(ls_p_vary, vals)
        plt.ylabel("Eq op dif")
        plt.xlabel("probability of keeping positive samples w.r.t gender")
        plt.title("Correlation analysis")
        plt.show()
        print(f" Gender correlation: {correlation_test(ls_p_vary, vals)}")

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


##MAIN FUNCTION THAT ANALYSIS FAIRNES CORRELATION WITH ACCURACY OR FAIRNESS FOR ANOTHER SENSITIVE FEATURE
def find_trend_optimization(before_fairness_age, before_fairness_gender, before_accuracy, after_fairness_age,
                            after_fairness_gender, after_accuracy, correlation_test, ls_p_vary, vary_age=True, std=True,
                            size1=0, size2=0, p_fixed = 0.25):
    '''
    data_fairness_age, gender, acc should be from the same set (i.e. all training/test)
    first position in tuple is for age, second for gender
    Take care! Optimization is done w.r.t what we vary. I.e, vary age fairness, optimize w.r.t age

    The function can compute the following:
    Correlation between keeping positive samples w.r.t age and fairness increase after optimization (for both age and gender)
    Correlation between keeping positive samples w.r.t gender and fairness increase after optimization (for both age and gender)
    Correlation between AGE FAIRNESS and GENDER FARINESS when optimizing w.r.t one of them
    Correlation between (AGE FARINESS or GENDER FAIRNESS) and ACCURACY  when optimizing w.r.t one of the sensitive features

    '''

    if std == True:
        assert size1 > 0 and size2 > 0, "Both sizes must be positive"
    if vary_age == True:
        print("Vary age and optimize w.r.t age")
        print()

        vals_age = []
        margin_of_error = []
        for p_vary in ls_p_vary:
            vals_age.append(
                after_fairness_age[(p_vary, p_fixed)] - before_fairness_age[(p_vary, p_fixed)])  ## Fairness age
            if std == True:
                margin_of_error.append(
                    deviation_for_CI(size1=size1, size2=size2, p1=before_fairness_age[(p_vary, p_fixed)],
                                     p2=after_fairness_age[(p_vary, p_fixed)], alpha=0.05))

        plt.plot(ls_p_vary, vals_age)
        if std == True:
            plt.fill_between(ls_p_vary, np.array(vals_age) - np.array(margin_of_error),
                             np.array(vals_age) + np.array(margin_of_error), alpha=0.2, label='CI')
        plt.ylabel("Eq op dif")
        plt.xlabel("probability of keeping positive samples w.r.t age")
        plt.title("Correlation analysis")
        plt.show()
        print(
            f" Correlation between probability of keeping positive samples w.r.t age and age fairness increase: {correlation_test(ls_p_vary, vals_age)}")

        vals_gender = []

        margin_of_error = []

        for p_vary in ls_p_vary:
            vals_gender.append(after_fairness_gender[(p_vary, p_fixed)] - before_fairness_gender[
                (p_vary, p_fixed)])  ## Fairness gender
            if std == True:
                margin_of_error.append(
                    deviation_for_CI(size1=size1, size2=size2, p1=before_fairness_gender[(p_vary, p_fixed)],
                                     p2=after_fairness_gender[(p_vary, p_fixed)], alpha=0.05))

        plt.plot(ls_p_vary, vals_gender)
        if std == True:
            plt.fill_between(ls_p_vary, np.array(vals_gender) - np.array(margin_of_error),
                             np.array(vals_gender) + np.array(margin_of_error), alpha=0.2, label='CI')

        plt.ylabel("Eq op dif")
        plt.xlabel("probability of keeping positive samples w.r.t age")
        plt.title("Correlation analysis")
        plt.show()
        print(
            f"  Correlation between probability of keeping positive samples w.r.t age and gender fairness increase: {correlation_test(ls_p_vary, vals_gender)}")  # gender

        vals_acc = []
        margin_of_error = []
        for p_vary in ls_p_vary:
            vals_acc.append(after_accuracy[(p_vary, p_fixed)] - before_accuracy[(p_vary, p_fixed)])  ## Fairness age
            if std == True:
                margin_of_error.append(deviation_for_CI(size1=size1, size2=size2, p1=before_accuracy[(p_vary, p_fixed)],
                                                        p2=after_accuracy[(p_vary, p_fixed)], alpha=0.05))

        plt.plot(ls_p_vary, vals_acc)
        if std == True:
            plt.fill_between(ls_p_vary, np.array(vals_acc) - np.array(margin_of_error),
                             np.array(vals_acc) + np.array(margin_of_error), alpha=0.2, label='CI')

        plt.ylabel("Accuracy")
        plt.xlabel("probability of keeping positive samples w.r.t age")
        plt.title("Correlation analysis")
        plt.show()
        print(
            f" Correlation between probability of keeping positive samples w.r.t age and accuracy increase: {correlation_test(ls_p_vary, vals_acc)}")
        print()

        print(f"-- RESULTS ABOUT CORRELATION BETWEEN FAIRNESS AND ACC --")
        print(
            f"Coorelation between age fairness improvement and gender fairness : {correlation_test(vals_age, vals_gender)}")
        print(
            f"Coorelation between age fairness improvement and accuracy : {correlation_test(vals_age, vals_acc)}")  # we do only gender and acc bc we optimize w.r.t gender
        print()
        print()



    else:  # vary gender

        print("Vary gender and optimize w.r.t gender")
        print()

        vals_age = []
        margin_of_error = []
        for p_vary in ls_p_vary:
            if std == True:
                margin_of_error.append(
                    deviation_for_CI(size1=size1, size2=size2, p1=before_fairness_age[(p_fixed, p_vary)],
                                     p2=after_fairness_age[(p_fixed, p_vary)], alpha=0.05))
            vals_age.append(
                after_fairness_age[(p_fixed, p_vary)] - before_fairness_age[(p_fixed, p_vary)])  ## Fairness age

        plt.plot(ls_p_vary, vals_age)
        if std == True:
            plt.fill_between(ls_p_vary, np.array(vals_age) - np.array(margin_of_error),
                             np.array(vals_age) + np.array(margin_of_error), alpha=0.2, label='CI')
        plt.ylabel("Eq op dif")
        plt.xlabel("probability of keeping positive samples w.r.t gender")
        plt.title("Correlation analysis")
        plt.show()
        print(
            f" Correlation between probability of keeping positive samples w.r.t gender and age fairness increase: {correlation_test(ls_p_vary, vals_age)}")

        vals_gender = []
        margin_of_error = []
        for p_vary in ls_p_vary:
            if std == True:
                margin_of_error.append(
                    deviation_for_CI(size1=size1, size2=size2, p1=before_fairness_gender[(p_fixed, p_vary)],
                                     p2=after_fairness_gender[(p_fixed, p_vary)], alpha=0.05))
            vals_gender.append(after_fairness_gender[(p_fixed, p_vary)] - before_fairness_gender[
                (p_fixed, p_vary)])  ## Fairness gender

        plt.plot(ls_p_vary, vals_gender)
        if std == True:
            plt.fill_between(ls_p_vary, np.array(vals_gender) - np.array(margin_of_error),
                             np.array(vals_gender) + np.array(margin_of_error), alpha=0.2, label='CI')
        plt.ylabel("Eq op dif")
        plt.xlabel("probability of keeping positive samples w.r.t gender")
        plt.title("Correlation analysis")
        plt.show()
        print(
            f" Correlation between probability of keeping positive samples w.r.t gender and gender fairness increase: {correlation_test(ls_p_vary, vals_gender)}")

        vals_acc = []
        margin_of_error = []
        for p_vary in ls_p_vary:
            if std == True:
                margin_of_error.append(deviation_for_CI(size1=size1, size2=size2, p1=before_accuracy[(p_fixed, p_vary)],
                                                        p2=after_accuracy[(p_fixed, p_vary)], alpha=0.05))
            vals_acc.append(after_accuracy[(p_fixed, p_vary)] - before_accuracy[(p_fixed, p_vary)])  ## Accuracy

        plt.plot(ls_p_vary, vals_acc)
        if std == True:
            plt.fill_between(ls_p_vary, np.array(vals_acc) - np.array(margin_of_error),
                             np.array(vals_acc) + np.array(margin_of_error), alpha=0.2, label='CI')
        plt.ylabel("Accuracy")
        plt.xlabel("probability of keeping positive samples w.r.t gender")
        plt.title("Correlation analysis")
        plt.show()
        print(
            f"Correlation between probability of keeping positive samples w.r.t gender and accuracy increase:: {correlation_test(ls_p_vary, vals_acc)}")
        print()

        print(f"-- RESULTS ABOUT CORRELATION BETWEEN FAIRNESS AND ACC --")
        print(
            f"Coorelation between gender fairness improvement and age fairness : {correlation_test(vals_gender, vals_age)}")
        print(
            f"Coorelation between gender fairness improvement and accuracy : {correlation_test(vals_gender, vals_acc)}")  # we do only gender and acc bc we optimize w.r.t gender
        print()
        print()


####Accuracy fairness correlation backup

def find_trend_optimization_backup(before_fairness_age, before_fairness_gender, before_accuracy, after_fairness_age,
                                   after_fairness_gender, after_accuracy, correlation_test, ls_p_vary, vary_age=True):
    '''
    data_fairness_age, gender, acc should be from the same set (i.e. all training/test)
    first position in tuple is for age, second for gender
    '''
    if vary_age == True:
        p_fixed = 1  # Fix no drop in positive samples w.r.t gender
        vals = []
        for p_vary in ls_p_vary:
            vals.append(after_fairness_age[(p_vary, p_fixed)] - before_fairness_age[(p_vary, p_fixed)])  ## Fairness age

        plt.plot(ls_p_vary, vals)
        plt.ylabel("Eq op dif")
        plt.xlabel("probability of keeping positive samples w.r.t age")
        plt.title("Correlation analysis")
        plt.show()
        print(f" Age correlation: {correlation_test(ls_p_vary, vals)}")

        p_fixed = 1  # Fix no drop in positive samples w.r.t age  (keep it fixed)
        vals = []
        for p_vary in ls_p_vary:
            vals.append(after_fairness_gender[(p_vary, p_fixed)] - before_fairness_gender[
                (p_vary, p_fixed)])  ## Fairness gender

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
            vals.append(after_fairness_gender[(p_fixed, p_vary)] - before_fairness_gender[
                (p_fixed, p_vary)])  ## Fairness gender

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


def create_iterative_german_bias(X, y, unprivileged_class_name1="Age_group", unprivileged_class_name2="Gender",
                                 unprivileged_class_value1=0, unprivileged_class_value2=0,
                                 p_range1=[0.2, 0.5, 0.8], p_range2=[0.2, 0.5, 0.8], verbose=True,
                                 dataset_name='German_credit_biased'):
    '''Function that  creates subsamples of the initial dataset for different levels of bias injection
       The function stores all these datasets.
       p_range1 is the bias range for age
       p_range2 is the bias range for gender
       class 1 is Age
       class 2 is Gender'''

    for p1 in p_range1:
        for p2 in p_range2:
            Xc, yc = add_bias(X=X, unprivileged_class_name=unprivileged_class_name1,
                              unprivileged_class_value=unprivileged_class_value1, y=y, p=p1,
                              verbose=True)  # Age bias
            Xc, yc = add_bias(X=Xc, unprivileged_class_name=unprivileged_class_name2,
                              unprivileged_class_value=unprivileged_class_value2, y=yc, p=p2,
                              verbose=True)  # Gender bias

            num_features = ["Attribute2", "Attribute5", "Attribute8", "Attribute11", "Attribute13", "Attribute16",
                            "Attribute18"]
            cat_features = [col_name for col_name in X.columns if col_name not in num_features]
            Xc = pd.get_dummies(Xc, columns=cat_features, drop_first=True)

            X_train, X_test, y_train, y_test = train_test_split(Xc, yc, test_size=0.4, random_state=12)

            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                            test_size=0.5, random_state=12)  ##this make 0.2 for both val and test

            ## Save sensitive attributes

            gender_train = X_train["Gender_1"]
            age_train = X_train["Age_group_1"]

            gender_test = X_test["Gender_1"]
            age_test = X_test["Age_group_1"]

            gender_val = X_val["Gender_1"]
            age_val = X_val["Age_group_1"]

            X_train_with_A = X_train.copy()  # X with sensitive_attributes
            X_val_with_A = X_val.copy()  # X with sensitive_attributes
            X_test_with_A = X_test.copy()  # X with sensitive_attributes



            store_data(dataset_name=dataset_name, X_train_with_A=X_train_with_A, X_val_with_A=X_val_with_A,
                       X_test_with_A=X_test_with_A, y_train=y_train,y_val=y_val, y_test=y_test, age=[age_train, age_val, age_test],
                       gender=[gender_train, gender_val, gender_test], education=None, sufix_name=f"_{p1}_{p2}")



