#!/usr/bin/python
import numpy as np # for numercial computations
import matplotlib.pyplot as plt # for graphing
import seaborn as sns # for extra plots such as heatmap
import pickle # for loading and dumping datasets
import pandas as pd # for easy exploratory data analysis
from time import time # to measure ML process time
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### set random seed for reproducing same results in the future
np.random.seed(42)

### load dictionary and see dataset structure
data_dict = pickle.load(open('final_project_dataset.pkl', 'rb'))

# keys in dictionary
print('keys in dataset: {}'.format(data_dict.keys()))

# number of observations
print('\nThere are {} observations in the dataset.'.format(len(data_dict)))

# number of pois(target)
count = 0
for i in data_dict:
    if data_dict[i]['poi']:
        count += 1
print('\nThere are {} POIs in the dataset.'.format(count))

# number of features
print('\nThere are {} predictor features and 1 target feature in the dataset.'.format(len(data_dict['METTS MARK'])))



### exploratory data analysis and outliers
# remove 'THE TRAVEL AGENCY IN THE PARK' - this is not an employee
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

# rmoeve 'LOCKHART EUGENE E' as this peron only has NaN value for predictive variables
data_dict.pop('LOCKHART EUGENE E')

# create pandas dataframe for exploratory data analysis
data_df = pd.DataFrame(data_dict).transpose()

# get rid of 'email_address'(non-numerical) and 'poi'(target) from the list of variables
data_df.drop(['email_address', 'poi'], axis = 1, inplace = True)

# convert all 'NaN' string values to np.NaN values, then turn all values to float values for numerical computations
for i in data_df:
    for j in range(len(data_df)):
        if data_df[i][j] == 'NaN':
            data_df[i][j] = np.NaN

data_df = data_df.astype(np.float)

# check for null values for each variable
print(data_df.info())

# create heatmap to view correlation among variables
data_df_corr = data_df.corr()
mask = np.zeros_like(data_df_corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(data_df_corr, cmap = 'RdBu_r', mask = mask, alpha = 0.7)
plt.show()

# choose two variables to see if they are truly correlated
plt.scatter(data = data_df, x = 'salary', y = 'bonus')
plt.show()

# identify outlier
data_df[data_df['salary'] > 25000000]

# get rid of 'TOTAL' from both the dataframe and the dictionary
data_dict.pop('TOTAL')
data_df = data_df[data_df['salary'] < 25000000]

# create the heatmap and scatterplot again to examine changes after getting rid of huge outlier
plt.scatter(data = data_df, x = 'salary', y = 'bonus')
plt.show()

mask = np.zeros_like(data_df.corr(), dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(data_df.corr(), cmap = 'RdBu_r', mask = mask, alpha = 0.7)
plt.show()



### Feature engineering, scaling, and selection
# original features_list
features_list = ['poi', 'salary', 'deferral_payments', 
                 'total_payments', 'loan_advances', 'bonus', 
                 'restricted_stock_deferred', 'deferred_income', 
                 'total_stock_value', 'expenses', 
                 'exercised_stock_options', 'other', 
                 'long_term_incentive', 'restricted_stock', 
                 'director_fees','to_messages', 
                 'from_poi_to_this_person', 'from_messages', 
                 'from_this_person_to_poi', 'shared_receipt_with_poi',
                 'stock_salary_proportion']

# create new feature 'stock_salary_proportion', which shows the proportion of stock out of the sum of stock and salary
# this measures how much of the person's compensation is of 'arbitary value'
# my theory is that the person will be more interested in boosting the company's arbitary face value
# if they have high stock proportion, so that they can also benefit from improved stock value
for i in data_dict:
    emp = data_dict[i]
    try:
        emp['stock_salary_proportion'] = emp['total_stock_value'] / (emp['total_stock_value'] + emp['salary'])
    except:
        emp['stock_salary_proportion'] = 'NaN'

# store new dataset into my_dataset
my_dataset = data_dict

# create function select k best features from dataset
def select_k_best(data_dict, features_list, k):
    # import selectkbest module and f_classif for selector function
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # split target and data
    data = featureFormat(data_dict, features_list)
    target, features = targetFeatureSplit(data)
    
    # create selector and fit data
    selector = SelectKBest(f_classif, k = k)
    selector.fit(features, target)
    
    # get scores of individual features and group with feature name
    scores = selector.scores_
    feature_scores = zip(features_list[1:], scores)
    
    # list of features in order of score
    scores_ordered = list(sorted(feature_scores, key = lambda x: x[1], reverse = True))
    
    # select k best features from the list
    k_best = scores_ordered[:k]
    
    # print scores of k best features
    print('scores of {} best features:\n'.format(k))
    for i in k_best:
        print(i)
        
    # return list of best features
    return [i[0] for i in k_best]

# create a list of best features
k_list = [7, 9, 10, 11, 12, 13]
best_features = [select_k_best(my_dataset, features_list, k) for k in k_list]

# create function to transform data dictionary into features and labels
def format_data(data_dict, features_list):
    # add 'poi' to the begnning of the list if not in features_list
    if 'poi' not in features_list:
        features_list.insert(0, 'poi')

    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    return labels, features

# use 7 features for now
labels, features = format_data(my_dataset, best_features[0])
print('These are the 7 best features.')
print(best_features[0])

# split data to train and test set
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size = 0.3)

print('There are {} training points.'.format(len(train_X)))
print('There are {} test points.'.format(len(test_X)))

# scale features using minmaxscaler
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

min_max_scaler.fit(train_X)
train_X = min_max_scaler.transform(train_X)
test_X = min_max_scaler.transform(test_X)




### Testing different algorithms
# binary classification task, and a supervised learning. import all the algorithms to use.
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# create function to create dataframe from classification report
def classification_report_to_df(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['precision'] = float(row_data[2].strip())
        row['recall'] = float(row_data[3].strip())
        row['f1-score'] = float(row_data[4].strip())
        row['support'] = float(row_data[5].strip())
        report_data.append(row)
        
    result = lines[-2].split('      ')
    result = [float(i.strip()) for i in result[1:]]
    
    df_test = pd.DataFrame.from_dict(report_data)
    df_test = df_test.append(pd.DataFrame([result],
                                        columns = ['precision','recall','f1-score','support']),
                           ignore_index = True)
    df_test.index = ['non-poi', 'poi', 'avg/total']
    
    return df_test

classifier_name = ['Decision tree', 'Gaussian Naive Bayes',
                   'SVC', 'Random Forest', 'AdaBoost','KNN']
classifier = [DecisionTreeClassifier(),
             GaussianNB(),
             SVC(),
             RandomForestClassifier(),
             AdaBoostClassifier(),
             KNeighborsClassifier()]

# create function to evaluate each algorithm and return a classification report
def get_scores(train_X, test_X, train_y, test_y, 
               classifier_list, classifier_names):
    
    for i, model in enumerate(classifier):
        clf = classifier[i]
        clf.fit(train_X, train_y)
        pred = clf.predict(test_X)
        report = classification_report(test_y, pred)
        report_df = classification_report_to_df(report)
        print(classifier_names[i],'\n')
        print(report_df, '\n')

# get scores for each algorithm without any tuning
get_scores(train_X, test_X, train_y, test_y, classifier, classifier_name)



### Parameter tuning
### use GridSearchCV and StratifiedShuffleSplit to find best parameters for each module and for each number of features
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

# define scoring function for GridSearchCV, ONLY returning f1_score if both the recall and precision scores are above 0.3 threshold
def scoring(estimator, features_test, labels_test):
    labels_pred = estimator.predict(features_test)
    p = precision_score(labels_test, labels_pred, average='micro')
    r = recall_score(labels_test, labels_pred, average='micro')
    if p > 0.3 and r > 0.3:
        return f1_score(labels_test, labels_pred, average='macro')
    return 0

# define parameters
param_dt = {'max_features': [3,4,5],
           'criterion': ('gini', 'entropy'),
           'max_depth': [1,2,3,4,5,6],
           'min_samples_split': [2,3],
           'min_samples_leaf': [1,2,3,4,5]}
param_rf = {'max_features': [3,4,5],
           'criterion': ('gini', 'entropy'),
           'min_samples_leaf': [1,2,3,4,5],
           'n_estimators': [10,50,100,500]}
param_ada = {'n_estimators': [10,50,100,500],
            'algorithm': ('SAMME','SAMME.R'),
            'learning_rate': [0.05, 0.1, 1.0] }
param_svc = {'C': [1,10,100,1000],
             'gamma': [0.001, 0.0001]}
param_knn = {'n_neighbors': [3, 5, 7, 9],
            'weights': ('uniform', 'distance'),
            'metric': ('minkowski','euclidean'),
            'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute')}
param_nb = {}

params = []
params.append(param_dt)
params.append(param_rf)
params.append(param_ada)
params.append(param_svc)
params.append(param_knn)
params.append(param_nb)

# define classifiers
classifier_name = ['Decision tree', 'Random Forest', 
                   'AdaBoost','SVC','KNN','Gaussian Naive Bayes' ]
classifier = [DecisionTreeClassifier(),
              RandomForestClassifier(),
              AdaBoostClassifier(),
              SVC(kernel = 'rbf'),
              KNeighborsClassifier(),
              GaussianNB()]

# create function to find best estimator for an algorithm and return that estimator
def find_best_estim(classifier, classifier_name, 
                    param, scoring, features, target):
    # create StratifiedShuffleSplit instance
    cv = StratifiedShuffleSplit(n_splits = 10, test_size = 0.1,
                               random_state = 42)
    
    # scale features
    min_max_scaler = MinMaxScaler()
    features = min_max_scaler.fit_transform(features)
    
    # create grid and fit data
    t0 = time()
    clf = classifier
    
    grid = GridSearchCV(clf, param, cv=cv, scoring=scoring)
    grid.fit(features, target)
    
    print(classifier_name)
    print('time taken to process grid: {}'.format(time() - t0))
    print('best parameters: {}'.format(grid.best_params_))
    print('best score: {}'.format(grid.best_score_))
    
    return grid.best_estimator_

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# create labels and features for each number of features
target7, features7 = format_data(data_dict, best_features[0])
target9, features9 = format_data(data_dict, best_features[1])
target10, features10 = format_data(data_dict, best_features[2])
target11, features11 = format_data(data_dict, best_features[3])
target12, features12 = format_data(data_dict, best_features[4])
target13, features13 = format_data(data_dict, best_features[5])

# find best parameters for 7 features
best_estimators_7 = []

print('<7 best features>\n')
for i, model in enumerate(classifier):
    best_estimators_7.append(find_best_estim(model, classifier_name[i], params[i],
                                          scoring, features7, target7))
    print('\n')

# find best parameters for 9 features
best_estimators_9 = []

print('<9 best features>\n')
for i, model in enumerate(classifier):
    best_estimators_9.append(find_best_estim(model, classifier_name[i], params[i],
                                          scoring, features9, target9))
    print('\n')

# find best parameters for 10 features
best_estimators_10 = []

print('<10 best features>\n')
for i, model in enumerate(classifier):
    best_estimators_10.append(find_best_estim(model, classifier_name[i], params[i],
                                          scoring, features10, target10))
    print('\n')

# find best parameters for 11 features
best_estimators_11 = []

print('<11 best features>\n')
for i, model in enumerate(classifier):
    best_estimators_11.append(find_best_estim(model, classifier_name[i], params[i],
                                          scoring, features11, target11))
    print('\n')

# find best parameters for 12 features
best_estimators_12 = []

print('<12 best features>\n')
for i, model in enumerate(classifier):
    best_estimators_12.append(find_best_estim(model, classifier_name[i], params[i],
                                          scoring, features12, target12))
    print('\n')

# find best parameters for 13 features
best_estimators_13 = []

print('<13 best features>\n')
for i, model in enumerate(classifier):
    best_estimators_13.append(find_best_estim(model, classifier_name[i], params[i],
                                          scoring, features13, target13))
    print('\n')



### Validation
### choose only classifiers with scores over 0.7 to test the classifier

print('testing decision tree with 10 variables:')
t0 = time()
test_classifier(best_estimators_10[0], my_dataset, best_features[2], folds = 1000)
print('time taken to process: {}'.format(time() - t0))
print('\n')

print('testing adaboost with 13 variables:')
t0 = time()
test_classifier(best_estimators_13[2], my_dataset, best_features[5], folds = 1000)
print('time taken to process: {}'.format(time() - t0))
print('\n')

print('testing gaussian naive bayes with 9 variables:')
t0 = time()
test_classifier(best_estimators_9[5], my_dataset, best_features[1], folds = 1000)
print('time taken to process: {}'.format(time() - t0))
print('\n')



### final winner
clf = best_estimators_9[5]
features_list = best_features[1]
print('final classifier is: ')
test_classifier(best_estimators_9[5], my_dataset, best_features[1], folds = 1000)


### dump classifer and data
from tester import dump_classifier_and_data
dump_classifier_and_data(clf, my_dataset, features_list)
