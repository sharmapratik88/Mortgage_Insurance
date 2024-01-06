## Credits: https://towardsdatascience.com/machine-learning-classifiers-comparison-with-python-33149aecdbca
# Import required libraries for performance metrics
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate

# Define dictionary with performance metrics
scoring = {'accuracy': make_scorer(accuracy_score), 
           'precision':make_scorer(precision_score, average = 'weighted'),
           'recall':make_scorer(recall_score, average = 'weighted'), 
           'f1_score':make_scorer(f1_score, average = 'weighted')}

# Import required libraries for machine learning classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Instantiate the machine learning classifiers
log_model = LogisticRegression(max_iter=10000)
svc_model = LinearSVC(dual=False)
dtr_model = DecisionTreeClassifier()
rfc_model = RandomForestClassifier()
gnb_model = GaussianNB()
knn_model = KNeighborsClassifier(3)
xgb_model = XGBClassifier(eval_metric = 'mlogloss')
ada_model = AdaBoostClassifier()
gb_model = GradientBoostingClassifier()
cb_model = CatBoostClassifier(verbose = False)

# Define the models evaluation function
def models_evaluation(X, y, folds):
    
    '''
    X : data set features
    y : data set target
    folds : number of cross-validation folds
    
    '''
    
    # Perform cross-validation to each machine learning classifier
    log = cross_validate(log_model, X, y, cv=folds, scoring=scoring)
    svc = cross_validate(svc_model, X, y, cv=folds, scoring=scoring)
    dtr = cross_validate(dtr_model, X, y, cv=folds, scoring=scoring)
    rfc = cross_validate(rfc_model, X, y, cv=folds, scoring=scoring)
    gnb = cross_validate(gnb_model, X, y, cv=folds, scoring=scoring)
    knn = cross_validate(knn_model, X, y, cv=folds, scoring=scoring)
    xgb = cross_validate(xgb_model, X, y, cv=folds, scoring=scoring)
    ada = cross_validate(ada_model, X, y, cv=folds, scoring=scoring)
    gb = cross_validate(gb_model, X, y, cv=folds, scoring=scoring)
    cb = cross_validate(cb_model, X, y, cv=folds, scoring=scoring)

    # Create a data frame with the models perfoamnce metrics scores
    models_scores_table = pd.DataFrame({'Logistic Regression':[log['test_accuracy'].mean(),
                                                               log['test_precision'].mean(),
                                                               log['test_recall'].mean(),
                                                               log['test_f1_score'].mean()],
                                       
                                      'Support Vector Classifier':[svc['test_accuracy'].mean(),
                                                                   svc['test_precision'].mean(),
                                                                   svc['test_recall'].mean(),
                                                                   svc['test_f1_score'].mean()],
                                       
                                      'Decision Tree':[dtr['test_accuracy'].mean(),
                                                       dtr['test_precision'].mean(),
                                                       dtr['test_recall'].mean(),
                                                       dtr['test_f1_score'].mean()],
                                       
                                      'Random Forest':[rfc['test_accuracy'].mean(),
                                                       rfc['test_precision'].mean(),
                                                       rfc['test_recall'].mean(),
                                                       rfc['test_f1_score'].mean()],
                                       
                                      'Gaussian Naive Bayes':[gnb['test_accuracy'].mean(),
                                                              gnb['test_precision'].mean(),
                                                              gnb['test_recall'].mean(),
                                                              gnb['test_f1_score'].mean()],
                                       
                                       'KNearest Neighbor':[knn['test_accuracy'].mean(),
                                                            knn['test_precision'].mean(),
                                                            knn['test_recall'].mean(),
                                                            knn['test_f1_score'].mean()],
                                       
                                       'XG Boost':[xgb['test_accuracy'].mean(),
                                                   xgb['test_precision'].mean(),
                                                   xgb['test_recall'].mean(),
                                                   xgb['test_f1_score'].mean()],
                                       
                                       'AdaBoost':[ada['test_accuracy'].mean(),
                                                   ada['test_precision'].mean(),
                                                   ada['test_recall'].mean(),
                                                   ada['test_f1_score'].mean()],
                                        
                                       'Gradient Boosting':[gb['test_accuracy'].mean(),
                                                            gb['test_precision'].mean(),
                                                            gb['test_recall'].mean(),
                                                            gb['test_f1_score'].mean()],
                                        
                                       'Cat Boost':[cb['test_accuracy'].mean(),
                                                    cb['test_precision'].mean(),
                                                    cb['test_recall'].mean(),
                                                    cb['test_f1_score'].mean()]},
                                      
                                      index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    
    # Add 'Best Score' column
    models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)
    
    # Return models performance metrics scores data frame
    return(models_scores_table)