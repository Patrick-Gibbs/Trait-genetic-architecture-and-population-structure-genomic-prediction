
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_predict, cross_validate, KFold, LeaveOneOut, GridSearchCV, RepeatedKFold

from glmnet import ElasticNet

import numpy as np
import pandas as pd
from statistics import mean, stdev
from scipy.stats import pearsonr

from collections.abc import Iterable
from copy import deepcopy
PEARSON_R_STATISTIC = 0

# used to know when a model is using grid search
GRIDSEARCH_TYPE = type(GridSearchCV(MLPRegressor(), {"hidden_layer_sizes": [(10,), (20,),]}))

"""
Contains helper functions for measuring the performance of machine learning models.
"""

def std(x):
    """finds the standard devation of an array, catches the case where np.nan is in the array"""
    try:
        if np.nan in x:
            x = list(x)
            x.remove(np.nan)
            return std(x)
        else:
            return stdev(x)
    except:
        return np.nan


def my_Kfold(k=10, state=42):
    """Returns a KFold object with k splits and a random state of 42"""
    return KFold(n_splits=k, random_state=state, shuffle=True)

def my_RepeatedKfold(k=10, repeats=10):
    """Returns a RepeatedKFold object with k splits and a random state of 42"""

    return RepeatedKFold(n_splits=k, n_repeats=repeats, random_state=42)


class Result:
    """Object which stores the computes and stores a summary stastics for a machine learning model
    on a specified dataset."""
    def __init__(self, name: str, important_parameters: dict, X, y):
        self.r2 = None
        self.RMSE = None
        self.name = name
        important_parameters = deepcopy(important_parameters)
        important_parameters.update({'Dimentionality': len(X[0])})
        self.important_parameters = important_parameters

    def get_header_row(self) -> list:
        """Makes a header row for a csv based on the summery stastics held in the object"""
        return list(self.important_parameters.keys()) + ['r2_summary', 'RMSE_summary', 'r2_train', 'RMSE_train']

    def make_resuslts_objects_into_csv(self, result_objects: Iterable, return_csv=True) -> pd.DataFrame:
        """Takes a set of Results objects and greats a csv file organising them"""
        return pd.DataFrame([result_object.get_csv_row() for result_object in result_objects],
                            columns=self.get_header_row())

    def get_csv_row(self) -> list:
        """outputs a list of all the summerary stastics heald in this object a list, such that
        it can from a row of a csv"""
        return list(self.important_parameters.values()) + [self.r2, self.RMSE, self.r2_train, self.RMSE_train]

    def get_important_paramters(self):
        """returns the important parameters of the model"""
        return self.important_parameters

    def get_verbose_header_row(self) -> list:
        """Makes a header row for a csv based on the summery stastics held in the object"""
        return list(self.get_header_row() + list(self.get_fold_metrics().keys()))

    def get_verbose_csv_rows(self):
        """outputs a list of all the summerary stastics heald in this object a list, such that
        it can from a row of a csv as well as the RMSE/r2 for every indidual fold of validation"""
        rows = []
        for i in range(len(list(self.get_fold_metrics().values())[0])):
            values = [value[i] for value in self.get_fold_metrics().values()]
            rows.append(self.get_csv_row() + values)
        return rows

    def make_resuslts_objects_into_vervose_csv(self, result_objects: Iterable, return_csv=True) -> pd.DataFrame:
        """Takes a set of Results objects and greats a csv file organising them"""
        # get all headers
        all_cv_metrics = [result_object.get_fold_metrics() for result_object in result_objects]
        cv_num = len(list(all_cv_metrics[0].values())[0])
        sums_stats_headers = [result_object.get_header_row() for result_object in result_objects]
        sum_stats_rows = [result_object.get_csv_row() for result_object in result_objects]

        summ_stats_dicts = [{k:[v]*cv_num for k,v in zip(sums_stats_header, sum_stats_row)} for sums_stats_header, sum_stats_row in zip(sums_stats_headers, sum_stats_rows)]
        for i, cv_metric in enumerate(all_cv_metrics):
            cv_metric.update(summ_stats_dicts[i])
        dfs = [pd.DataFrame(cv_metric) for cv_metric in all_cv_metrics]
        # merge all dfs
        df = pd.concat(dfs, join = 'outer')
        
        return pd.DataFrame(df)

    def get_fold_metrics(self):
        """returns the metrics for each fold of validation"""
        return self.cross_fold_metrics

class Result_10fold(Result):
    """Object to store the suits for 10fold, represted 10fold CV"""

    def __init__(self, model, X, y, important_parameters, name='', cv=KFold(10, shuffle=True, random_state=42), n_jobs=-1, save_ind=True):
        """takes a `model` and a dataset (`X`, `y`), does leave one out cross validation, the stores the result"""
        super().__init__(name, important_parameters, X ,y)

        # finds results for each fold of the from 10fold CV
        scorer = {'r2': 'r2', 'neg_RMSE': 'neg_root_mean_squared_error'}
        scores = cross_validate(model, X, y, n_jobs=n_jobs, cv=cv, scoring=scorer, return_estimator=True, return_train_score=True, verbose=2)

        # save the predictions for each fold
        if save_ind:
            i = 0
            self.predictions = {'test_index': [], 'y_pred': [], 'y_true': []}
            for train_index, test_index in cv.split(X,y):
                model = scores['estimator'][i]
                y_pred = model.predict(X[test_index])
                self.predictions['test_index'].extend(test_index)
                self.predictions['y_pred'].extend(y_pred)
                self.predictions['y_true'].extend(y[test_index])
                i += 1
            
        # computes summary statistics
        self.r2, self.RMSE = mean(scores['test_r2']), mean(-scores['test_neg_RMSE'])
        
        self.r2_train, self.RMSE_train = mean(scores['train_r2']), mean(-scores['train_neg_RMSE'])

        # multiply 1.96 becouse 95% of the normal curve in +/1 1.96 standard devarions from the mean
        self.standard_error_r2, self.standard_error_RMSE =  \
            std(scores['test_r2'])*1.96, std(-scores['test_neg_RMSE'])*1.96

        self.standard_error_r2_train, self.standard_error_RMSE_train =  \
            std(scores['train_r2'])*1.96, std(-scores['train_neg_RMSE'])*1.96

        self.cross_fold_metrics = {'r2s': scores['test_r2'], 'RMSEs': -scores['test_neg_RMSE'],
            'r2s_train': scores['train_r2'], 'RMSEs_train': -scores['train_neg_RMSE']}
        self.r2s = scores['test_r2']
        self.RMSEs = -scores['test_neg_RMSE']
        self.r2s_train = scores['train_r2']
        self.RMSEs_train = -scores['train_neg_RMSE']

        # case that grid search is used we want to extract all information from the best model
        blank_row = ['-' for _ in range(len(self.r2s))]
        if type(model) == GRIDSEARCH_TYPE:
            best_params = [score.best_params_ for score in scores["estimator"]]
            best_params_combine = {k: [] for k in best_params[0].keys()}
            for best_param in best_params:
                for k in best_param.keys():
                    best_params_combine[k].append(best_param[k])
            self.cross_fold_metrics.update(best_params_combine)

            # catches case of glm net package where there is some nested infomation
            if scores["estimator"][0].best_estimator_.__class__.__name__ == "ElasticNet":
                penalisations = [score.best_estimator_.lambda_best_[0] for score in scores["estimator"]]
                #alphas = [score.best_estimator_.alpha for score in scores["estimator"]]
                self.cross_fold_metrics.update({"lambda_pen": penalisations})


        elif model.__class__.__name__ == "RidgeCV" or model.__class__.__name__ == "LassoCV" or model.__class__.__name__ == "ElasticNetCV":
            alphas = [score.alpha_ for score in scores["estimator"]]
            alphas_dict = {'lambda_pen': alphas}
            self.cross_fold_metrics.update(alphas_dict)
            if model.__class__.__name__ == "ElasticNetCV":
                l2_ratios = [score.l1_ratio_ for score in scores["estimator"]]
                self.cross_fold_metrics.update({'l2_ratio': l2_ratios})

        """if type(model) == type(rrBLUP()):
            var_gs = [score.genetic_varince for score in scores["estimator"]]
            var_es = [score.residual_variance for score in scores["estimator"]]
            self.cross_fold_metrics.update({'var_g': var_gs, 'var_e': var_es})"""
       
        
    def get_indidual_predictions(self):
        """returns the predictions for each fold of validation"""
        return pd.DataFrame(self.predictions)

    def get_header_row(self) -> list:
        """Makes a header row for a csv based on the summery stastics held in the object"""
        return super().get_header_row() + ['standard_error_r2', 'standard_error_RMSE', 'standard_error_r2_train', 'standard_error_RMSE_train']

    def get_csv_row(self):
        """outputs a list of all the summerary stastics heald in this object a list, such that
        it can from a row of a csv"""
        return super().get_csv_row() + [self.standard_error_r2, self.standard_error_RMSE, self.standard_error_r2_train, self.standard_error_RMSE_train]

    def get_RMSEs(self):
        """returns the RMSEs for each fold of validation"""
        return (self.RMSEs)

    def get_r2s(self):
        """returns the r2s for each fold of validation"""
        return np.array(self.r2s)

   

"""The following functions are used for experiment 3 and allow classification and regression tasks to be handled."""
def test_linear_model(X, y, 
    import_parameters = {}, 
    n_jobs = -1, 
    cv=my_Kfold, 
    lasso_cv=5,
    alphas = [1.5**x for x in range(-40,45)][::-1], 
    ridge_lasso_enet = [True, True, True], 
    ratios = [0.01, 0.1, 0.5, 0.85, 0.9, 0.97, 0.99, 0.995],
    fast_compute=False,
    pca_variance='-',
    name_of_feature_representations='-',
    enet_reg_path = None,
    smooth=False,
    trait_name = None,
    lasso_tol = 0.0001,
    lasso_learning_rate = 0.001,
    save_ind=True
    ):
    """tests ridge, lasso, and elastic net regression on a dataset for a range of different penalisation parameters.
    `import_parameters` is a dictionary of ifomation to be saved in the csv with the model.
    `n_jobs` is the number of jobs to run in parallel, -1 means use all processors.
    `cv` is the cross validation method to use.
    `alphas` is the range of penalisation parameters to test.
    `ridge_lasso_enet` is a list of booleans, if true then the corresponding model is tested.
    `ratios` is the range of ratios to test for elastic net regression.
    `fast_compute` is a boolean, if true then the search space is reduced to make the computation faster.
    `pca_variance` is the variance of the pca used to reduce the dimensionality of the data.
    `name_of_feature_representations` is the name of the feature representation used.
    `enet_reg_path` is the path to the file containing the elastic net penalisation parameters.
    `trait_name` is the name of the trait being predicted.
  
    """
    """tests ridge, lasso, and elastic net regression on a dataset for a range of different penalisation parameters.
    `import_parameters` is a dictionary of ifomation to be saved in the csv with the model.
    `n_jobs` is the number of jobs to run in parallel, -1 means use all processors.
    `cv` is the cross validation method to use.
    `alphas` is the range of penalisation parameters to test.
    `ridge_lasso_enet` is a list of booleans, if true then the corresponding model is tested.
    `ratios` is the range of ratios to test for elastic net regression.
    `fast_compute` is a boolean, if true then the search space is reduced to make the computation faster.
    `pca_variance` is the variance of the pca used to reduce the dimensionality of the data.
    `name_of_feature_representations` is the name of the feature representation used.
    `enet_reg_path` is the path to the file containing the elastic net penalisation parameters.
    `trait_name` is the name of the trait being predicted.
  
    """

    penalisation_values = alphas
    
    # sets the number of jobs to use to avoid memory overflow


    # option for testing, makes search space smaller
    if fast_compute:
        penalisation_values = [1.25**x for x in range(-2,2)]
        ratios = [0.5, 0.99]
        n_lambda = 2
    else:
        n_lambda = len(penalisation_values)
    
    # elastic net model to be tested, uses GLM net package
    # elastic net model to be tested, uses GLM net package
    if enet_reg_path is not None:
        enet = GridSearchCV(ElasticNet(n_splits=5, n_jobs=10, lambda_path=enet_reg_path, max_iter=1000000), param_grid = {'alpha': ratios}, n_jobs=n_jobs)
    else:
        enet = GridSearchCV(ElasticNet(n_splits=5, n_jobs=10, n_lambda=n_lambda, max_iter=1000000), param_grid = {'alpha': ratios}, n_jobs=n_jobs)
    
    
    models = [RidgeCV(alphas = np.array(penalisation_values)), 
              LassoCV(alphas=penalisation_values, cv=lasso_cv, max_iter=1000000, tol=lasso_tol, eps=lasso_learning_rate),
              enet]

    # includes only chosen models based on the function arguments 
    models = [model for i, model in enumerate(models) if ridge_lasso_enet[i]]
    model_names = np.array(['Ridge', 'Lasso', 'ElasticNet'])[np.array(ridge_lasso_enet)]
    
    # carries params through the model
    params = [deepcopy(import_parameters) for i in range(len(models))]
    for i, name in enumerate(model_names):
        params[i]['Model_Name'] = name
        params[i]["feature_represenation"] = name_of_feature_representations
        params[i]["variance_maintained"] = pca_variance

    # runs the models and records results
    res =  [Result_10fold(model, X, y, cv=cv, important_parameters=param, n_jobs=n_jobs, save_ind=save_ind)
                for model, param in zip(models, params)]
    
    return res

def make_results_to_csv(results_objects, results_dir, other_info, trait, model):
    """saves the results of the models from Result object to a csv file."""

    # saves the results for linear regression to a csv
    df_v = results_objects[-1].make_resuslts_objects_into_vervose_csv(results_objects)
    df_v.to_csv(f"{results_dir}_{trait}_{model}_verbose_{other_info}.csv")
    
    # saves the results for linear regression to a csv
    df = results_objects[-1].make_resuslts_objects_into_csv(results_objects)
    if other_info:
        df.to_csv(f"{results_dir}{trait}_{model}_{other_info}.csv")
    else:
        df.to_csv(f"{results_dir}{trait}_{model}.csv")

