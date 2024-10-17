"""
This script fits linear models and Random Forest to SNP
for each trait in the dataset.
"""

from sklearn.ensemble import RandomForestRegressor
from Main.HelperClasses.GetAraData import *
from Main.HelperClasses.MeasurePerformance import *

kb = 10000
r2_fil = 0.1

import os

INDIVIDUAL_RESULTS = f'Main/results/more_fil_{kb}kb_{r2_fil}/individual'
RESULTS = f"Main/results/more_fil_{kb}kb_{r2_fil}/"
os.mkdir(f"Main/results/more_fil_{kb}kb_{r2_fil}")
os.mkdir(f'Main/results/more_fil_{kb}kb_{r2_fil}/individual')
ara_data = GetAraData(path_to_data='/Research_Data_new/ReasearchProject/data', maf=0.05, window_kb=kb, r2=r2_fil)

traits =  sorted(ara_data.get_trait_names())

cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42) 
for trait in tqdm(list(pd.read_csv('Main/results/traits_used.csv')['name'])):

    # get data
    print("trait: ", trait)
    X = ara_data.get_genotype(trait)
    print(X.shape)
    y = ara_data.get_normalised_phenotype(trait) 

    results_linear = []
    results_rf = []

    ## fits linear models
    #linear_models = ['ridge', 'lasso', 'elasticnet']
    ## bounds for the hyperparameters dont have to be the same for all models
    #results_linear += test_linear_model(X, y, cv=cv, alphas=[1.5**x for x in range(-45,40)][::-1], n_jobs=-1, name_of_feature_representations='SNPs',
    #                                ridge_lasso_enet=[True, False, False])
    #results_linear += test_linear_model(X, y, cv=cv, alphas=[1.5**x for x in range(-20,3)][::-1], n_jobs=-1, name_of_feature_representations='SNPs',
    #                                ridge_lasso_enet=[False, True, False], precomute_gram='auto')
    #results_linear += test_linear_model(X, y, cv=cv, alphas=[1.5**x for x in range(-20,20)][::-1], n_jobs=5, name_of_feature_representations='SNPs',
    #                                ridge_lasso_enet=[False, False, True])
    #
    #linear_v = results_linear[-1].make_resuslts_objects_into_vervose_csv(results_linear)
    #linear = results_linear[-1].make_resuslts_objects_into_csv(results_linear)

    #for linear_result, name, feature in zip(results_linear, linear_models, ["SNPs"]*3):
    #    linear_result.get_indidual_predictions().to_csv(f"{INDIVIDUAL_RESULTS}/{trait}{name}{feature}.csv")#
    #linear.to_csv(f"{RESULTS}/{trait}_linear.csv") 

    #linear_v.to_csv(f"{RESULTS}/{trait}_linear_verbose.csv")

    #    ## fit random forest model

    rf_params = {'Model_Name': 'Random_Forest', 'feature_represenation': 'SNPs', 'variance_maintained': '-'}
    result = [(Result_10fold(RandomForestRegressor(n_estimators=500), X, y, important_parameters=rf_params, name='rf', cv=cv, n_jobs=-1))]
    result[0].get_indidual_predictions().to_csv(f"{INDIVIDUAL_RESULTS}/{trait}_rf_gs.csv")
    make_results_to_csv(result, RESULTS, '', trait, 'rf_gs')
    

