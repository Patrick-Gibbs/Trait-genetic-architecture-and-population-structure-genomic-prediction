from genomic_prediction_programs.HelperFunctionsAndClasses.GetAraData import *
from genomic_prediction_programs.HelperFunctionsAndClasses.MeasurePerformance import *
from genomic_prediction_programs.HelperFunctionsAndClasses.rrBLUP import *
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.dummy import DummyRegressor
import pandas as pd
from tqdm import tqdm
from sklearn.tree import DecisionTreeRegressor

INDIVIDUAL_RESULTS = 'genomic_prediction_programs/Experiments/Paper/results_choosen_traits/results_choosen_traits_individual'
RESULTS = "genomic_prediction_programs/Experiments/Paper/results_choosen_traits/"
ara_data = GetAraData(path_to_data='./data', maf=0.05)
traits = ['study_1_FRI']#['dummy'] + sorted(ara_data.get_filtered_traits())

cv = my_RepeatedKfold(k=10, repeats=1)
import warnings

for trait in tqdm(traits):
    X = ara_data.get_genotype(trait)
    y = ara_data.get_normalised_phenotype(trait)

    print("trait: ", trait)
    rf_params = {'Model_Name': 'Random_Forest_vanilla', 'feature_represenation': 'SNPs', 'variance_maintained': '-'}
    result = [(Result_10fold(RandomForestRegressor(n_estimators=500), X, y, important_parameters=rf_params, name='rf_vanilla', cv=cv, n_jobs=-1))]
    result[0].get_indidual_predictions().to_csv(f"{INDIVIDUAL_RESULTS}/{trait}rf_vanilla_500.csv")
    make_results_to_csv(result, RESULTS, '', trait, 'rf_vanilla_500')
    
    linear_models = ['ridge', 'lasso', 'elasticnet']
    
    # linear
    warnings.filterwarnings('ignore')
    results_linear = []
    results_linear += test_ridge_model(X, y, cv=cv, alphas=[2**x for x in range(-10,20)][::-1], n_jobs=-1, name_of_feature_representations='SNPs',
                                    ridge_lasso_enet=[True, False, False])
    results_linear += test_ridge_model(X, y, cv=cv, alphas=[2**x for x in range(-20,20)][::-1], n_jobs=-1, name_of_feature_representations='SNPs',
                                    ridge_lasso_enet=[False, True, False], lasso_cv=my_Kfold(5))
    results_linear += test_ridge_model(X, y, cv=cv, alphas=[2**x for x in range(-10,10)][::-1], n_jobs=6, name_of_feature_representations='SNPs',
                                    ridge_lasso_enet=[False, False, True])
    
   
    make_results_to_csv(results_linear, RESULTS, '', trait, 'linear')

    for linear_result, name, feature in zip(results_linear, linear_models, ["SNPs"]*3):
        linear_result.get_indidual_predictions().to_csv(f"{INDIVIDUAL_RESULTS}/{trait}{name}{feature}.csv")

    print("done")
