from Main.HelperClasses.GetAraData import *
from Main.HelperClasses.MeasurePerformance import *
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.dummy import DummyRegressor
import pandas as pd
from tqdm import tqdm
from sklearn.tree import RandomForestRegressor


ara_data = GetAraData(path_to_data='./data', maf=0.05, window_kb=200, r2=0.6)

cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42)
PC_VARIANCES_TO_TEST = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,  0.8,  0.9, 0.95, 0.97, 0.99, 1]

alphas= [2**x for x in range(-20,20)]
INDIVIDUAL = 'genomic_prediction_programs/Experiments/Paper/pca_other/individual'
RESULTS_DIR = "genomic_prediction_programs/Experiments/Paper/pca_other/"
for trait in (['study_12_FT10', 'herbavore_resistance_G2P', 'study_4_M130T666']):
    trait_results = []
    for pc_var in tqdm(PC_VARIANCES_TO_TEST):
        print(trait, pc_var)

        X = ara_data.get_pca_feature_reduced_SNPs(trait, variance_maintained=pc_var)
        y = ara_data.get_normalised_phenotype(trait)
       
        trait_results += test_linear_model(X, y, cv=cv, alphas= [1.5**x for x in range(-20,40)], n_jobs=-1, pca_variance=pc_var, name_of_feature_representations='PCA_SNPS'
                                          , ridge_lasso_enet=[True, False, False], )
        

        trait_results += test_linear_model(X, y, cv=cv, alphas= [1.5**x for x in range(-10,10)], n_jobs=-1, pca_variance=pc_var, name_of_feature_representations='PCA_SNPS',
                                           ridge_lasso_enet=[False, True, False], lasso_tol=0.01, lasso_cv=my_Kfold(k=5,state=2))
        
        
        trait_results += test_linear_model(X, y, cv=cv, alphas= [1.5**x for x in range(-15,20)], n_jobs=-1, pca_variance=pc_var, name_of_feature_representations='PCA_SNPS',
                                            ridge_lasso_enet=[False, False, True])

        trait_results += [Result_10fold(RandomForestRegressor(n_estimators=500), X, y,
        important_parameters={'Model_Name': 'rf', 'feature_represenation':'PCA_SNPS', 'variance_maintained': str(pc_var)}, 
        cv=cv, n_jobs=-1)]

        make_results_to_csv(trait_results, RESULTS_DIR, '', trait, 'linear')
        trait_results[-1].get_indidual_predictions().to_csv(f"{INDIVIDUAL}/{trait}_linear_{'PCA_SNPS'}_individual_pca{(str(pc_var))}.csv")
        