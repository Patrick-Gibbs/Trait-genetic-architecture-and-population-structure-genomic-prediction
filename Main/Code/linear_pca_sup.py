from Main.HelperClasses.GetAraData import *
from Main.HelperClasses.MeasurePerformance import *
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.dummy import DummyRegressor
import pandas as pd
from tqdm import tqdm


ara_data = GetAraData(path_to_data='/Research_Data_new/ReasearchProject/data/.', maf=0.05, window_kb=200, r2=0.6)

cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42)
PC_VARIANCES_TO_TEST = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1]

RESULTS_DIR = "Main/results/linear_pca_sup/"
INDIVIDUAL = 'Main/results/linear_pca_sup/individual'

for trait in ['study_1_Storage_28_days']:#list(pd.read_csv('Main/results/traits_used.csv')['name']):
    trait_results = []
    for pc_var in tqdm(PC_VARIANCES_TO_TEST):
        print(trait, pc_var)

        X = ara_data.get_pca_feature_reduced_SNPs(trait, variance_maintained=pc_var)
        y = ara_data.get_normalised_phenotype(trait)
       
        trait_results += test_linear_model(X, y, cv=cv, alphas= [1.25**x for x in range(-50,50)], n_jobs=-1, pca_variance=pc_var, name_of_feature_representations='PCA_SNPS'
                                          , ridge_lasso_enet=[True, False, False])
        

        trait_results += test_linear_model(X, y, cv=cv, alphas=None, n_jobs=-1, pca_variance=pc_var, name_of_feature_representations='PCA_SNPS',
                                           ridge_lasso_enet=[False, True, False], lasso_tol=0.01, lasso_cv=my_Kfold(k=5,state=2), precomute_gram=False)
        
        
        trait_results += test_linear_model(X, y, cv=cv, alphas= [1.5**x for x in range(-15,20)], n_jobs=-1, pca_variance=pc_var, name_of_feature_representations='PCA_SNPS',
                                            ridge_lasso_enet=[False, False, True])

        #trait_results += Measure

        make_results_to_csv(trait_results, RESULTS_DIR, '', trait, 'linear')
        trait_results[-1].get_indidual_predictions().to_csv(f"{INDIVIDUAL}/{trait}_linear_{'PCA_SNPS'}_individual_pca{(str(pc_var))}.csv")
        