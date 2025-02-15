"""
This script fits linear models and Random Forest to SNP
for each trait in the dataset.
"""

from sklearn.ensemble import RandomForestRegressor
from Main.HelperClasses.GetAraData import *
from Main.HelperClasses.MeasurePerformance import *

INDIVIDUAL_RESULTS = 'Main/results/all_traits_snps/individual'
RESULTS = "Main/results/all_traits_snps/"
ara_data = GetAraData(path_to_data='/Research_Data_new/ReasearchProject/data', maf=0.05, window_kb=200, r2=0.6)

traits =  sorted(ara_data.get_trait_names())
traits = list(pd.read_csv('Main/results/traits_used.csv')['name'])

cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42) 
for trait in tqdm(traits):

    # get data
    print("trait: ", trait)
    X = ara_data.get_genotype(trait)
    y = ara_data.get_normalised_phenotype(trait) 

 
    ## fit random forest model
    results_linear = []
    results_rf = []
    rf_params = {'Model_Name': 'Random_Forest', 'feature_represenation': 'SNPs', 'variance_maintained': '-'}
    model_params = {'n_estimators':[200], 'max_features':[0.1, 0.5, 0.7, 1], 'n_jobs':[-1]}
    model = GridSearchCV(RandomForestRegressor(), param_grid=model_params, cv=KFold(n_splits=5, shuffle=True), n_jobs=-1)
    result = [(Result_10fold(model, X, y, important_parameters=rf_params, name='rf', cv=cv, n_jobs=-1))]
    result[0].get_indidual_predictions().to_csv(f"{INDIVIDUAL_RESULTS}/{trait}_rf_gs.csv")
    make_results_to_csv(result, RESULTS, '', trait, 'rf_gs')
    

    # fits linear models
    linear_models = ['ridge', 'lasso', 'elasticnet']
    # bounds for the hyperparameters dont have to be the same for all models given I have run this so many times we know for example
    # CV will always choose alpha < 1.5^3 for any model in lasso regression.
    results_linear += test_linear_model(X, y, cv=cv, alphas=[1.5**x for x in range(-45,40)][::-1], n_jobs=-1, name_of_feature_representations='SNPs',
                                    ridge_lasso_enet=[True, False, False])
    results_linear += test_linear_model(X, y, cv=cv, alphas=[1.5**x for x in range(-20,3)][::-1], n_jobs=-1, name_of_feature_representations='SNPs',
                                    ridge_lasso_enet=[False, True, False])
    results_linear += test_linear_model(X, y, cv=cv, alphas=[1.5**x for x in range(-20,20)][::-1], n_jobs=5, name_of_feature_representations='SNPs',
                                    ridge_lasso_enet=[False, False, True])
    
    linear_v = results_linear[-1].make_resuslts_objects_into_vervose_csv(results_linear)
    linear = results_linear[-1].make_resuslts_objects_into_csv(results_linear)
    for linear_result, name, feature in zip(results_linear, linear_models, ["SNPs"]*3):
        linear_result.get_indidual_predictions().to_csv(f"{INDIVIDUAL_RESULTS}/{trait}{name}{feature}.csv")
    linear.to_csv(f"{RESULTS}/{trait}_linear.csv") 
    linear_v.to_csv(f"{RESULTS}/{trait}_linear_verbose.csv")

