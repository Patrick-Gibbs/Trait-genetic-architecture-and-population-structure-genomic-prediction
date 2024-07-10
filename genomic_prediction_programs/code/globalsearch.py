"""
This script fits linear models and Random Forest to SNP
for each trait in the dataset.
"""


from genomic_prediction_programs.HelperFunctionsAndClasses.GetAraData import *
from genomic_prediction_programs.HelperFunctionsAndClasses.MeasurePerformance import *

INDIVIDUAL_RESULTS = 'genomic_prediction_programs/Experiments/Paper/results_indivdual'
RESULTS = "genomic_prediction_programs/Experiments/Paper/results/"
ara_data = GetAraData(path_to_data='./data', maf=0.05, window_kb=200, r2=0.6)
traits = ['dummy'] + sorted(ara_data.get_trait_names())
cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42)
    
for trait in tqdm(traits):

    # get data
    print("trait: ", trait)
    X = ara_data.get_genotype(trait)
    y = ara_data.get_normalised_phenotype(trait) 

 
    # fit random forest model
    results_linear = []
    results_rf = []
    rf_params = {'Model_Name': 'Random_Forest_vanilla', 'feature_represenation': 'SNPs', 'variance_maintained': '-'}
    result = [(Result_10fold(RandomForestRegressor(n_estimators=500), X, y, important_parameters=rf_params, name='rf_vanilla', cv=cv, n_jobs=-1))]
    result[0].get_indidual_predictions().to_csv(f"{INDIVIDUAL_RESULTS}/{trait}rf_vanilla_500.csv")
    make_results_to_csv(result, RESULTS, '', trait, 'rf_vanilla_500')
    

    # fits r1 model
    r1_params = {'Model_Name': 'r1_tree', 'feature_represenation': 'SNPs', 'variance_maintained': '-'}
    result = [(Result_10fold(DecisionTreeRegressor(max_depth=1), X, y, important_parameters=r1_params, name='r1_tree', cv=cv, n_jobs=-1))]
    result[0].get_indidual_predictions().to_csv(f"{INDIVIDUAL_RESULTS}/{trait}r1_tree.csv")
    make_results_to_csv(result, RESULTS, '', trait, 'r1_tree')


    # fits linear models
    linear_models = ['ridge', 'lasso', 'elasticnet']
    # bounds for the hyperparameters dont have to be the same for all models
    results_linear += test_ridge_model(X, y, cv=cv, alphas=[2**x for x in range(-10,20)][::-1], n_jobs=-1, name_of_feature_representations='SNPs',
                                    ridge_lasso_enet=[True, False, False])
    results_linear += test_ridge_model(X, y, cv=cv, alphas=[2**x for x in range(-20,1)][::-1], n_jobs=-1, name_of_feature_representations='SNPs',
                                    ridge_lasso_enet=[False, True, False])
    results_linear += test_ridge_model(X, y, cv=cv, alphas=[2**x for x in range(-15,15)][::-1], n_jobs=6, name_of_feature_representations='SNPs',
                                    ridge_lasso_enet=[False, False, True])
    
    linear_v = results_linear[-1].make_resuslts_objects_into_vervose_csv(results_linear)
    linear = results_linear[-1].make_resuslts_objects_into_csv(results_linear)

    for linear_result, name, feature in zip(results_linear, linear_models, ["SNPs"]*3):
        linear_result.get_indidual_predictions().to_csv(f"{INDIVIDUAL_RESULTS}/{trait}{name}{feature}.csv")

    linear.to_csv(f"{RESULTS}/{trait}_results.csv")
    linear_v.to_csv(f"{RESULTS}/{trait}_results_v.csv")
    print("done")