"""
This script fits linear models and Random Forest to SNP
for each trait in the dataset.
"""

from sklearn.ensemble import RandomForestRegressor
from Main.HelperClasses.GetAraData import *
from Main.HelperClasses.MeasurePerformance import *

RESULTS = "Main/results/tree_num_sense/"
ara_data = GetAraData(path_to_data='/Research_Data_new/ReasearchProject/data', maf=0.05, window_kb=200, r2=0.6)

traits =  sorted(ara_data.get_trait_names())

cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42) 
for trait in tqdm(list(pd.read_csv('Main/results/traits_used.csv')['name'])):

    # get data
    print("trait: ", trait)
    X = ara_data.get_genotype(trait)
    y = ara_data.get_normalised_phenotype(trait) 

    result = []
    ## fit random forest model
    for t in [3,5,10,50]:
        print(t, 'trees')
        rf_params = {'Model_Name': 'Random_Forest', 'feature_represenation': 'SNPs', 'variance_maintained': '-', 'tree_num': str(t)}
        result += [(Result_10fold(RandomForestRegressor(n_estimators=t), X, y, important_parameters=rf_params, name='rf', cv=cv, n_jobs=-1))]
        
    make_results_to_csv(result, RESULTS, '', trait, 'rf1')

    