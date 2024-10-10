from Main.HelperClasses.GetAraData import *
from Main.HelperClasses.MeasurePerformance import *
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from tqdm import tqdm


OTHER_INFO = ""
jobs = -1

# directory to store the results generated.
RESULTS_DIR = "Main/results/rf_pca_sup/"
INDIVIDUAL = "Main/results/rf_pca_sup/individual/"
PC_VARIANCES_TO_TEST = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1]

getAraData = GetAraData(path_to_data='/Research_Data_new/ReasearchProject/data/.', maf=0.05, window_kb=200, r2=0.6)
cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42)
# traits to be tested

traits = list(pd.read_csv('Main/results/traits_used.csv')['name'])[::-1]

for trait in (traits):
    print(trait)
    y = getAraData.get_normalised_phenotype(trait)
    results = []
    for i in tqdm(range(len(PC_VARIANCES_TO_TEST))):
        # it is impossible to fit a MLP to millions of SNPs with with my archecture 
        jobs=-1
        X = getAraData.get_pca_feature_reduced_SNPs(trait, variance_maintained=PC_VARIANCES_TO_TEST[i])
        varience_mained_in_PCA = PC_VARIANCES_TO_TEST[i]
        name_of_feature_representations = 'PCA_SNPs'

        model = RandomForestRegressor(n_estimators=500, n_jobs=jobs)
        important_parameters = {'variance_maintained': varience_mained_in_PCA, 'feature_represenation': name_of_feature_representations, 'trait': trait, 'Model_Name': 'rf'}
        
        results.append(Result_10fold(model, X=X,y=y, important_parameters=important_parameters,
                                    cv=cv, n_jobs=-1))
        
        make_results_to_csv(results, RESULTS_DIR, OTHER_INFO, trait, 'rf')
        results[-1].get_indidual_predictions().to_csv(f"{INDIVIDUAL}{trait}_rf_{OTHER_INFO}_individual_pca{(varience_mained_in_PCA)}_sup.csv")

