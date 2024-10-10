



pipeline = Pipeline([
    ('feature_selection', SelectFromModel(rf, threshold=-np.inf, max_features=n_features_to_select)),
    ('model', MLPClassifier(hidden_layer_sizes=(5,), random_state=42))
])

from Main.HelperClasses.GetAraData import *
from Main.HelperClasses.MeasurePerformance import *
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.dummy import DummyRegressor
import pandas as pd
from tqdm import tqdm


ara_data = GetAraData(path_to_data='/Research_Data_new/ReasearchProject/data/.', maf=0.05, window_kb=200, r2=0.6)

cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42)
FEATURES = [1,2,3,4,5,10,50,100,500,1000]

RESULTS_DIR = "Main/results/linear_pca_sup"
INDIVIDUAL = 'Main/results/linear_pca_sup/individual'

for trait in list(pd.read_csv('Main/results/traits_used.csv')['name']):
    trait_results = []
    X = ara_data.get_genotype(trait)
    y = ara_data.get_normalised_phenotype(trait)
    for num_features in tqdm(FEATURES):
        print(trait, num_features)
        models = []
        ...
        pipeline = Pipeline([
            ('feature_selection', SelectFromModel(rf, threshold=-np.inf, max_features=n_features_to_select)),
            ('model', MLPClassifier(hidden_layer_sizes=(5,), random_state=42))
        ])
        trait_results += test_linear_model(X, y, cv=cv, alphas= [1.5**x for x in range(-20,40)], n_jobs=-1, pca_variance=num_features, name_of_feature_representations='PCA_SNPS'
                                          , ridge_lasso_enet=[True, False, False])
        

        trait_results += test_linear_model(X, y, cv=cv, alphas=None, n_jobs=-1, pca_variance=num_features, name_of_feature_representations='PCA_SNPS',
                                           ridge_lasso_enet=[False, True, False], lasso_tol=0.01, lasso_cv=my_Kfold(k=5,state=2), precomute_gram=False)
        
        
        trait_results += test_linear_model(X, y, cv=cv, alphas= [1.5**x for x in range(-15,20)], n_jobs=-1, pca_variance=num_features, name_of_feature_representations='PCA_SNPS',
                                            ridge_lasso_enet=[False, False, True])

       
        make_results_to_csv(trait_results, RESULTS_DIR, '', trait, 'linear')
        trait_results[-1].get_indidual_predictions().to_csv(f"{INDIVIDUAL}/{trait}_linear_{'PCA_SNPS'}_individual_select{(str(num_features))}.csv")
        