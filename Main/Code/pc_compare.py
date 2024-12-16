"""
Generates results across each model using PCs describing 0.3 of marker variance (code orginally used before first submission), now superseaded by linear_pca.py / rf_pca.py
"""


from Main.HelperClasses.GetAraData import *
from Main.HelperClasses.MeasurePerformance import *
from sklearn.ensemble import RandomForestRegressor
import pickle
import pandas as pd
araData = GetAraData(path_to_data='./data', maf=0.05, window_kb=200, r2=0.6)
variance_maintained=0.3

cv = my_Kfold()
failed_lasso = []
for variance_maintained in [0.3]:
    pc_r2 = {'trait': [], 'r2_best': [], 'r2s': [], 'model': [], 'num_pc': []}
    for i, trait in enumerate(list(pd.read_csv('Main/results/traits_used.csv')['name'])):
        print(trait)
        x = araData.get_pca_feature_reduced_SNPs(trait, variance_maintained=variance_maintained)
        print(x.shape)
        y = araData.get_normalised_phenotype(trait)

        print(x.shape)
        result = test_linear_model(x,y, cv=cv, alphas = [1.5 ** x for x in range(-30,45)], ridge_lasso_enet = [True, False, False], import_parameters={'trait': trait})
        result += test_linear_model(x,y, cv=my_Kfold(), alphas = [1.5 ** x for x in range(-20,20)][::-1], ridge_lasso_enet = [False, True, False],
                                   precomute_gram=False)
        result += test_linear_model(x,y, cv=my_Kfold(), alphas = [1.5** x for x in range(-15,15)], ridge_lasso_enet = [False, False, True])
        rf = Result_10fold(RandomForestRegressor(n_estimators=500), x, y, important_parameters={'trait': trait, 'Model_Name': 'rf'}, cv=cv)

        r2s = [r.r2 for r in result]+ [rf.r2]
        models = [r.important_parameters['Model_Name'] for r in result] + ['rf']
        r2 = max(r2s)
        print(models)
        ith = np.argmax(r2s)
        print(ith)
        pc_r2['trait'] += [trait] 
        pc_r2['r2_best'].append(r2)
        pc_r2['r2s'].append(r2s)
        pc_r2['model'].append(models[ith])
        pc_r2['num_pc'].append(x.shape[1])

        path = f'Main/results/best_model_pca_{variance_maintained}_var/'
        pd.DataFrame(pc_r2).to_csv(f'{path}1')
        