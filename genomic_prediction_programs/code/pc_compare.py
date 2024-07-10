from genomic_prediction_programs.HelperFunctionsAndClasses.GetAraData import *
from genomic_prediction_programs.HelperFunctionsAndClasses.MeasurePerformance import *
import pickle
import pandas as pd
araData = GetAraData(path_to_data='./data', maf=0.05, window_kb=200, r2=0.6)
variance_maintained=0.3

cv = my_Kfold()
failed_lasso = []
for variance_maintained in [0.3,0.05,0.7]:
    pc_r2 = {'trait': [], 'r2_best': [], 'r2s': [], 'model': [], 'num_pc': []}
    for i, trait in list(enumerate(['study_1_FRI', 'study_38_CL', 'study_38_RL', 'study_16_Cd111'])):

        x = araData.get_pca_feature_reduced_SNPs(trait, variance_maintained=variance_maintained)
        y = araData.get_phenotype(trait)

        result = test_ridge_model(x,y, cv=cv, alphas = [2 ** x for x in range(-15,15)], ridge_lasso_enet = [True, False, False], import_parameters={'trait': trait})
        try:
            result += test_ridge_model(x,y, cv=my_Kfold(), alphas = [2 ** x for x in range(-15,15)], ridge_lasso_enet = [False, True, False])
        except:
            failed_lasso.append((i,trait))
        result += test_ridge_model(x,y, cv=my_Kfold(), alphas = [2** x for x in range(-15,15)], ridge_lasso_enet = [False, False, True])
        rf = Result_10fold(RandomForestRegressor(n_estimators=500), x,y, important_parameters={'trait': trait, 'Model_Name': 'rf'}, cv=cv)
        print(result[0].important_parameters)
        r2s = [r.r2 for r in result]+ [rf.r2]
        models = [r.important_parameters['Model_Name'] for r in result] + ['rf']
        r2 = max(r2s)
        ith = np.argmax(r2s)
        pc_r2['trait'] += [trait] 
        pc_r2['r2_best'].append(r2)
        pc_r2['r2s'].append(r2s)
        pc_r2['model'].append(models[ith])
        pc_r2['num_pc'].append(x.shape[1])
        print(pc_r2)
        print(i)
        path = f'/Research_Data_new/ReasearchProject/genomic_prediction_programs/Experiments/Paper/pc{variance_maintained}/'

        pd.DataFrame(pc_r2).to_csv(f'{path}best_pc_model_{variance_maintained}.csv')
print(failed_lasso)
        