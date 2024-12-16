"""
Code added in response to first round of reviewer comments.

Tests the sensistivty to the number of markers used (pre-selected with random forest).

Runs across all modeling inc. MLP
"""




from Main.HelperClasses.GetAraData import *
from Main.HelperClasses.MeasurePerformance import *
from Main.HelperClasses.MLP import *
ara_data = GetAraData(path_to_data='/Research_Data_new/ReasearchProject/data/.', maf=0.05, window_kb=200, r2=0.6)
cv = my_Kfold()
mlp_set_up = {
    'hidden': [1,2,3],'lr': [0.0015], 'weight_decay': [0, 0.01, 0.1], 
    'memory': [10], 'tol': [0.05, 0.01, 0.005], 'drop_out':[0, 0.1], 
    'epoch': [250], 'brach_size': [100], 'hidden_layer_size': [2/3]
    }


def rf_select_tester(model, X, y, features, feature_nums, path, path_ind, trait, model_name):
    """
    Special function to test any model for RF feature selection. note features are preselected.

    `model` str/model object: the model getting tested
    `X` np 2d array: the full matrix of SNPs
    `y` np 1d array: the phenotype
    `features`: a list of np 1d arrays. each array in the list is the indicies of the features that should be used.
    `feature_nums` 1d array. each element if the number of features correponsing to each element of `features`
    `path` str: where the summary results are written
    `path_ind` str: path where the per-accession predictions are written
    `model_name` str: the name of the model being tested
    """
    score_dict = {k:[] for k in feature_nums}    
    score_ind = {k: {'y_test':[], 'y_pred': []} for k in feature_nums}    
    for i, (train, test) in tqdm(list(enumerate(cv.split(X,y)))):
        for feature_num in feature_nums:

            X_train = X[train][:,features[feature_num][i]]
            y_train = y[train]
            X_test = X[test][:,features[feature_num][i]]
            y_test = y[test]

            if model != 'MLP':
                model.fit(X_train, y_train)
                pred = model.predict(X_test)    
            elif model == 'MLP':
                gs = MLP_Grid_search()
                gs.grid_search(mlp_set_up, X_train, y_train)
                pred = gs.predict(X_test)
                pred = torch.Tensor.cpu(pred).detach().numpy().flatten()
            else:
                raise RuntimeError

            score_dict[feature_num].append(r2_score(y_test,pred))
            score_ind[feature_num]['y_test'].extend(y_test)
            score_ind[feature_num]['y_pred'].extend(pred)
            
            pd.DataFrame({'y_test': score_ind[feature_num]['y_test'], 'y_pred': score_ind[feature_num]['y_test']}
            ).to_csv(f'{path_ind}/{trait}.{model_name}.{feature_num}')

        df_v = pd.DataFrame({'feature_num': np.array([[j]*(i+1) for j in feature_nums]).flatten(), 
                'r2s': np.array([score_dict[j] for j in feature_nums]).flatten()})
        df = df_v.groupby('feature_num').mean()
        pd.DataFrame(df_v).to_csv(f'{path}/_{trait}.{model_name}.verbose')
        pd.DataFrame(df).to_csv(f'{path}/{trait}.{model_name}')
        
PATH = 'Main/results/snp_selection'
PATH_IND = 'Main/results/snp_selection/individual'

for trait in list(pd.read_csv('Main/results/traits_used.csv')['name']):
    # precomputes the features selected using random forest for each number of possible features tested.
    features = ara_data.get_rf_features(trait)

for i, trait in  enumerate(list(pd.read_csv('Main/results/traits_used.csv')['name'])):
    print('-'*5 + str(i) + '-'*5)
    X = ara_data.get_genotype(trait)
    y = ara_data.get_normalised_phenotype(trait)
    features = ara_data.get_rf_features(trait)
    model = RidgeCV(alphas = [1.15**x for x in range(-140,140)])
    rf_select_tester(model, X, y, features, ara_data.rf_feature_nums, PATH, PATH_IND, trait, 'Ridge')
    
    model = LassoCV(precompute=False)
    rf_select_tester(model, X, y, features, ara_data.rf_feature_nums, PATH, PATH_IND, trait, 'lasso')
    
    model = RandomForestRegressor(n_estimators=500)
    rf_select_tester(model, X, y, features, ara_data.rf_feature_nums, PATH, PATH_IND, trait, 'rf')

    ratios = [0.01, 0.1, 0.5, 0.85, 0.9, 0.97, 0.99, 0.995]
    enet = GridSearchCV(ElasticNet(n_splits=5, n_jobs=10, max_iter=1000000), param_grid = {'alpha': ratios}, n_jobs=-1)
    rf_select_tester(enet, X, y, features, ara_data.rf_feature_nums, PATH, PATH_IND, trait, 'ElasticNet')
    rf_select_tester('MLP', X, y, features, ara_data.rf_feature_nums, PATH, PATH_IND, trait, 'MLP')
    

