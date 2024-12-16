"""
Fits a MLP model to the data using PCA to reduce the number of features. used to making PCA-releated figures.
Code simular to `mlp_lasso.py` but rather uses PCs instead of preselected SNPs
"""


import torch
from sklearn.metrics import r2_score
from Main.HelperClasses.GetAraData import *
from Main.HelperClasses.MeasurePerformance import *
from Main.HelperClasses.TorchMLP import *
from sklearn.model_selection import KFold
from itertools import product
from sklearn.model_selection import train_test_split


RESULTS_PATH = "Main/results/mlp_pca"
INDIVIDUAL = "Main/results/mlp_pca/individual"

getAraData = GetAraData(path_to_data='./data', maf=0.05, window_kb=200, r2=0.6)

for trait in ['study_12_FT10', 'herbavore_resistance_G2P', 'study_4_M130T666']:
    print(trait)
    y = getAraData.get_normalised_phenotype(trait)

    cv = my_Kfold()
    set_up = {'hidden': [1,2,3],'lr': [0.0015], 'weight_decay': [0, 0.01, 0.1], 
                    'memory': [10], 'tol': [0.05, 0.01, 0.005], 'drop_out':[0, 0.1], 'epoch': [250], 'brach_size': [100], 'hidden_layer_size': [2/3]}

    # Used for testsing
    file_addon = ''
    def to_cuda(X):
        if type(X) == np.ndarray:
            return torch.from_numpy(X).float().to('cpu')
        else:
            return X

    class MLP_Grid_search:
        def __init__(self, device='cpu', ) -> None:
            self.device = device

        def train(self, X_train, Y_train, device='cpu', epoch=800, brach_size=200, drop_out=0.1, lr=0.001, weight_decay=0.1, memory=10, tol=0.05, hidden_layer_size=1, hidden=5):
                
                X_train = torch.from_numpy(X_train).float()
                Y_train = torch.from_numpy(Y_train).float()

                hidden_layer_size = int(X_train.shape[1] * hidden_layer_size)

                model = MLP_complex(input_dim=X_train.shape[1], dropout=drop_out, activation=torch.nn.ReLU(), hidden_layer_size=hidden_layer_size, hidden=hidden)
                model.to('cpu')
                
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                criterion = torch.nn.MSELoss()

        
                # make split for validatoin
                early_stopping_losses = []
                X_train, X_validate, Y_train, Y_validate = train_test_split(X_train, Y_train, test_size=0.15)
                ### FIT MODEL ###
                for epoch in range(epoch):
                    losses_validate_e = []
                    for batch in range(0, len(X_train), brach_size-1):
                        x_train = X_train[batch:batch+brach_size]
                        y_train = Y_train[batch:batch+brach_size]
                        x_val = X_validate
                        y_val = Y_validate
                        # Forward pass
                        y_pred = model.forward(x_train)
                        loss = criterion(y_pred.flatten(), y_train.flatten())

                        losses_validate_e.append(criterion(model.forward(x_val).flatten(), y_val.flatten()).item())
                        # Backward pass
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    # shuffle data
                    perm = torch.randperm(X_train.shape[0])
                    X_train = X_train[perm]
                    Y_train = Y_train[perm]
                    self.epoch = epoch
                    
                    if len(early_stopping_losses) < memory + 1:
                        early_stopping_losses.append(np.mean(losses_validate_e))
                    else:
                        early_stopping_losses.pop(0)
                        early_stopping_losses.append(np.mean(losses_validate_e))
                        if not all(early_stopping_losses[i] - i * tol > early_stopping_losses[i+1] - (i+1)*tol for i in range(len(early_stopping_losses)-1)):
                            break
            
                self.model = model

        def predict(self, X):
            if type(X) == np.ndarray:
                X = torch.from_numpy(X).float().to(self.device)
            return self.model.forward(X).flatten()

        def grid_search(self, params: dict, X, Y, cv=KFold(n_splits=5, random_state=42, shuffle=True)):
            # make grid search
            param_tups = list(product(*params.values()))
            param_names = list(params.keys())
            parameterisations_mlp = ([{k:v for k,v in zip(param_names, param_tup)} for param_tup in param_tups])
            parameterisations = []
            scores = []
            print(len(parameterisations))
        
            for parameterisation_mlp in tqdm(parameterisations_mlp):
                parameterisations.append(parameterisation_mlp)
                score = []
                for train_i, test_i in cv.split(X, Y):
                    X_train = X[train_i]
                    X_test = X[test_i]
                    Y_train = Y[train_i]
                    Y_test = Y[test_i]
                    self.train(X_train, Y_train, device='cpu', **parameterisation_mlp)
                    parameterisation_mlp['epoch'] = self.epoch
                    X_test = torch.from_numpy(X_test).float().to('cpu')
                    y_hat = self.predict(X_test)
                    score.append(r2_score(Y_test.flatten(), torch.Tensor.cpu(y_hat).detach().numpy().flatten()))
                scores.append(np.mean(score))

            best = np.argmax(scores)
            self.best_params = parameterisations[best]

            self.train(X, Y,device='cpu', **parameterisations[best])

    def test(params, X, Y, cv):
        print('test')
        metrics = {'r2s': [], 'RMSEs':[], 'features':[]}
        i = 0
        y_pred = []
        y_test = []
        index = []
        for train_i, test_i in cv.split(X, Y):
            i += 1
            X_train = X[train_i]
            X_test = X[test_i]
            Y_train = Y[train_i]
            Y_test = Y[test_i]

            gs = MLP_Grid_search()
            gs.grid_search(params, X_train, Y_train)


            y_hat = gs.predict(X_test)
            y_hat = torch.Tensor.cpu(y_hat).detach().numpy().flatten()

            y_pred.extend(y_hat)
            y_test.extend(Y_test.flatten())
            index.extend(test_i)


            metrics['r2s'].append(r2_score(Y_test.flatten(), y_hat))
            metrics['RMSEs'].append(np.sqrt(mean_squared_error(Y_test.flatten(), y_hat)))
            for para in gs.best_params.keys():
                if para not in metrics:
                    metrics[para] = [gs.best_params[para]]
                else:
                    metrics[para].append(gs.best_params[para])

            metrics['features'].append(X.shape[1])

        return metrics, {'index': index, 'y_pred': y_pred, 'y_test': y_test}


    all_metrics = []
    all_indivual_results = []

    for pc_var in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 1]:
        X = getAraData.get_pca_feature_reduced_SNPs(trait, variance_maintained=pc_var)
        print(X.shape)
        metrics, indidual_results = test(set_up, X, y, cv)
        print(metrics)
        metrics['variance_maintained'] = pc_var
        all_metrics.append(pd.DataFrame(metrics))
        all_indivual_results.append(pd.DataFrame(indidual_results))



        # concat all results
        pd.concat(all_indivual_results, join='outer', ignore_index=True).to_csv(f'{INDIVIDUAL}/individual_results{trait}_pc{pc_var}.csv')
        pd.concat(all_metrics, join='outer', ignore_index=True).to_csv(f'{INDIVIDUAL}.csv')
