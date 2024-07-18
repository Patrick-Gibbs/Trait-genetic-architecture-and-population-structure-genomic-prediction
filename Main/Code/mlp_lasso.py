import torch
from sklearn.metrics import r2_score
from Main.HelperClasses.GetAraData import *
from Main.HelperClasses.MeasurePerformance import *
from Main.HelperClasses.TorchMLP import *
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from itertools import product
from copy import deepcopy

RESULTS_PATH = 'Main/results/mlp_lasso_snps'
INDIVDUAL_RESULTS_PATH = 'Main/results/mlp_lasso_snps/individual'

getAraData = GetAraData(path_to_data='./data', maf=0.05, window_kb=200, r2=0.6)

with open(f'{RESULTS_PATH}/features.csv', 'w') as f:
    pass


#for trait in sorted(getAraData.get_filtered_traits()):
file_addon = ''
for trait in ['study_126_Trichome_stem_length']:
    print(trait)
    y = getAraData.get_normalised_phenotype(trait)
    X = getAraData.get_genotype(trait)

    cv = my_Kfold()
    set_up = {'hidden': [1,2,3], 'lr': [0.0015], 'weight_decay': [0, 0.01, 0.1], 
                    'memory': [10], 'tol': [0.05, 0.01, 0.005], 'drop_out':[0, 0.1], 'epoch': [250], 'brach_size': [100], 'hidden_layer_size': [2/3]}

    # Used for testing
    FAST = False
    lasso_space = [2**x for x in range(-7,0)][::-1]
    if FAST:
        lasso_space = [0.1,0.01]
        cv = my_RepeatedKfold(2,1)
        trait = "seed_weight_137"
        set_up = {'hidden': [3],'lr': [0.0015], 'weight_decay': [0], 
                        'memory': [10], 'tol': [0.05], 'drop_out':[0, 0.1], 'epoch': [175], 'brach_size': [100], 'hidden_layer_size': [2/3]}

    def to_cuda(X):
        if type(X) == np.ndarray:
            return torch.from_numpy(X).float().to('cpu')
        else:
            return X

    class MLP_Grid_search:
        def __init__(self, device='cpu', ) -> None:
            self.device = device

        def train(self, X_train, Y_train, device='cpu', epoch=800, brach_size=200, drop_out=0.1, lr=0.001, weight_decay=0.1, memory=10, tol=4, hidden_layer_size=1, hidden=5, feature_selection_alpha=None, features = None):
                X_train = X_train[:, self.features]
                
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
            X = X[:, self.features]
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
            for alpha in lasso_space:
                print('fitting lasso')
                lasso = Lasso(alpha=alpha, max_iter=1000)
                lasso.fit(X, Y)
                try:
                    with open(f'{RESULTS_PATH}/features.csv', 'a') as f:
                        f.write(f'{trait},{alpha},{(sum(lasso.coef_ != 0))}\n')
                except:
                    with open(f'{RESULTS_PATH}/features.csv', 'w') as f:
                        f.write(f'{trait},{alpha},{(sum(lasso.coef_ != 0))}\n')
                self.features = lasso.coef_ != 0
                print('lasso fitted', sum(lasso.coef_ != 0), 'features selected')
                if sum(self.features) < 2:
                    print('skip')
                    continue
                for parameterisation_mlp in tqdm(parameterisations_mlp):
                    parameterisation_mlp['feature_selection_alpha'] = alpha
                    score = []
                    for train_i, test_i in cv.split(X, Y):
                        X_train = X[train_i]
                        X_test = X[test_i]
                        Y_train = Y[train_i]
                        Y_test = Y[test_i]
                        self.train(X_train, Y_train, device='cpu', **parameterisation_mlp)
                        parameterisation_mlp['epoch'] = self.epoch
                        parameterisation_mlp['features'] = self.features
                        X_test = torch.from_numpy(X_test).float().to('cpu')
                        y_hat = self.predict(X_test)
                        score.append(r2_score(Y_test.flatten(), torch.Tensor.cpu(y_hat).detach().numpy().flatten()))
                    parameterisations.append((deepcopy(parameterisation_mlp)))
                    scores.append(np.mean(deepcopy(score)))

            best = np.argmax(scores)
            self.best_params = parameterisations[best]

            lasso = Lasso(alpha=self.best_params['feature_selection_alpha'], max_iter=1000)
            lasso.fit(X_train, Y_train)
            self.features = lasso.coef_ != 0

            self.train(X, Y,device='cpu', **parameterisations[best])

    def test(params, X, Y, cv):
        print('test')
        metrics = {'r2s': [], 'RMSEs':[]}
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

            metrics_copy = metrics.copy()
            metrics_copy['features'] = [sum(e) for e in metrics['features']]
            pd.DataFrame(metrics_copy).to_csv(f'{RESULTS_PATH}/my_lasso_mlp_{trait}_{file_addon}.csv')
            pd.DataFrame({'index': index, 'y_pred': y_pred, 'y_test': y_test}).to_csv(f'{INDIVDUAL_RESULTS_PATH}/my_lasso_mlp_individual_results{trait}_{file_addon}.csv')
        return metrics, pd.DataFrame({'index': index, 'y_pred': y_pred, 'y_test': y_test})

    metrics, indidual_results = test(set_up, X, y, cv)
    metrics['features'] = [sum(e) for e in metrics['features']]
    indidual_results.to_csv(f'{INDIVDUAL_RESULTS_PATH}/_individual_results{trait}_{file_addon}.csv')
    print(metrics)
    pd.DataFrame(metrics).to_csv(f'{RESULTS_PATH}/_{trait}_{file_addon}.csv')