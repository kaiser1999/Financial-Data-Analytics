from sklearn.neural_network import MLPRegressor, MLPClassifier

def ANNet(X, y, size, linout=True, max_iter=1e4, trial=5):
    kwargs = {"hidden_layer_sizes": size, "max_iter": max_iter,
              "solver": "lbfgs", "activation": "logistic"}
    Best_ANN = MLPRegressor(**kwargs) if linout else MLPClassifier(**kwargs)
    Best_ANN.fit(X, y)
    Best_score = Best_ANN.score(X, y)
    for i in range(1, trial):
        model = MLPRegressor(**kwargs) if linout else MLPClassifier(**kwargs)
        model.fit(X, y)
        if model.score(X, y) > Best_score:      # check if improved
            Best_score = model.score(X, y)      # save the best score
            Best_ANN = model		            # save the best model
    return Best_ANN
