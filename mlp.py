from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, mean_squared_error
from time import sleep 
from math import sqrt

class MLP():
    def classificacaoMLP(self, X_train, X_test, y_train, y_test, parametros, i):
        classificador = MLPClassifier(
            hidden_layer_sizes=parametros["hidden_layer_sizes"][i],
            activation=parametros["activation"][i],
            solver=parametros["solver"][i],
            alpha=parametros["alpha"][i],
            random_state=None,
            max_iter=2000,
            learning_rate_init=0.0001)
    
        #treinamento da rede
        classificador.fit(X_train, y_train)

        #validacao da rede
        acuracia = classificador.score(X_test, y_test) * 100


        predicoes = []
        for x in X_test: 
            predicao = classificador.predict([x])
            predicoes.append(predicao)

        matrixConfusao = confusion_matrix(y_test, predicoes)
        #print("Matriz de confusão: ",matrixConfusao)
       
        prf = precision_recall_fscore_support(y_test, predicoes, average='weighted')
        #print("Precisio, recall e fscore:", prf)

        metricas = [matrixConfusao, prf]

        return acuracia, classificador, metricas


    def regressaoMLP(self, X_train, X_test, y_train, y_test, parametros, i):    
        regressor = MLPRegressor(
            hidden_layer_sizes=parametros["hidden_layer_sizes"][i],
            activation=parametros["activation"][i],
            solver=parametros["solver"][i],
            alpha=parametros["alpha"][i],
            random_state=None,
            max_iter=2000,
            learning_rate_init=0.0001)
    
        #treinamento da rede
        regressor.fit(X_train, y_train)

        #validacao da rede
        acuracia = regressor.score(X_test, y_test) * 100
        #print(acuracia)

        predicoes = []
        for x in X_test: 
            predicao = regressor.predict([x])
            predicoes.append(predicao)

        mse = mean_squared_error(y_test, predicoes)
        mse = sqrt(mse)
        print("Raiz quadrada do Erro quadrativo médio: ",mse)

        metricas = [mse]

        return acuracia, regressor, metricas