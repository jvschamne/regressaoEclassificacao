from random import random
from sklearn import tree
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, mean_squared_error
from math import sqrt
class Tree():
    def classificacaoTree(self, X_train, X_test, y_train, y_test, parametros, i):
        classificador = tree.DecisionTreeClassifier(
            criterion=parametros['criterion'][i],
            max_depth=parametros['max_depth'][i],
            max_leaf_nodes=parametros['max_leaf_nodes'][i],
            min_samples_leaf=parametros['min_samples_leaf'][i],
            min_weight_fraction_leaf=parametros['min_weight_fraction_leaf'][i],
            splitter=parametros['splitter'][i],
            random_state=parametros['random_state'][i]
        )

        #treinamento
        classificador.fit(X_train, y_train)
        
        #validacao
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


    def regressaoTree(self, X_train, X_test, y_train, y_test, parametros, i):
        regressor = tree.DecisionTreeRegressor(
            criterion=parametros['criterion'][i],
            max_depth=parametros['max_depth'][i],
            max_leaf_nodes=parametros['max_leaf_nodes'][i],
            min_samples_leaf=parametros['min_samples_leaf'][i],
            min_weight_fraction_leaf=parametros['min_weight_fraction_leaf'][i],
            splitter=parametros['splitter'][i],
            random_state=parametros['random_state'][i]
        )

        #treinamento da rede
        regressor.fit(X_train, y_train)

        #validacao da rede
        acuracia = regressor.score(X_test, y_test) * 100
        
        predicoes = []
        for x in X_test: 
            predicao = regressor.predict([x])
            predicoes.append(predicao)

        mse = mean_squared_error(y_test, predicoes)
        mse = sqrt(mse)
        #print("Raiz quadrada do Erro quadrativo médio: ",mse)

        metricas = [mse]

        return acuracia, regressor, metricas