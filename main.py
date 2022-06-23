import os, sys
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from random import randint
from mlp import MLP
from tree import Tree
from sklearn.metrics import confusion_matrix

def validacaoCruzada(n, k, sinais, tecnica, parametros, iteracao):#validacao cruzada k-folds
    mlPerceptron = MLP()
    arvoreIndutiva = Tree()
    acuracias = 0 #acuracia de cada repetição
    modelo = None

    for i in range (0, n):#repete n vezes
            X_train, X_test, y_train, y_test = randomLines(k, sinais, tecnica)
            acuracia = None
            if tecnica == "regressaoMLP":
                acuracia, modelo, metricas  = mlPerceptron.regressaoMLP(X_train, X_test, y_train, y_test, parametros, iteracao)
            elif tecnica == "classificacaoMLP":
                acuracia, modelo, metricas = mlPerceptron.classificacaoMLP(X_train, X_test, y_train, y_test, parametros, iteracao)
            elif tecnica == "regressaoTree":
                acuracia, modelo, metricas  = arvoreIndutiva.regressaoTree(X_train, X_test, y_train, y_test, parametros, iteracao)
            elif tecnica == "classificacaoTree":
                acuracia, modelo, metricas = arvoreIndutiva.classificacaoTree(X_train, X_test, y_train, y_test, parametros, iteracao)

            acuracias += acuracia

    acuraciaFinal = acuracias/n
        
    return acuraciaFinal, modelo, metricas
        

def randomLines(k, sinais, tecnica):#escolhe linhas do dataset aleatoriamente
    """
        selecionar linhas aleatorias do dataset para treinar e validar o modelo
    """
    quantdTreinar = (k-1)/k       #quantidade de linhas usadas para treinar
    X = []
    y = []

    if tecnica == "regressaoMLP" or tecnica == "regressaoTree":
        for s in sinais:
            dados = [s[0], s[1], s[2]]
            X.append(dados)
            y.append(s[3])
    elif tecnica == "classificacaoMLP" or tecnica == "classificacaoTree":
        for s in sinais:
            X.append([s[3]])
            y.append(s[4])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=quantdTreinar, random_state=42)

    return  X_train, X_test, y_train, y_test


def treinaValidaMLP(sinais):
    """melhores parametros para regressor MLP = 75% acurácia
            'hidden_layer_sizes': [(100), (100), (100), (100)],
            'activation': ['tanh', 'tanh', 'tanh', 'tanh'],
            'solver': ['sgd', 'lbfgs', 'lbfgs', 'lbfgs'],
            'alpha': [0.05, 0.05, 0.05, 0.05]
    """
    #PARAMETROS:
    parametros = {
        'hidden_layer_sizes': [(100), (100), (100), (100)],
            'activation': ['tanh', 'relu', 'tanh', 'tanh'],
            'solver': ['lbfgs', 'lbfgs', 'adam', 'adam'],
            'alpha': [0.05, 0.001, 0.01, 0.001]
    }
    acuraciasReg = []
    acuraciasClass = []
    melhorRegressor = None
    melhorClassificador = None
    melhorResultado = 0
    metricasClassificador = []
    metricasRegressor = []

    for i in range(0, 4):
        acuraciaFinal, MLPRegressor, metricas = validacaoCruzada(n=2, k=4, sinais=sinais, tecnica="regressaoMLP", parametros=parametros, iteracao=i)
        acuraciasReg.append(acuraciaFinal)
        
        if melhorResultado == 0: 
            melhorRegressor = MLPRegressor
            melhorResultado = acuraciaFinal
            metricasRegressor = metricas
    
        elif acuraciaFinal > melhorResultado and acuraciaFinal < 90:
            melhorRegressor = MLPRegressor
            melhorResultado = acuraciaFinal
            metricasRegressor = metricas

        elif melhorResultado > 90: 
            melhorRegressor = MLPRegressor
            melhorResultado = acuraciaFinal
            metricasRegressor = metricas
        
            
    acuraciaR = melhorResultado

    parametros = {
        'hidden_layer_sizes': [(100), (100), (100), (100)],
        'activation': ['tanh', 'relu', 'relu', 'tanh'],
        'solver': ['lbfgs', 'lbfgs', 'adam', 'adam'],
        'alpha': [0.01, 0.001, 0.001, 0.05]
    }

    melhorResultado = 0
    for i in range(0, 4):
        acuraciaFinalC, MLPClassificador, metricas = validacaoCruzada(n=2, k=4, sinais=sinais, tecnica="classificacaoMLP", parametros=parametros, iteracao=i)
        acuraciasClass.append(acuraciaFinalC)
    
        if melhorResultado == 0: 
            melhorClassificador = MLPClassificador
            melhorResultado = acuraciaFinalC
            metricasClassificador = metricas
        elif acuraciaFinalC > melhorResultado and acuraciaFinalC < 90:
            melhorClassificador = MLPClassificador
            melhorResultado = acuraciaFinalC
            metricasClassificador = metricas
        elif melhorResultado > 90: 
            melhorClassificador = MLPClassificador
            melhorResultado = acuraciaFinalC
            metricasClassificador = metricas
        
            
    acuraciaC = melhorResultado
    #print("Acuracia Regressor=",acuraciaR,"%, Acuracia Classificador=", acuraciaC,"%")    
    print(acuraciasReg)
    print(acuraciasClass)
    
    return acuraciaR, acuraciaC, melhorRegressor, melhorClassificador, metricasRegressor,metricasClassificador


def treinaValidaTree(sinais):
    """melhores parametros classificador: 100% acurácia p/o classificador
            'max_depth': [5, 10],
            'max_features': ['sqrt', 'log2'],
            'max_leaf_nodes': [50, 10],
            'min_samples_leaf': [2, 5],
            'min_weight_fraction_leaf': [0.1, 0.05],
            'splitter': ['best', 'best']
    """
    #PARAMETROS:
    parametros = {
        'criterion':['poisson', 'poisson', 'poisson', 'poisson','poisson'],
        'max_depth': [8, 8, 8, 8, 8],
        'max_leaf_nodes': [80, 80, 80, 80, 80],
        'min_samples_leaf': [5, 5, 5, 5, 5], 
        'min_weight_fraction_leaf': [0.0001, 0.001, 0.005, 0.0001, 0.0001],
        'splitter': ['best', 'best', 'best', 'best', 'best'],
        'random_state': [24,10,5,10,24]
    }
    acuraciasReg = []
    acuraciasClass = []
    melhorRegressor = None
    melhorClassificador = None
    melhorResultado = 0
    metricasRegressor = []
    metricasClassificador = []
    
    for i in range(0, 4):
        acuraciaFinal, treeRegressor, metricas = validacaoCruzada(n=2, k=4, sinais=sinais, tecnica="regressaoTree", parametros=parametros, iteracao=i)
        #print("Acuracia final={0:.2f}%".format(acuraciaFinal))
        acuraciasReg.append(acuraciaFinal)

        if melhorResultado == 0: 
            melhorRegressor = treeRegressor
            melhorResultado = acuraciaFinal
            metricasRegressor = metricas
    
        elif acuraciaFinal > melhorResultado and acuraciaFinal < 90:
            melhorRegressor = treeRegressor
            melhorResultado = acuraciaFinal
            metricasRegressor = metricas

        elif melhorResultado > 90: 
            melhorRegressor = treeRegressor
            melhorResultado = acuraciaFinal
            metricasRegressor = metricas
            

    acuraciaR = melhorResultado

    parametros = {
        'criterion':['gini', 'entropy', 'log_loss','gini', 'entropy'],
        'max_depth': [8, 8, 8, 8, 8],
        'max_leaf_nodes': [5, 20, 15, 20, 20],
        'min_samples_leaf': [5, 2, 5, 5, 2], 
        'min_weight_fraction_leaf': [0.0001, 0.005, 0.001, 0.001, 0.0001],
        'splitter': ['random', 'random', 'random', 'random', 'random'],
        'random_state': [24,20,20,48,25]
    }

    melhorResultado = 0
    for i in range(0, 4):
        acuraciaFinalC, treeClassificador, metricas = validacaoCruzada(n=2, k=4, sinais=sinais, tecnica="classificacaoTree", parametros=parametros, iteracao=i)
        #print("Acuracia final={0:.2f}%".format(acuraciaFinalC))
        acuraciasClass.append(acuraciaFinalC)
    
        if melhorResultado == 0: 
            melhorClassificador = treeClassificador
            melhorResultado = acuraciaFinalC
            metricasClassificador = metricas
        elif acuraciaFinalC > melhorResultado and acuraciaFinalC < 90:
            melhorClassificador = treeClassificador
            melhorResultado = acuraciaFinalC
            metricasClassificador = metricas
        elif melhorResultado > 90: 
            melhorClassificador = treeClassificador
            melhorResultado = acuraciaFinalC
            metricasClassificador = metricas

    acuraciaC = melhorResultado
    #print("Acuracia Regressor=",acuraciaR,"%, Acuracia Classificador=", acuraciaC,"%") 
    print(acuraciasReg)
    print(acuraciasClass)

    return acuraciaR, acuraciaC, melhorRegressor, melhorClassificador, metricasRegressor,metricasClassificador


def testeCego(sinais, melhorRegressor, melhorClassificador):
    """
    Para cada exemplo do teste cego, o programa deverá gerar um vetor de 3 colunas por n linhas
    separados por vírgulas com os resultados de predição numérica da gravidade e da classe
    """
    matrix = []
    sys.path.append('sinais')
    arq = open(os.path.join("sinais","sinaisvitais_teste.txt"),"r")
    for line in arq:    
        matrix.append(line.split(","))

    sinais = []
    for array in matrix:
            dados = [float(array[0]), float(array[1]), float(array[2]), float(array[3])]
            sinais.append(dados)

    resultados = []

    for s in sinais:
        gravidade = melhorRegressor.predict([[s[1], s[2], s[3]]])
        classe = melhorClassificador.predict([gravidade])

        resultados.append([s[0], gravidade, classe])

    for r in resultados:
        print(r)



def main():
    print("-----------INICIO-----------")
    matrix = []
    sys.path.append('sinais')
    arq = open(os.path.join("sinais","sinaisvitais_hist.txt"),"r")
    for line in arq:    
        matrix.append(line.split(","))

    sinais = []
    for array in matrix:
            dados = [float(array[3]), float(array[4]), float(array[5]), float(array[6]), float(array[7])]
            sinais.append(dados)
    
    acuraciaRMLP, acuraciaCMLP, bestRegMLP, bestClassMLP, metricasReg, metricasClass = treinaValidaMLP(sinais)
    acuraciaRTree, acuraciaCTree, bestRegTree, bestClassTree, metricasRegTree, metricasClassTree = treinaValidaTree(sinais)
    
    print(metricasClass)

    print("\n---------ACURACIAS---------")
    print("Metricas melhor Regressor MLP:")
    print("Acuracia={}%\nRMSE={}".format(acuraciaRMLP, metricasReg[0]))
    
    
    print("\nMetricas melhor Classificador MLP:")
    print("Acuracia={}%\nMatriz de confusão=\n{}\nPrecision, recall, f-score:{}".format(acuraciaCMLP, metricasClass[0], metricasClass[1]))
    
    print("\nMetricas melhor Regressor Tree:")
    print("Acuracia={}%\nRMSE={}".format(acuraciaRTree, metricasRegTree[0]))
    
    
    print("\nMetricas melhor Classificador Tree:")
    print("Acuracia={}%\nMatriz de confusão=\n{}\nPrecision, recall, f-score:{}".format(acuraciaCTree, metricasClassTree[0], metricasClassTree[1]))

    melhorRegressor = None
    melhorClassificador = None

    if(acuraciaRMLP > acuraciaRTree):
        melhorRegressor = bestRegMLP
    else:
        melhorRegressor = bestRegTree
    if(acuraciaCMLP > acuraciaCTree):
        melhorClassificador = bestClassMLP
    else:
        melhorClassificador = bestClassTree

    #print(melhorRegressor, melhorClassificador)
    #testeCego(sinais, melhorRegressor, melhorClassificador)
   
if __name__ == '__main__':
    main()