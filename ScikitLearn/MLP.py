from email.contentmanager import raw_data_manager
from sklearn.neural_network import MLPClassifier
X = [[0., 0.], [1., 1.]]    #data
y = [0, 1]                  #targets

classificador = MLPClassifier(hidden_layer_sizes=100, activation="relu", solver='lbfgs',alpha=0.0001, random_state=None)
classificador.fit(X, y)
res = classificador.predict([[2., 2.], [-1., -2.]])
print(res)
res = classificador.predict([[1., 2.]])
print(res)
res = classificador.predict([[0., 0.]])
print(res)

#MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1, solver='lbfgs')