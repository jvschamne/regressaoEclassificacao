import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#dados e targets
data = [[0, 0], [1, 1], [2, 2]]
targets = [0, 1, 2]

#executando a regressao linear
linearReg = LinearRegression()
linearReg.fit(data, targets) #fit(X, y) -> X = dados, y = targets
print(linearReg.coef_)


xpoints = np.array([0, 1, 2])
ypoints = np.array([0, 1, 2])

plt.plot(xpoints, ypoints)
plt.show()