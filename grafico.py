import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1) Dados de treino
X = np.array([[1], [2], [3], [4], [5]]) # horas didelíceas ao estudo
y = np.array([5, 6, 7, 8, 9]) # notas do aluno

# 2) Treinando o modelo
modelo = LinearRegression()
modelo.fit(X, y)

# 3) Previúdos em uma faixa de valores (1 a 4 horas)
X_nova = np.array([[1], [2], [3], [4], [5], [6]])
y_pred = modelo.predict(X_nova)

# 4) Plotar gráfico de Regressão
plt.scatter(X, y, color='blue', label="Dados reais") # pontos reais
plt.plot(X_nova, y_pred, color='red', label="Nota de regressão") # reta aprendida
plt.scatter([6], modelo.predict([[6]]), color='green', label="Previúdo (dh)") # previúdo

plt.title("Relação:Horas de Estudo X Nota")
plt.xlabel("Horas de Estudo")
plt.ylabel("Nota")
plt.legend()
plt.grid(True)
plt.show()