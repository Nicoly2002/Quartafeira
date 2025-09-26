import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1) Dados de treino
X = np.array([[1], [2], [3], [4], [5]]) # horas dedicadas ao estudo
y = np.array([5, 6, 7, 8, 9]) # notas do aluno

# 2) Treinando o modelo
modelo = LinearRegression()
modelo.fit(X, y)

# 3) Previsões em uma faixa de valores (1 a 6 horas)
X_novo = np.array([[1], [2], [3], [4], [5], [6]])
y_pred = modelo.predict(X_novo)

# 4) Plotar gráfico de Regressão
plt.scatter(X, y, color="blue", label="Dados reais") # pontos reais
plt.plot(X_novo, y_pred, color="red", label="Reta de regressão") # reta aprendida
plt.scatter([6], modelo.predict([[6]]), color="green", label="Previsão (6h)") # previsão

plt.title("Relação:Horas de Estudo X Nota")
plt.xlabel("Horas de Estudo")
plt.ylabel("Nota")
plt.legend()
plt.grid(True)
plt.show()