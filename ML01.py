#Nicoly Cristina de Jesuis Laurindo
   #RGM 2519155 
   #4ºSemestre'''

# Aula de Machine Learning-Regressão Linear

from sklearn.linear_model import LinearRegression
import numpy as np

# 1) Dados de treino (X = horas de estudo, y = nota na prova)
X = np.array([[1], [2], [3], [4], [5]]) # horas dedicadas de estudo
y = np.array([5, 6, 7, 8, 9]) # previsão da nota

# 2) Criando o modelo de Regressão Linear
modelo = LinearRegression()

# 3) Treinando o modelo de Regressão Linear
modelo.fit(X, y)

# 4) Fazer previsões das notas
horas_estudo = np.array([[4]]) # aluno estudou 4 horas
previsao = modelo.predict(horas_estudo)

print(f"Se o aluno estudar 4 horas, a nota prevista é: {previsao[0]:.2f}")