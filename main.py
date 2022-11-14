from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# lê a base de dados
df = pd.read_csv('https://raw.githubusercontent.com/SeniorSA/seniorlabs-challenge/main/sms_senior.csv', 
                 encoding= 'unicode_escape')
# df.head()

#1
colunas = df.columns.values.tolist()
#2
tipos_numericos = ['float64', 'int64']

nome_colunas_numericas = []

for coluna in colunas:
    tipo = df[coluna].dtype
    if tipo in tipos_numericos:
      nome_colunas_numericas.append(coluna)

x = df[nome_colunas_numericas]

y = df['IsSpam']

modelo = LinearSVC(max_iter=100000)

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y)

modelo.fit(treino_x, treino_y)

previsao = modelo.predict(teste_x)

accuracy_score(teste_y, previsao) * 100
