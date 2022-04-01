# Projeto-de-Machine-Learning-Preven-o-de-doen-as-renais-de-Lucas-Alves-
# Importando Pacotes e Bibliotecas:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Conectando com os Dados:
dataset = pd.read_csv("Kidney_data.csv")

# Visualisando as 5 primeiras linhas
dataset.head(100)

# Verificando a Shape (dimensões) do dataset:
dataset.shape

#Verificando informações adicionais do dataset
dataset.info()

# Verificando se há valores Missing (valores ausentes) dos pacientes:
dataset.isnull().sum()

#Verificando se há linhas duplicadas
dataset.duplicated().sum()
dataset.info()

# Estatística Descritiva das Variáveis:
dataset.describe()
dataset.info()

#Variável Target (teve ou não doença nos rins)
# Tabela de Frequência da Variável "classification" - Nossa Classe ou Label ou Target ou Y ou Variável a ser Predita (o que a gente quer descobrir)
dataset.classification.value_counts()

#Criando Gráfico de Barras para variáveis Categóricas
sns.countplot(dataset['classification'])

Variável "rbc" (Glóbulos Vermelhos)
# Tabela de Frequência da Variável "rbc" 
dataset.rbc.value_counts()

# Criando Gráfico de Barras para variáveis Categóricas
sns.countplot(dataset['rbc'])

# Tabela de Frequência da Variável "ba" 
dataset.ba.value_counts()

# Criando Gráfico de Barras para variáveis Categóricas
sns.countplot(dataset['ba'])

# Tabela de Frequência da Variável "pc" 
dataset.pc.value_counts()

# Criando Gráfico de Barras para variáveis Categóricas
sns.countplot(dataset['pc'])

# Tabela de Frequência da Variável "ba" 
dataset.age.value_counts()

# Tabela de Frequência da Variável "rbc" 
dataset.rbc.value_counts()

# Criando Gráfico de Barras para variáveis Categóricas
sns.countplot(dataset['rbc'])

# Tabela de Frequência da Variável "ba" 
dataset.ba.value_counts()

# Criando Gráfico de Barras para variáveis Categóricas
sns.countplot(dataset['ba'])

# Tabela de Frequência da Variável "pc" 
dataset.pc.value_counts()

# Criando Gráfico de Barras para variáveis Categóricas
sns.countplot(dataset['pc'])

# Tabela de Frequência da Variável "ba" 
dataset.age.value_counts()

#Criando Gráfico de Distribuição para variáveis contínuas
sns.histplot(dataset['age'], bins=50, kde=True)

Pré-Processamento dos Dados

# Tipo das Variáveis (numeros ou texto?):
dataset.dtypes

# Eliminando Variáveis desnecessárias (CPF não é explicativa) :
dataset = dataset.drop('id', axis=1)

Substituindo Valores categóricos (object) em números:

# Tabela de Frequência
dataset['rbc'].value_counts()
dataset['rbc'] = dataset['rbc'].replace(to_replace = {'normal' : 0, 'abnormal' : 1})

# Tabela de Frequência
dataset['rbc'].value_counts()

# Tabela de Frequência
dataset['pc'].value_counts()
dataset['pc'] = dataset['pc'].replace(to_replace = {'normal' : 0, 'abnormal' : 1})

# Tabela de Frequência
dataset['pc'].value_counts()

# Tabela de Frequência
dataset['pcc'].value_counts()
dataset['pcc'] = dataset['pcc'].replace(to_replace = {'notpresent':0,'present':1})

# Tabela de Frequência
dataset['pcc'].value_counts()

# Tabela de Frequência
dataset['ba'].value_counts()
dataset['ba'] = dataset['ba'].replace(to_replace = {'notpresent':0,'present':1})

# Tabela de Frequência
dataset['ba'].value_counts()

# Tabela de Frequência
dataset['htn'].value_counts()

# Tabela de Frequência
dataset['dm'].value_counts()
dataset['dm'] = dataset['dm'].replace(to_replace = {'\tyes':'yes', ' yes':'yes', '\tno':'no'})
dataset['dm'] = dataset['dm'].replace(to_replace = {'yes' : 1, 'no' : 0})

# Tabela de Frequência
dataset['dm'].value_counts()

# Tabela de Frequência
dataset['cad'].value_counts()
dataset['cad'] = dataset['cad'].replace(to_replace = {'\tno':'no'})
dataset['cad'] = dataset['cad'].replace(to_replace = {'yes' : 1, 'no' : 0})

# Tabela de Frequência
dataset['cad'].value_counts()

# Tabela de Frequência
dataset['appet'].unique()
dataset['appet'] = dataset['appet'].replace(to_replace={'good':1,'poor':0,'no':np.nan})

# Tabela de Frequência
dataset['appet'].unique()

# Tabela de Frequência
dataset['pe'].value_counts()
dataset['pe'] = dataset['pe'].replace(to_replace = {'yes' : 1, 'no' : 0})

# Tabela de Frequência
dataset['pe'].value_counts()

# Tabela de Frequência
dataset['ane'].value_counts()
dataset['ane'] = dataset['ane'].replace(to_replace = {'yes' : 1, 'no' : 0})

# Tabela de Frequência
dataset['classification'].value_counts()
dataset['classification'] = dataset['classification'].replace(to_replace={'ckd\t':'ckd'})
dataset["classification"] = [1 if i == "ckd" else 0 for i in dataset["classification"]]

# Tabela de Frequência
dataset['classification'].value_counts()

# Verificando os primeiros registros(pacientes)
dataset.head()

# Verificando o tipo das variáveis:
dataset.dtypes

Convertento variáveis "Object" em numéricas:

dataset['rc'].value_counts()

dataset['pcv'] = pd.to_numeric(dataset['pcv'], errors='coerce')
dataset['wc'] = pd.to_numeric(dataset['wc'], errors='coerce')
dataset['rc'] = pd.to_numeric(dataset['rc'], errors='coerce')

# Tipos das Variáveis (núméricas ou object/string):
dataset.dtypes

# Estatística Descritiva das variáveis:
dataset.describe()

# Cheaking Missing (NaN) Values:
dataset.isnull().sum().sort_values(ascending=False)

Substituição/eliminação (ou também conhecido como imputação) de valores ausentes(missings):

# Verificando a lista de Colunas
dataset.columns

#Criando uma lista com o nome das Colunas para usar na substituição de missings)
features = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu',
           'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad',
           'appet', 'pe', 'ane']
           
# Função utilizada para varrer (loop) as colunas e a cada valor missing encontrado,
# ele será substituído pela median 
for feature in features:
    dataset[feature] = dataset[feature].fillna(dataset[feature].median())
    
# Confere se ainda persiste valor missing
dataset.isnull().any().sum()

# Criação da Figura Gráfica
plt.figure(figsize=(24,14))
# Criação do Gráfico heatmap
sns.heatmap(dataset.corr(), annot=True, cmap='YlGnBu')
plt.show()

# Elimionando a 'pcv'
dataset.drop('pcv', axis=1, inplace=True)

# Visualizando os Primeiros Registros
dataset.head()

# Visualizando o Target:
sns.countplot(dataset['classification'],saturation=0.95 )

# Independent and Dependent Feature:
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
y.head()

# Verificando quais Features são as mais Importantes:
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

# Usando o ExtraTree  para nos mostrar as Variáveis mais Importantes
model=ExtraTreesClassifier()
model.fit(X,y)
plt.figure(figsize=(8,6))
ranked_features=pd.Series(model.feature_importances_,index=X.columns)
ranked_features.nlargest(24).plot(kind='barh')
plt.show()

#Função para ranquear as 8 mais importantes
ranked_features.nlargest(8).index

# Separando as 8 variáveis mais importantes em "X" para que o algoritmo treine com esses dados
X = dataset[['hemo', 'htn', 'sg', 'dm', 'al', 'rc', 'appet', 'pc']]
X.head()
#Função para ver os últimos registros
X.tail()
#Verificando o target
y.head()

# Train Test Split:
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=33)
print(X_train.shape)
print(X_test.shape)

Criando o Baseline com o Algoritmo RandomForest

# RandomForestClassifier:
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#Realizando o treinamento (fit) com os dados de treino
RandomForest = RandomForestClassifier()
RandomForest = RandomForest.fit(X_train,y_train)

# Fazendo Previsões com dados de teste:
y_pred = RandomForest.predict(X_test)

# Avaliando a Performance comparando com o gabarito (y) de teste:
print('Accuracy:', accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

Criando a Máquina Preditiva com o Algoritmo GradientBoosting

# GradientBoostingClassifier:
from sklearn.ensemble import GradientBoostingClassifier
GradientBoost = GradientBoostingClassifier(n_estimators=2000)
GradientBoost = GradientBoost.fit(X_train,y_train)

# Predictions:
y_pred = GradientBoost.predict(X_test)

# Performance:
print('Accuracy:', accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))

Salvamento da Máquina Preditiva

# Creating a pickle file for the classifier
import pickle
filename = 'Maquina_Preditiva.pkl'
pickle.dump(GradientBoost, open(filename, 'wb'))
