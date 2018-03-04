# Data Preprocessing

# Importer les librairies
import numpy  as np
import matplotlib.pyplot as plt
import pandas as pd

# Importer le dataset
dataset = pd.read_csv('Salary_Data.csv') # permet d'importer les datas du csv

    # variables indépendantes: variables prédicitives
    # variables dépendantes: celles que l'on doit prédire

# Création de la matrice des variables indépendantes
    # ':' toutes les lignes du dataset
    # ':-1' toutes les colones sauf la dernière
x = dataset.iloc[:, :-1].values # récupération des indices du dataset
y = dataset.iloc[:, -1].values

# Diviser le dataset entre le Training set et le Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 1.0/3, random_state = 0) # random_state n'est pas obligatoire. Si on met la même valeur, on obtiendra les mêmesrésultats

# Construction du modele
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# on va lier regerssor au training set
regressor.fit(X_train, Y_train)

# faire de nouvelles prédictions
Y_pred = regressor.predict(X_test)
regressor.predict(15) # salaire d'un employé ayant 15 années d'xp prédit par le modèle