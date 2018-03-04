# Data Preprocessing

# Importer les librairies
import numpy  as np
import matplotlib.pyplot as plt
import pandas as pd

# Importer le dataset
dataset = pd.read_csv('Data.csv') # permet d'importer les datas du csv

# variables indépendantes: variables prédicitives
# variables dépendantes: celles que l'on doit prédire

# Création de la matrice des variables indépendantes
# ':' toutes les lignes du dataset
#☺ ':-1' toutes les colones sauf la dernière
x = dataset.iloc[:, :-1].values # récupération des indices du dataset
y = dataset.iloc[:, -1].values

# Gérer données manquantes
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#☻ associe l'imputer à x et remplace les dnnées manquantes par la moyenne des colonnes
imputer.fit(x[:, 1:3]) # range de 1 à 3 donc prends les colonnes 1 et 2
x[:, 1:3] = imputer.transform(x[:, 1:3])
