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
X = dataset.iloc[:, :-1].values # récupération des indices du dataset