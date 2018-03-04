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
# ':-1' toutes les colones sauf la dernière
x = dataset.iloc[:, :-1].values # récupération des indices du dataset
y = dataset.iloc[:, -1].values

# Gérer données manquantes
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# associe l'imputer à x et remplace les dnnées manquantes par la moyenne des colonnes
imputer.fit(x[:, 1:3]) # range de 1 à 3 donc prends les colonnes 1 et 2
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Gérer les variables catégoriques :
# countries et purchased ne sont pas des variables numériques continues
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:, 0] = labelencoder_X.fit_transform(x[:, 0]) #selection de la colonne pays
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)


# Diviser le dataset entre le Training set et le Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 0) # random_state n'est pas obligatoire. Si on met la même valeur, on obtiendra les mêmesrésultats
