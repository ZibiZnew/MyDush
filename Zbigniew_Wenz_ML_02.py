# -*- coding: UTF-8-*-
""" Zadanie 2

"""
import numpy as np
import operator
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
import plotly as plot
import seaborn as sns
import os
import plotly.express as px
from  IPython.display import display
import warnings
warnings.filterwarnings('ignore')
pd.options.plotting.backend = 'plotly'

# WAŻNE ustawia parametry ekrany PyCharm
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.width', 999)
pd.set_option('display.max_columns', None)
pd.options.display.max_rows = None
pd.set_option('display.max_colwidth', 2)
pd.options.display.colheader_justify = 'right'  # moje ustawienia
pd.options.display.max_info_columns = 150
np.set_printoptions(linewidth=300)

mpl.rcParams['figure.dpi'] = 110
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['font.size'] = 16
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'


# 1 Pobranie danych do zadania 2 i wstępna ocena ich zawartości
from sklearn.datasets import load_wine

data = load_wine()
# print(data.DESCR)

# http = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None )
# print(http.head())
# print(http.describe().T)
# print(http.info())

# Sprawdzenie rozmiaru danych
print(f"Rozmiar danych:\t {data.data.shape}")

# Sprawdzenie nazw cech
print(f"Nazwy cech:\t {data.feature_names}")
print(f"Liczność nazw cech:\t {len(data.feature_names)}")

# Sprawdzenie dostępnych klas
print(f"Dostępne klasy: \t{data.target_names}")

# 2 . Opracowanie modelu klasyfikacyjnego w XGBoost

import xgboost as xgb
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Wczytanie danych
data = load_wine()
X = data.data
y = data.target

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tworzenie obiektu modelu XGBoost
model = xgb.XGBClassifier()

# Trenowanie modelu
model.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred = model.predict(X_test)

# Obliczenie dokładności modelu
accuracy = accuracy_score(y_test, y_pred)
print(' - '*33)
print(f"Dokładność modelu XGBoost:\t {accuracy:.3f}")
print(' - '*33)


# 3. Opracowanie modelu klasyfikacyjnego w oparciu o prostą sieć neuronową

import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Wczytanie danych
data = load_wine()
X = data.data
y = data.target

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizacja danych
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Tworzenie modelu sieci neuronowej
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(data.target_names), activation='softmax')
                                                ])

# Kompilacja modelu
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Predykcja na zbiorze testowym
y_pred_prob = model.predict(X_test)
y_pred = tf.argmax(y_pred_prob, axis=1).numpy()

# Obliczenie dokładności modelu
accuracy = accuracy_score(y_test, y_pred)
print(' - '*22)
print(f"Dokładność prostego modelu sieci neuronowej:\t {accuracy:.3f}")



# 4.	Przeprowadzam  trening modeli i walidację wyników

import xgboost as xgb
import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Wczytanie danych
data = load_wine()
X = data.data
y = data.target

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Model klasyfikacyjny oparty na XGBoost
model_xgb = xgb.XGBClassifier()
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(' - '*22)
print(f"Dokładność modelu XGBoost:\t{accuracy_xgb:.3f}")

# 2. Model klasyfikacyjny oparty na prostym modelu sieci neuronowej
model_nn = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(data.target_names), activation='softmax')
                                                    ])
model_nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model_nn.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Predykcja na zbiorze testowym
y_pred_nn_prob = model_nn.predict(X_test)
y_pred_nn = tf.argmax(y_pred_nn_prob, axis=1).numpy()
accuracy_nn = accuracy_score(y_test, y_pred_nn)
print(' - '*22)
print(f"Dokładność prostego modelu sieci neuronowej:\t{accuracy_nn:.3f}")
print(' - '*22)

# Wykres dokładności dla obu modeli
plt.plot(history.history['accuracy'], label='Model sieci neuronowej')
plt.axhline(y=accuracy_xgb, color='r', linestyle='--', label='Model XGBoost')
plt.title('Dokładność modeli')
plt.xlabel('Epoki')
plt.ylabel('Dokładność')
plt.legend()
plt.show()

