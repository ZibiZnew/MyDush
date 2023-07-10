# -*- coding: UTF-8-*-
""" Zadanie 1
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


# 1 Pobrano dane bezpośrednio z github
df = pd.read_csv(r"https://raw.githubusercontent.com/ZibiZnew/MyDush/main/heart.csv", low_memory=True,)

print(df.columns)
print(df.head())
# print(os.listdir())

# 2 Przegląd danych i wstępne zrozumienie co zawierają
print(df.sample(5))
print(df.info())

print(df.describe())

info = ["age","1: mezczyzna, 0: kobieta",
            "typ bolu w klatce piersiowej, 1: typowa angina, 2: NIEtypowa  angina,  3: bol nieprawidłowy, 4: bezobjawowy  ",
            "ciśnienie krwi SPOCZYNKOWE",
            " cholestorol  [mg/dl]",
            "cukier na czczo > 120 mg/dl",
            "EKG spoczynkowe  (values 0,1,2)",
            " maxymalne tetno",
            "angina indukowana wysiłkiem fizycznym",
            "oldpeak = ST depression induced by exercise relative to rest",
            "nachylenie szczytowego odcinka ST podczas ćwiczenia",
            "liczba głównych naczyń (0-3) zabarwionych metodą fluorosopii",
            "thal: 3 = normalne; 6 = defect usunięty; 7 = defect odwracalny"]

for i in range(len(info)):
    print(df.columns[i]+":\t\t\t"+info[i])

print(df["num"].describe())

# Usunięcie kolumn 'ca' i 'thal', które mają zbyt wiele brakujących wartości
# df = df.drop(['ca', 'thal'], axis=1)

# Zamiana znaków zapytania na NaN
df = df.replace('?', np.NaN)

#  Zamiana typów kolumn na liczbowe

df['trestbps'] = pd.to_numeric(df['trestbps'], errors='coerce').astype('Int64')
# df['chol'] = pd.to_numeric(df['chol'], errors='coerce').astype('Int64')
df['fbs'] = pd.to_numeric(df['fbs'], errors='coerce').astype('Int64')
df['restecg'] = pd.to_numeric(df['restecg'], errors='coerce').astype('Int64')
df['thalach'] = pd.to_numeric(df['thalach'], errors='coerce').astype('Int64')
df['exang'] = pd.to_numeric(df['exang'], errors='coerce').astype('Int64')
# df['slope'] = pd.to_numeric(df['slope'], errors='coerce').astype('Int64')
# df['ca'] = pd.to_numeric(df['ca'], errors='coerce').astype('Int64')
# df['thal'] = pd.to_numeric(df['thal'], errors='coerce').astype('Int64')

# ponowne sprawdzenie poprawności
print(df.head())
print(df.info())

# Podział na cechy i etykiety
X = df.drop('num', axis=1)
y = df['num']

# print(X)
# print(y)

# Uzupełnienie brakujących danych
from sklearn.impute import SimpleImputer

# Uzupełnienie brakujących wartości najczęściej występującą wartością
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Podział na zbiór treningowy i testowy
from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.32, random_state=42)

#  CHYBA NAJLEPSZY OSIĄNIĘTY WYNI
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.11,
                                                     random_state=41)

# Uczenie modelu xgboost
import xgboost as xgb

model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Ocena modelu na zbiorze testowym
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność modelu:\t {accuracy * 100:.3f} %")

# Optymalizacja liczby kroków (num_boost_round) i learning rate korzystając z  cross-validation
params = model.get_xgb_params()
cv_results = xgb.cv(params, dtrain=xgb.DMatrix(X_train, label=y_train), num_boost_round=1000, nfold=5,
                    metrics='auc', early_stopping_rounds=10)
best_num_boost_rounds = cv_results.shape[0]
best_learning_rate = cv_results['test-auc-mean'].idxmax()

# Uczenie modelu z optymalnymi parametrami
model = xgb.XGBClassifier(n_estimators=best_num_boost_rounds, learning_rate=best_learning_rate)
model.fit(X_train, y_train)

# Ocena ostatecznego modelu na zbiorze testowym
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(' * ' *33)
print(f"Dokładność  modelu: końcowego \t {accuracy * 100:.3f} %")
print(' * ' *33)
print(df.head())