# -*- coding: UTF-8-*-
""" Zadanie 3 """
import random
import numpy as np
import operator
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
import plotly as plot
import plotly.express as px

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


# 1 Pobrano dane na dysk
df = pd.read_csv(r"https://raw.githubusercontent.com/ZibiZnew/MyDush/main/all_seasons.csv", low_memory=True,
                 parse_dates=['draft_year'])
# df.index = df["player_name"]
columns = df.columns
print(columns)

# Otworzono plik
print(df.head())
print(df.info())

# Przekonwertowano dane do typów najbardziej odpowiadających do dalszej analizy. Rozmiar pliku został zmniejszony do memory usage: 1.8+ MB
df['player_name'] = df['player_name'].astype('category')
df['team_abbreviation'] = df['team_abbreviation'].astype('category')
df['age'] = df['age'].astype(int)
df['draft_year'] = df['draft_year'].astype('category')
df['draft_year'] = df['draft_year'].convert_dtypes()
df['country'] = df['country'].astype('category')

print(df.info())

# 1. Czy jest jakikolwiek zawodnik spoza USA, który przed NBA uczęszczał do college'u Kentucky?

# print(df['country'].nunique())
college_Kentucky = df['college'].str.contains('Kentucky')
print(f' Ilość  Uczelni w Kentucky:\n {len(college_Kentucky)}')

# college_Kentucky = df['college'] == 'Kentucky'
# print(df[college_Kentucky].count())


# gracze NBA spoza USA
no_USA = df['country'] != 'USA'
# print(df[no_USA])

solution_1 = df[college_Kentucky & no_USA]
# print(solution_1)
print(f' Liczba zawodników  NBA spoza USA, który uczęszczał do college w Kentucky = {len(solution_1)}')
'''
Odpowiadając na pytanie 1, nie można jednoznacznie odpowiedzieć czy ci zawodnicy studiowali w Kentucky (na różnych Uczelniach w tym Stanie USA) przed, w trakcie czy już po NBA.
Dane z tego pliku nie podają jednoznacznie info, gdyż jest brak wiedzy o studiach dziennych, wieczorowych czy zaocznych. Sprawdzajac  poszczególnych zawodników (np. Facebook)
można uzyskac całkiem inną wiedzę niż płynącą wprost z posiadanych danych
'''

# 2. Ile wynosi średni wzrost zawodnika w calach? Czy od pierwszego sezonu w tym zbiorze do teraz powiększył się czy zmniejszył?

height_player = df['player_height']
print(height_player.head())
print(f'ŚREDNI WZROST ZAWODNIKA NBA w cm:\t\t\t{height_player.mean():.2f} cm')
na_cale = height_player / 2.54
print(f'ŚREDNI WZROST ZAWODNIKA NBA w calach:\t{na_cale.mean():.2f} cala')


# 3.	Kto w badanym okresie rzucił łącznie najwięcej punktów?
points = df['pts'].max()
print(f' ŚREDNIA LICZBA PUNKTÓW ZDOBYTYCH W MECZU: {points} pkt')

zawodnik = df.groupby('pts')
print(zawodnik.first())
print(f' ZAWODNIKIEM, KTÓY ZDOBYŁ NAJWIĘKSZĄ ILOŚĆ PUNKTÓW JEST:\n{df.loc[10572]}')



#  4.	W której rundzie draftu największy udział procentowy mieli zawodnicy ważący  więcej niż 100 kg?

df['draft_round'] = df['draft_round'].replace('Undrafted', 9)  # 9 tj Undrafted
player_100 = df['player_weight'] > 100
print(df[player_100].head())

runda_draftu = df['draft_round']
print(runda_draftu.unique())
print(runda_draftu.info())

# tworzę plik z zakresem zawodnicy ważący  więcej niż 100 kg oraz w której rundzie draftu największy udział procentowy mieli ci zawodnicy
four = df[player_100 & runda_draftu]
print(four.info())
fat_player = four.groupby('draft_round')

print(f"Liczność zawodników ważcych więcej niż 100 kg w  drafcie numer 1:\t {len(fat_player.get_group('1') )}")
print(f"Liczność zawodników ważcych więcej niż 100 kg w  drafcie numer 2:\t {len(fat_player.get_group('2') )}")
print(f"Liczność zawodników ważcych więcej niż 100 kg w  drafcie numer 3:\t {len(fat_player.get_group('3') )}")
print(f"Liczność zawodników ważcych więcej niż 100 kg w  drafcie numer 4:\t {len(fat_player.get_group('4') )}")
print(f"Liczność zawodników ważcych więcej niż 100 kg w  drafcie numer 5:\t 0")
print(f"Liczność zawodników ważcych więcej niż 100 kg w  drafcie numer 6:\t {len(fat_player.get_group('6') )}")
print(f"Liczność zawodników ważcych więcej niż 100 kg w  drafcie numer 7:\t {len(fat_player.get_group('7') )}")

# Odpowiadając na pytanie, w której rundzie draftu zawodnicy powyżej 100 kg mieli najwiiększy udział, to jest to fraft numer 1


