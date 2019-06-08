import numpy as np
import pandas as pd


# **Задание 1**
# Создайте numpy array с элементами от числа N до 0 (например, для N = 10 это будет array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])).

n = 10
print(np.linspace(n - 1, 0, num=n,dtype=int))
print(np.array(range(n - 1, -1, -1)))

# **Задание 2**
# Создайте диагональную матрицу с элементами от N до 0. Посчитайте сумму ее значений на диагонали.

n = 10
arr = np.linspace(n, 0, num=n + 1, dtype=int)
print(np.diag(arr).sum())

# **Задание 3**
# Скачайте с сайта https://grouplens.org/datasets/movielens/ датасет любого размера. Определите какому фильму было выставлено больше всего оценок 5.0.

movies = pd.read_csv('movies.csv', usecols=[0, 1])
ratings = pd.read_csv('ratings.csv', usecols=[1, 2])

mov_rat = pd.merge(ratings[ratings.rating == 5.0], movies, 'left' ,on='movieId')
print(mov_rat['title'].value_counts()[:1].keys().tolist()[0])

# **Задание 4**
# По данным файла power.csv посчитайте суммарное потребление стран Прибалтики (Латвия, Литва и Эстония) категорий 4, 12 и 21 за период с 2005 по 2010 года. Не учитывайте в расчетах отрицательные значения quantity.

power = pd.read_csv('power.csv')

result = power[ ((power['country']=='Lithuania') | (power['country']=='Latvia') | (power['country']=='Estonia')) & ((power['year']>=2005) & (power['year']<=2010)) & ((power['quantity']>0)) & ((power['category']==4) | (power['category']==12) | (power['category']==21)) ]['quantity'].sum()

print(result)

# **Задание 5**
# Решите систему уравнений:
# 4x + 2y + z = 4
# x + 3y = 12
# 5y + 4z = -3

from numpy import linalg

a = np.array([[4, 2, 1], [1, 3, 0], [0, 5, 4]])
b = np.array([4, 12, -3])

print(linalg.solve(a, b))
