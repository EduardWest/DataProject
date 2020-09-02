#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Импортируем все необходимые библиотеки
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


# In[4]:


# Смотрим на первоначальные данные
df = pd.read_csv('house_sales.csv')
df.head()


# In[3]:


df.info() #Проверяем тип данных


# In[9]:


frame = df.iloc[:,2:6] #строим диаграмму рассеяния для некоторых переменных
scatter_matrix(frame, figsize=(16, 9), diagonal="kde") 
#scatter_matrix(frame, figsize=(16, 9), diagonal="hist") 
plt.show()


# In[11]:


# Как можно увидеть - целевая переменная имеет лог-нормальное распределение
Y = df['price']
#график рассеяния
for i in range(len(df.columns)):
    print(df.columns[i])
    plt.scatter(df[df.columns[i]], Y, c='red', marker = '.')
    plt.show()


# 

# In[6]:


f, axes = plt.subplots(1, 2,figsize=(16,9)) #Строим ящик с усами
sns.boxplot(x=df['bedrooms'],y=df['price'], ax=axes[0])
sns.boxplot(x=df['floors'],y=df['price'], ax=axes[1])


# In[12]:


df.isnull().any() #Проверка на нулевые значения


# In[13]:


# Удаляем поле 'id'
df.drop("id", axis = 1, inplace = True)
df.describe()


# In[14]:


df['floors'].value_counts()


# In[37]:


correlation = df.corr()
fig=plt.figure(figsize=(16, 16))
heatmap = sns.heatmap(correlation, annot=True)
fig.savefig('correlation.png')


# In[11]:


df.corr()['price'].sort_values()


# In[15]:


#Гистограмма распределения
df1=df[[
        'price', 'bedrooms', 'bathrooms', 'sqft_living','sqft_lot',
      'floors', 'waterfront', 'view', 'condition', 'grade','sqft_above', 
      'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15'
       ]]
h = df1.hist(figsize=(15,15),bins=30)
sns.despine(left=True, bottom=True)
[x.title for x in h.ravel()];
[x.yaxis.tick_left() for x in h.ravel()];


# In[16]:


f = plt.subplots(figsize=(15,9))
sns.boxplot(x=df['floors'],y=df['price'])


# In[14]:


f = plt.subplots(figsize=(15,9))
sns.boxplot(x=df['bedrooms'],y=df['price'])
# Для 8 комнатных домов самый большой разброс в цене


# In[17]:


f = plt.subplots()
sns.boxplot(x=df['waterfront'], y=df['price'])


# In[18]:


f = plt.subplots(figsize=(15,9))
sns.boxplot(x=df['grade'],y=df['price'])


# model
# buldings
# 

# 

# In[19]:


features =["long","sqft_lot15","yr_renovated" ,"yr_built",
           "condition","sqft_lot","floors", "waterfront",
           "lat" ,"bedrooms" ,"sqft_basement" ,"view" ,
           "bathrooms","sqft_living15","sqft_above",
           "grade","sqft_living"]    
price = df['price']
price_log = np.log(price)

X = df[features ]
Y = price_log
# Делим выборку на обучающую и тестовую
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1)

print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# In[24]:


# L1 регрессия
for i in np.arange(0, 1, 0.1):
    RidgeModel = Ridge(alpha = i, fit_intercept = True,
                       normalize = True, tol = 0.001, solver='auto') 
    RidgeModel.fit(x_train, y_train)
    print(cross_val_score(RidgeModel, x_train, y_train, cv = 3))

    pred = RidgeModel.predict(x_test)
    accuracy = RidgeModel.score(x_test, y_test)
    print('Accuracy:{}, alpha:{}'.format(accuracy, round(i, 1)))


# In[25]:


for i in np.arange(0,1.1,0.1):
    lasso = Lasso(alpha = i, fit_intercept = True, normalize = False,
                  max_iter = 1000, tol = 0.0001, selection = 'cyclic')
    lasso.fit(x_train, y_train)
    print(cross_val_score(lasso, x_train, y_train, cv = 3))

    pred = lasso.predict(x_test)
    accuracy = lasso.score(x_test, y_test)
    print('Accuracy:{}, alpha:{}'.format(accuracy, round(i, 1)))


# In[33]:


# Линейная регрессия
lr = LinearRegression(fit_intercept = True, normalize=True,
                      copy_X = True, n_jobs = 8)
lr.fit(x_train, y_train)
print(cross_val_score(lr, x_train, y_train, cv = 3))

pred = lr.predict(x_test)
accuracy = lr.score(x_test, y_test)
print('predict data ', accuracy) 


# In[34]:


# L1 + L2 Регрессия
for i in np.arange(0, 1.1, 0.1):
    en = ElasticNet(alpha = i, l1_ratio = 0.5, fit_intercept = True,
                    normalize = False, max_iter = 1000,
                    copy_X = True, tol = 0.0001, selection = 'cyclic')
    en.fit(x_train, y_train)
    print(cross_val_score(en, x_train, y_train, cv = 3))

    pred = en.predict(x_test)
    accuracy = en.score(x_test, y_test)
    print('Accuracy:{}, alpha:{}'.format(accuracy, round(i, 1)) 


# In[38]:


# Метод ближайших соседей
for i in np.arange(1, 21, 1):
    knn = KNeighborsRegressor(n_neighbors = i, weights = 'uniform',
                              leaf_size = 30, p = 2, metric='minkowski', n_jobs = 8)
    knn.fit(x_train, y_train)
    print(cross_val_score(knn, x_train, y_train, cv = 3))

    pred = knn.predict(x_test)
    accuracy = knn.score(x_test, y_test)
    print('Accuracy:{},k = {}'.format(accuracy, i)) 


# In[37]:


# Дерево решений
for i in range(1, 15):
    dtr = DecisionTreeRegressor(criterion = 'mse', splitter = 'best',
                                max_depth = i, max_features = 8)
    dtr.fit(x_train, y_train)
    print(cross_val_score(dtr, x_train, y_train, cv = 3))

    pred = dtr.predict(x_test)
    accuracy = dtr.score(x_test, y_test)
    print('Accuracy: {}, depth:{}'.format(accuracy, i))


# In[39]:


# Градиентный бустинг
for i in np.arange(0.05, 0.4, 0.05):
    gbr = GradientBoostingRegressor(loss = 'ls', learning_rate = i, n_estimators = 100,
                                    subsample = 1.0, criterion = 'friedman_mse',
                                    max_depth = 7)
    gbr.fit(x_train, y_train)
    print(cross_val_score(gbr, x_train, y_train, cv = 3))

    pred = gbr.predict(x_test)
    accuracy = gbr.score(x_test, y_test)
    print('Accuracy:{}, learning rate:{}'.format(accuracy, i))


# best

# In[40]:


# Обработка данных
df1 = df
df1['sqft_living']/df1['floors']


# In[41]:


df1['sqft_living/floors'] = df1['sqft_living']/df1['floors']
df1.head()


# In[42]:


df1.drop("sqft_living", axis = 1, inplace = True)


# In[43]:


df1.drop("floors", axis = 1, inplace = True)


# In[44]:


df1.head()


# In[45]:


df1["coordinate"] = df1["long"] + df1["lat"]
df1.head()


# In[46]:


df1.drop("lat", axis = 1, inplace = True)
df1.drop("long", axis = 1, inplace = True)
df1.head()


# In[47]:


df1["yr_renovated"].value_counts()


# In[48]:


len(df1["yr_renovated"])


# In[ ]:


df1["yr_renovated"]


# In[ ]:


df1["yr_renovated"][1]='0'


# In[ ]:


df1.drop("yr_renovated", axis = 1, inplace = True)
df1


# In[49]:


df1.columns


# In[51]:


features =['bedrooms', 'bathrooms', 'sqft_lot', 'waterfront','view', 'condition', 
           'grade', 'sqft_above', 'sqft_basement',
           'yr_built', 'sqft_living15', 'sqft_lot15', 'sqft_living/floors','coordinate']    
price = df1['price']
price_log= np.log(price)

X = df[features ]
Y = price_log
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state=1)

print("number of test samples :", x_test.shape[0])
print("number of training samples:", x_train.shape[0])


# In[52]:


# L1 регрессия
for i in np.arange(0, 1, 0.1):
    RidgeModel = Ridge(alpha = i, fit_intercept = True,
                       normalize = True, tol = 0.001, solver='auto') 
    RidgeModel.fit(x_train, y_train)
    print(cross_val_score(RidgeModel, x_train, y_train, cv = 3))

    pred = RidgeModel.predict(x_test)
    accuracy = RidgeModel.score(x_test, y_test)
    print('Accuracy:{}, alpha:{}'.format(accuracy, round(i, 1)))


# In[53]:


from math import exp
for i in np.arange(0,1.1,0.1):
    lasso = Lasso(alpha = i, fit_intercept = True, normalize = False,
                  max_iter = 1000, tol = 0.0001, selection = 'cyclic')
    lasso.fit(x_train, y_train)
    print(cross_val_score(lasso, x_train, y_train, cv = 3))

    pred = lasso.predict(x_test)
    accuracy = lasso.score(x_test, y_test)
    print('Accuracy:{}, alpha:{}'.format(accuracy, round(i, 1)))


# In[54]:


lr = LinearRegression(fit_intercept = True, normalize = True, copy_X = True, n_jobs=8)
lr.fit(x_train, y_train)
print(cross_val_score(lr, x_train, y_train, cv = 3))

pred=lr.predict(x_test)
accuracy=lr.score(x_test, y_test)
print('predict data ',accuracy) 


# In[56]:


# L1 + L2 Регрессия
for i in np.arange(0, 1.1, 0.1):
    en = ElasticNet(alpha = i, l1_ratio = 0.5, fit_intercept = True,
                    normalize = False, max_iter = 1000,
                    copy_X = True, tol = 0.0001, selection = 'cyclic')
    en.fit(x_train, y_train)
    print(cross_val_score(en, x_train, y_train, cv = 3))

    pred = en.predict(x_test)
    accuracy = en.score(x_test, y_test)
    print('Accuracy:{}, alpha:{}'.format(accuracy, round(i, 1)))


# In[64]:


# Метод ближайших соседей
for i in np.arange(1, 21, 1):
    knn = KNeighborsRegressor(n_neighbors = i, weights = 'uniform',
                              leaf_size = 30, p = 2, metric='minkowski', n_jobs = 8)
    knn.fit(x_train, y_train)
    print(cross_val_score(knn, x_train, y_train, cv = 3))

    pred = knn.predict(x_test)
    accuracy = knn.score(x_test, y_test)
    print('Accuracy:{}, k = {}'.format(accuracy, i))
    print()


# In[74]:


plt.plot(pred ** 2.81812)


# In[75]:


plt.plot(y_test ** 2.81812)


# In[80]:


# Дерево решений
for i in range(1, 15):
    dtr = DecisionTreeRegressor(criterion = 'mse', splitter = 'best',
                                max_depth = i, max_features = 8)
    dtr.fit(x_train, y_train)
    print(cross_val_score(dtr, x_train, y_train, cv = 3))

    pred = dtr.predict(x_test)
    accuracy = dtr.score(x_test, y_test)
    print('Accuracy: {}, depth:{}'.format(accuracy, i))
    print(sep = ' ')


# In[81]:


# Градиентный бустинг
for i in np.arange(0.05, 0.4, 0.05):
    gbr = GradientBoostingRegressor(loss = 'ls', learning_rate = i, n_estimators = 100,
                                    subsample = 1.0, criterion = 'friedman_mse',
                                    max_depth = 7)
    gbr.fit(x_train, y_train)
    print(cross_val_score(gbr, x_train, y_train, cv = 3))

    pred = gbr.predict(x_test)
    accuracy = gbr.score(x_test, y_test)
    print('Accuracy:{}, learning rate:{}'.format(accuracy, i))


# In[ ]:




