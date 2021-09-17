#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries
import numpy as np
import pandas as pd
import matplotlib as plt


# In[2]:


#uploading csv file
data = pd.read_csv(r"C:\Users\SAGNIK DAS\OneDrive\Desktop\New folder (3)\train 1.csv")
orgdata = data


# In[3]:


# understanding the data
data.head()


# In[4]:


data.shape


# In[5]:


data.tail()


# In[6]:


data.columns


# In[7]:


data.info()


# In[8]:


# summing up the nissing values (column wise)
data.isnull()


# In[9]:


data.isnull().sum(axis=0).sort_values(ascending=False)


# In[10]:


# summing up the nissing values (row wise)
data.isnull().sum(axis=1).sort_values(ascending=False)


# In[11]:


# columns having atleast one missing value
d = data.isnull().sum() > 0
d


# In[12]:


d.index


# In[13]:


index = d.index[d.values]
index


# In[14]:


data.isnull().all(axis=1).sum()


# In[15]:


# summing up missing values (column wise): cal in %
data.isnull().sum(axis=0).sort_values(ascending=False)/len(data) * 100


# In[16]:


# removing the four columns where the null value % is max
col = data.isnull().sum(axis=0).sort_values(ascending=False).head(4).index.values
col


# In[17]:


data = data.drop(col,axis='columns')
data


# In[18]:


# check the rows with more than five missing values
len(data[data.isnull().sum(axis=1) > 5])/len(data)*100


# In[19]:


# retaining the rows with five or less missing values
data = data[data.isnull().sum(axis=1) <=5]
data


# In[20]:


data


# In[21]:


round(data.isnull().sum(axis=0).sort_values(ascending=False)/len(data) * 100,2)


# In[22]:


data.describe()


# In[23]:


data = data[data['FireplaceQu'].notnull()]
data


# In[24]:


data = data[data['LotFrontage'].notnull()]
data


# In[25]:


data = data[data['BsmtFinType2'].notnull()]
data


# In[26]:


data = data[data['BsmtExposure'].notnull()]
data


# In[27]:


data = data[data['GarageType'].notnull()]
data


# In[28]:


data = data[data['GarageYrBlt'].notnull()]
data


# In[29]:


data = data[data['GarageFinish'].notnull()]
data


# In[30]:


data = data[data['GarageQual'].notnull()]
data


# In[31]:


data = data[data['MasVnrType'].notnull()]
data


# In[32]:


data = data[data['MasVnrArea'].notnull()]
data


# In[33]:


round(data.isnull().sum(axis=0).sort_values(ascending=False)/len(data) * 100,2)


# In[34]:


len(data)/len(orgdata) * 100


# In[35]:


data.to_csv(r"C:\Users\SAGNIK DAS\OneDrive\Desktop\New folder (3)\train 12.csv", index=False)


# In[36]:


#uploading new csv file
data = pd.read_csv(r"C:\Users\SAGNIK DAS\OneDrive\Desktop\New folder (3)\train 12.csv")


# In[37]:


# understanding the data
data.head()


# In[38]:


data.shape


# In[39]:


data.tail()


# In[40]:


data.columns


# In[41]:


data.info()


# In[42]:


# find the null values
data.isnull().any()


# In[43]:


data.isnull().sum()


# In[44]:


data_num = data.select_dtypes(include = ['float64', 'int64', 'object'])
data_num.head()


# In[45]:


data_num.hist(figsize=(18, 22), bins=55, xlabelsize=10, ylabelsize=10); 


# In[46]:


data.drop(['MSZoning','Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType','HouseStyle', 'OverallQual', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',  'ExterQual' ,'ExterCond','Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '2ndFlrSF', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'SaleType', 'SaleCondition'], axis = 1)


# In[47]:


# Created new csv file and importing the new csv file with new data.
data = pd.read_csv(r"C:\Users\SAGNIK DAS\OneDrive\Desktop\New folder (3)\train 121.csv")


# In[48]:


# understanding the data
data.head()


# In[49]:


data.shape


# In[50]:


data.describe()


# In[51]:


data.tail()


# In[52]:


data.columns


# In[53]:


data.info()


# In[54]:


# importing required library
import seaborn as sns
import os
import csv
import sklearn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
plt.style.use('bmh')


# In[55]:


SalePrice = pd.pivot_table(data,index = 'Id', values='SalePrice')


# In[56]:


SalePrice


# In[57]:


SalePrice.plot(kind='bar')


# In[58]:


SalePrice = pd.pivot_table(data,index = 'MSSubClass', values='SalePrice')


# In[59]:


SalePrice


# In[60]:


SalePrice.plot(kind='bar')


# In[61]:


SalePrice = pd.pivot_table(data,index = 'LotFrontage', values='SalePrice')


# In[62]:


SalePrice


# In[63]:


SalePrice.plot(kind='bar')


# In[64]:


SalePrice = pd.pivot_table(data,index = 'LotArea', values='SalePrice')


# In[65]:


SalePrice


# In[66]:


SalePrice.plot(kind='bar')


# In[67]:


SalePrice = pd.pivot_table(data,index = 'OverallQual', values='SalePrice')


# In[68]:


SalePrice


# In[69]:


SalePrice.plot(kind='bar')


# In[70]:


SalePrice = pd.pivot_table(data,index = 'YearBuilt', values='SalePrice')


# In[71]:


SalePrice


# In[72]:


SalePrice.plot(kind='bar')


# In[73]:


SalePrice = pd.pivot_table(data,index = 'YearRemodAdd', values='SalePrice')


# In[74]:


SalePrice


# In[75]:


SalePrice.plot(kind='bar')


# In[76]:


SalePrice = pd.pivot_table(data,index = 'BsmtUnfSF', values='SalePrice')


# In[77]:


SalePrice


# In[78]:


SalePrice.plot(kind='bar')


# In[79]:


SalePrice = pd.pivot_table(data,index = 'TotalBsmtSF', values='SalePrice')


# In[80]:


SalePrice


# In[81]:


SalePrice.plot(kind='bar')


# In[82]:


SalePrice = pd.pivot_table(data,index = '1stFlrSF', values='SalePrice')


# In[83]:


SalePrice


# In[84]:


SalePrice.plot(kind='bar')


# In[85]:


SalePrice = pd.pivot_table(data,index = 'G1ivArea', values='SalePrice')


# In[86]:


SalePrice


# In[87]:


SalePrice.plot(kind='bar')


# In[88]:


SalePrice = pd.pivot_table(data,index = 'Tot2sAbvGrd', values='SalePrice')


# In[89]:


SalePrice


# In[90]:


SalePrice.plot(kind='bar')


# In[91]:


SalePrice = pd.pivot_table(data,index = 'GarageYrBlt', values='SalePrice')


# In[92]:


SalePrice


# In[93]:


SalePrice.plot(kind='bar')


# In[111]:


SalePrice = pd.pivot_table(data,index = 'GarageArea', values='SalePrice')


# In[112]:


SalePrice


# In[113]:


SalePrice.plot(kind='bar')


# In[107]:


SalePrice = pd.pivot_table(data,index = 'YrSold', values='SalePrice')


# In[108]:


SalePrice


# In[109]:


SalePrice.plot(kind='bar')


# In[114]:


corelation = data.corr() 


# In[115]:


sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns
            ,annot=True)


# In[116]:


sns.boxplot


# In[117]:


sns.pairplot


# In[118]:


y = np.array(data['SalePrice'])
y.shape


# In[119]:


x = np.array(data.loc[:, 'Id' : 'YrSold'])
x.shape


# In[120]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.9,)


# In[121]:


x_train.shape


# In[122]:


x_test.shape


# In[123]:


y_train.shape


# In[124]:


y_test.shape


# In[125]:


from sklearn.model_selection import KFold
folds = (KFold(n_splits = 10, shuffle = True, random_state = 100))


# In[142]:


hyper_params = [{'n_features_to_select':list(range(1,17))}]


# In[143]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)


# In[144]:


from sklearn.feature_selection import RFE
rfe = RFE(lm)
from sklearn.model_selection import GridSearchCV
modelcv = GridSearchCV(estimator = rfe,
                      param_grid = hyper_params,
                      scoring = 'r2',
                      cv = folds,
                      verbose = 1,
                      return_train_score = True)
modelcv.fit(x_train, y_train)


# In[145]:


cvresults = pd.DataFrame(modelcv.cv_results_)
cvresults


# In[146]:


data.shape


# In[147]:


print(np.mean(cvresults))


# In[148]:


plt.figure(figsize = (20,17))


# In[134]:


plt.plot(cvresults['param_n_features_to_select'], cvresults['mean_test_score'])
plt.plot(cvresults['param_n_features_to_select'], cvresults['mean_train_score'])
plt.xlabel('Number of features')
plt.ylabel('Optimal number of features')


# In[149]:


n_features_optimal = 6


# In[150]:


lm = LinearRegression()
lm.fit(x_train, y_train)


# In[151]:


rfe = RFE(lm, n_features_to_select = n_features_optimal)


# In[152]:


rfe.fit(x_train, y_train)


# In[153]:


y_pred = lm.predict(x_test)
y_pred


# In[154]:


r2 = sklearn.metrics.r2_score(y_test, y_pred)
print(r2)


# In[156]:


#uploading test csv file for data cleaning, eda analysis and hyperparameter optimization 
data = pd.read_csv(r"C:\Users\SAGNIK DAS\OneDrive\Desktop\New folder (3)\test.csv")
orgdata = data


# In[157]:


# understanding the data
data.head()


# In[158]:


data.shape


# In[159]:


data.tail()


# In[160]:


data.columns


# In[161]:


data.info()


# In[162]:


# summing up the nissing values (column wise)
data.isnull()


# In[163]:


data.isnull().sum(axis=0).sort_values(ascending=False)


# In[164]:


# summing up the nissing values (row wise)
data.isnull().sum(axis=1).sort_values(ascending=False)


# In[165]:


# columns having atleast one missing value
d = data.isnull().sum() > 0
d


# In[166]:


d.index


# In[167]:


index = d.index[d.values]
index


# In[168]:


data.isnull().all(axis=1).sum()


# In[169]:


# summing up missing values (column wise): cal in %
data.isnull().sum(axis=0).sort_values(ascending=False)/len(data) * 100


# In[170]:


# removing the four columns where the null value % is max
col = data.isnull().sum(axis=0).sort_values(ascending=False).head(4).index.values
col


# In[171]:


data = data.drop(col,axis='columns')
data


# In[172]:


# check the rows with more than five missing values
len(data[data.isnull().sum(axis=1) > 5])/len(data)*100


# In[173]:


# retaining the rows with five or less missing values
data = data[data.isnull().sum(axis=1) <=5]
data


# In[174]:


data


# In[175]:


round(data.isnull().sum(axis=0).sort_values(ascending=False)/len(data) * 100,2)


# In[176]:


data.describe()


# In[177]:


data = data[data['FireplaceQu'].notnull()]
data


# In[178]:


data = data[data['LotFrontage'].notnull()]
data


# In[179]:


data = data[data['GarageCond'].notnull()]
data


# In[180]:


data = data[data['GarageType'].notnull()]
data


# In[181]:


data = data[data['GarageYrBlt'].notnull()]
data


# In[182]:


round(data.isnull().sum(axis=0).sort_values(ascending=False)/len(data) * 100,2)


# In[183]:


data = data[data['BsmtFinType2'].notnull()]
data


# In[184]:


data = data[data['BsmtExposure'].notnull()]
data


# In[185]:


data = data[data['BsmtCond'].notnull()]
data


# In[186]:


data = data[data['BsmtQual'].notnull()]
data


# In[187]:


round(data.isnull().sum(axis=0).sort_values(ascending=False)/len(data) * 100,2)


# In[188]:


len(data)


# In[189]:


len(data)/len(orgdata) * 100


# In[190]:


data.to_csv(r"C:\Users\SAGNIK DAS\OneDrive\Desktop\New folder (3)\test12.csv", index=False)


# In[191]:


#uploading new csv file
data = pd.read_csv(r"C:\Users\SAGNIK DAS\OneDrive\Desktop\New folder (3)\test12.csv")


# In[192]:


# understanding the data
data.head()


# In[193]:


data.shape


# In[194]:


data.tail()


# In[195]:


data.columns


# In[196]:


data.info()


# In[197]:


data.isnull().sum()


# In[198]:


data_num = data.select_dtypes(include = ['float64', 'int64', 'object'])
data_num.head()


# In[199]:


data_num.hist(figsize=(18, 22), bins=55, xlabelsize=10, ylabelsize=10); 


# In[200]:


data.drop(['MSZoning','Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType','HouseStyle', 'OverallQual', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',  'ExterQual' ,'ExterCond','Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '2ndFlrSF', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'SaleType', 'SaleCondition'], axis = 1)


# In[201]:


# Created new csv file and importing the new csv file.
data = pd.read_csv(r"C:\Users\SAGNIK DAS\OneDrive\Desktop\New folder (3)\test121.csv")


# In[202]:


# understanding the data
data.head()


# In[203]:


data.shape


# In[204]:


data.describe()


# In[205]:


data.tail()


# In[206]:


data.columns


# In[207]:


data.info()


# In[208]:


YearSold = pd.pivot_table(data,index = 'Id', values='YrSold')


# In[209]:


YearSold


# In[210]:


YearSold.plot(kind='bar')


# In[211]:


YearSold = pd.pivot_table(data,index = 'MSSubClass', values='YrSold')


# In[212]:


YearSold


# In[213]:


YearSold.plot(kind='bar')


# In[214]:


YearSold = pd.pivot_table(data,index = 'LotFrontage', values='YrSold')


# In[215]:


YearSold


# In[216]:


YearSold.plot(kind='bar')


# In[217]:


YearSold = pd.pivot_table(data,index = 'LotArea', values='YrSold')


# In[218]:


YearSold


# In[219]:


YearSold.plot(kind='bar')


# In[220]:


YearSold = pd.pivot_table(data,index = 'OverallCond', values='YrSold')


# In[221]:


YearSold


# In[222]:


YearSold.plot(kind='bar')


# In[223]:


YearSold = pd.pivot_table(data,index = 'YearBuilt', values='YrSold')


# In[224]:


YearSold


# In[225]:


YearSold.plot(kind='bar')


# In[226]:


YearSold = pd.pivot_table(data,index = 'YearRemodAdd', values='YrSold')


# In[227]:


YearSold


# In[228]:


YearSold.plot(kind='bar')


# In[229]:


YearSold = pd.pivot_table(data,index = 'BsmtUnfSF', values='YrSold')


# In[230]:


YearSold


# In[231]:


YearSold.plot(kind='bar')


# In[232]:


YearSold = pd.pivot_table(data,index = 'TotalBsmtSF', values='YrSold')


# In[233]:


YearSold


# In[234]:


YearSold.plot(kind='bar')


# In[235]:


YearSold = pd.pivot_table(data,index = '1stFlrSF', values='YrSold')


# In[236]:


YearSold


# In[237]:


YearSold.plot(kind='bar')


# In[238]:


YearSold = pd.pivot_table(data,index = 'GrLivArea', values='YrSold')


# In[239]:


YearSold


# In[240]:


YearSold.plot(kind='bar')


# In[241]:


YearSold = pd.pivot_table(data,index = 'TotRmsAbvGrd', values='YrSold')


# In[242]:


YearSold


# In[243]:


YearSold.plot(kind='bar')


# In[244]:


YearSold = pd.pivot_table(data,index = 'BsmtHalfBath', values='YrSold')


# In[245]:


YearSold


# In[246]:


YearSold.plot(kind='bar')


# In[247]:


YearSold = pd.pivot_table(data,index = 'TotRmsAbvGrd.1', values='YrSold')


# In[248]:


YearSold


# In[249]:


YearSold.plot(kind='bar')


# In[250]:


YearSold = pd.pivot_table(data,index = 'GarageYrBlt', values='YrSold')


# In[251]:


YearSold


# In[252]:


YearSold.plot(kind='bar')


# In[253]:


YearSold = pd.pivot_table(data,index = 'GarageArea', values='YrSold')


# In[254]:


YearSold


# In[255]:


YearSold.plot(kind='bar')


# In[256]:


corelation = data.corr() 


# In[257]:


sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns
            ,annot=True)


# In[258]:


sns.boxplot


# In[259]:


sns.pairplot


# In[260]:


y = np.array(data['YrSold'])
y.shape


# In[263]:


x = np.array(data.loc[:, 'Id' : 'OverallCond'])
x.shape


# In[264]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,)


# In[265]:


x_train.shape


# In[266]:


x_test.shape


# In[267]:


y_train.shape


# In[268]:


y_test.shape


# In[269]:


folds = (KFold(n_splits = 10, shuffle = True, random_state = 100))


# In[270]:


hyper_params = [{'n_features_to_select':list(range(1,5))}]


# In[271]:


lm = LinearRegression()
lm.fit(x_train, y_train)


# In[272]:


rfe = RFE(lm)
modelcv = GridSearchCV(estimator = rfe,
                      param_grid = hyper_params,
                      scoring = 'r2',
                      cv = folds,
                      verbose = 1,
                      return_train_score = True)
modelcv.fit(x_train, y_train)


# In[273]:


cvresults = pd.DataFrame(modelcv.cv_results_)
cvresults


# In[274]:


data.shape


# In[275]:


print(np.mean(cvresults))


# In[276]:


plt.figure(figsize = (20,5))


# In[277]:


plt.plot(cvresults['param_n_features_to_select'], cvresults['mean_test_score'])
plt.plot(cvresults['param_n_features_to_select'], cvresults['mean_train_score'])
plt.xlabel('Number of features')
plt.ylabel('Optimal number of features')


# In[278]:


n_features_optimal = 6


# In[279]:


lm = LinearRegression()
lm.fit(x_train, y_train)


# In[280]:


rfe = RFE(lm, n_features_to_select = n_features_optimal)


# In[281]:


rfe.fit(x_train, y_train)


# In[282]:


y_pred = lm.predict(x_test)
y_pred


# In[283]:


r2 = sklearn.metrics.r2_score(y_test, y_pred)
print(r2)


# In[ ]:




