
# coding: utf-8

# In[144]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[145]:


train = pd.read_csv('trgdata1.csv')
test = pd.read_csv('challenge_data.csv')

# train.columns
# test = pd.read_csv('test.csv')


# In[146]:


cols = ['T2','P2','P15','Nf','Nc','EPR','NRf','Nrc','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32']


# In[147]:


train = train.drop(cols,axis = 1)
test = test.drop(cols,axis = 1)


# In[148]:


train = train.drop('Ops Mode',axis = 1)
# test = test.drop('Ops Mode',axis = 1)


# In[149]:


TRA = train['TRA']
OpsMode = []
for x in TRA:
    if x == 0:
        OpsMode.append(1)
    if x == 20:
        OpsMode.append(2)
    if x == 40:
        OpsMode.append(3)
    if x == 60:
        OpsMode.append(4)
    if x == 80:
        OpsMode.append(5)
    if x == 100:
        OpsMode.append(6)

        


# In[125]:


TRA1 = test['TRA']
OpsMode1 = []
for x in TRA1:
    if x == 0:
        OpsMode1.append(1)
    if x == 20:
        OpsMode1.append(2)
    if x == 40:
        OpsMode1.append(3)
    if x == 60:
        OpsMode1.append(4)
    if x == 80:
        OpsMode1.append(5)
    if x == 100:
        OpsMode1.append(6)
    


# In[126]:


TRA1


# In[106]:


v = TRA1.unique()
v


# In[150]:


train['OpsMode'] = OpsMode
# test['OpsMode'] = OpsMode1
train = train.drop(['Altitude','Mach No','TRA'],axis=1)
# test = test.drop(['Altitude','Mach No','TRA'],axis=1)


# In[151]:


unitlife = []
n = train['A/C No.'].nunique()


# In[152]:


for i in range(1,n+1):
    k = train[train['A/C No.'] == i].shape[0]
    unitlife += k*[k]


# In[153]:


train['unitlife'] = unitlife
train['Time'] = train['Time']-train['unitlife']
train = train.drop('unitlife',axis = 1)
# test = train


# In[154]:


import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler,StandardScaler
get_ipython().magic('matplotlib inline')


# In[155]:


import seaborn as sns
sns.set()
d = train[train['OpsMode']==1]
y = d['T24']
x = d['Time']
ax = plt.scatter(x ,y)


# In[204]:


X = train[(train['Time']<-300) | (train['Time']>-5)]
X = X.drop('A/C No.',axis = 1)


# In[205]:


X.head()


# In[207]:


X_prescaled_df = pd.DataFrame(X)
X_prescaled_df


# In[208]:


X_prescaled_df.to_excel('prescaled_data.xlsx')


# In[201]:


scaler = MinMaxScaler()
Xscaled = X
print(Xscaled)
scaler.fit(Xscaled)
Xscaled = scaler.transform(Xscaled)


# In[206]:


Xscaled_df = pd.DataFrame(Xscaled)
Xscaled_df


# In[203]:


Xscaled_df.to_excel('scaled_data.xlsx')


# In[45]:


# X1 = X[X['OpsMode'] == 1]
# X1 = X1.drop('OpsMode',axis = 1)
# y1 = np.where(X1['Time']>-5,0,1)


# In[46]:


# (y1 == 1).any()


# In[47]:


# scaler = MinMaxScaler()
# X11 = X1
# scaler.fit(X11)
# X11 = scaler.transform(X11)


# In[48]:


# X1_train, X1_test, y1_train, y1_test = train_test_split(X11, y1, test_size=0.2, random_state=0)


# In[49]:


# regressor = LinearRegression()
# regressor.fit(X1_train,y1_train)


# In[50]:


# coeff_df = pd.DataFrame(regressor.coef_,X1.columns, columns = ['Coefficient'])
# coeff_df


# In[51]:


# y1_pred = regressor.predict(X1_test)
# y1_pred


# In[52]:


# df = pd.DataFrame({'Actual':y1_test, 'Predict':y1_pred})
# df[df['Actual'] == 1]


# In[180]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[181]:


scaler = MinMaxScaler()
M = []
score = []
for i in range (1,7):
    Xi = X[X['OpsMode'] == i]
    Xi = Xi.drop('OpsMode',axis=1)
    yi = np.where(Xi['Time']>-5,0,1)
    Time = Xi['Time']
    Xi = Xi.drop('Time',axis=1)
    scaler.fit(Xi)
    Xi = scaler.transform(Xi)
    Xi_train, Xi_test, yi_train, yi_test = train_test_split(Xi, yi, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    Mi = regressor.fit(Xi_train,yi_train)
    yi_pred = Mi.predict(Xi_test)
    score.append(r2_score(yi_test,yi_pred))
    coeff_df = pd.DataFrame(Mi.coef_, columns = ['Coefficient'])
    print(coeff_df)
    M.append(Mi)


# In[182]:



score


# In[190]:


#with non-linear regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
scaler = MinMaxScaler()
M_ = []
score_ = []
for i in range (1,7):
    Xi = X[X['OpsMode'] == i]
    Xi = Xi.drop('OpsMode',axis=1)
    yi = np.where(Xi['Time']>-5,0,1)
    Time = Xi['Time']
    Xi = Xi.drop('Time',axis=1)
    scaler.fit(Xi)
    Xi = scaler.transform(Xi)
    Xi_train, Xi_test, yi_train, yi_test = train_test_split(Xi, yi, test_size=0.2, random_state=0)
    poly = PolynomialFeatures(degree=2)
    Xi_train = poly.fit_transform(Xi_train)
    Xi_test = poly.fit_transform(Xi_test)
    regressor = LinearRegression()
    Mi_ = regressor.fit(Xi_train,yi_train)
    yi_pred = Mi_.predict(Xi_test)
    score_.append(r2_score(yi_test,yi_pred))
    coeff_df = pd.DataFrame(Mi_.coef_, columns = ['Coefficient'])
#     print(coeff_df)
    M_.append(Mi_)


# In[191]:


print(score)
score_
#output of R2 score for all the 6 models
"""[0.9528724312590026,
 0.9503008982193133,
 0.9324757448014721,
 0.9404550141050213,
 0.9143505859600716,
 0.9434097021360036]

"""
# In[193]:


#with non-linear regression
scaler = MinMaxScaler()
data_ = []
for i in range(1,7):
    Xi = train[train['OpsMode'] == i]
    Xii = Xi.drop(['A/C No.','OpsMode','Time'],axis = 1)
    Xi = Xi.drop(['T24','T30','T50','P30','Ps30','phi','BPR','OpsMode'],axis = 1)
    scaler.fit(Xii)
    Xii = scaler.transform(Xii)
    poly = PolynomialFeatures(degree=2)
    Xii = poly.fit_transform(Xii)
    yi = M_[i-1].predict(Xii)
    Xi['HI'] = yi
    data_.append(Xi)


# In[195]:


#with non-linear regression data
df_ = pd.concat(data_)
df_.sort_index(inplace = True)


# In[159]:



scaler = MinMaxScaler()
data = []
for i in range(1,7):
    Xi = train[train['OpsMode'] == i]
#     OpsMode = Xi['OpsMode']
#     Unit = Xi['A/C No.'] 
    Xii = Xi.drop(['A/C No.','OpsMode','Time'],axis = 1)
    Xi = Xi.drop(['T24','T30','T50','P30','Ps30','phi','BPR','OpsMode'],axis = 1)
    scaler.fit(Xii)
    Xii = scaler.transform(Xii)
    yi = M[i-1].predict(Xii)
    Xi['HI'] = yi
    data.append(Xi)
#     df = pd.DataFrame([Unit,OpsMode,yi],columns = ['A/C No.','OpsMode','HI'])
#     print(Xi)


# In[160]:


df = pd.concat(data)
df.sort_index(inplace = True)


# In[199]:


#with linear data
import seaborn as sns
sns.set()
d = df[df['A/C No.']==2]
ax = sns.lmplot(x = 'Time',y = 'HI',data = d)


# In[200]:


#with non-linear data
import seaborn as sns
sns.set()
d_ = df_[df_['A/C No.']==2]
ax = sns.lmplot(x = 'Time',y = 'HI',data = d_)


# In[168]:


from scipy.optimize import curve_fit
def func(x,a,b,c,d):
    return a*(np.exp(b*x+c)-np.exp(c))+d
x = np.array(d['Time'])
y = d['HI']
popt, pcov = curve_fit(func,x,y)
plt.plot(x,func(x,*popt))
# print(popt)
# print(pcov)


# In[58]:


from scipy.optimize import curve_fit
def func(x,a,b):
    return a*x+b
fitted_curve_parameters = []
covariance = []
for i in range(1,n+1):
    data = df[df['A/C No.'] == i]
    x = np.array(data['Time'])
    y = np.array(data['HI'])
    popt, pcov = curve_fit(func,x,y)
    fitted_curve_parameters.append(popt)
    covariance.append(pcov)
    


# In[60]:


test.head()


# In[61]:


scaler = MinMaxScaler()
data = []
for i in range(1,7):
    Xi = test[test['OpsMode'] == i]
    Xii = Xi.drop(['A/C No.','OpsMode'],axis = 1)
    Xi = Xi.drop(['T24','T30','T50','P30','Ps30','phi','BPR','OpsMode'],axis = 1)
    scaler.fit(Xii)
    Xii = scaler.transform(Xii)
    yi = M[i-1].predict(Xii)
    Xi['HI'] = yi
    data.append(Xi)


# In[65]:


df_test = pd.concat(data)
df_test.sort_index(inplace = True)


# In[64]:





# In[177]:


