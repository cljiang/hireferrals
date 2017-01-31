# This python script is to examine the relation between partial pressure of CO2, pco2, 
# with sea surface temperature, sst,
# and sea surface salinity, sss,
# and chlorophil-a, chl,
# and mixed layer depths, mld,
# in one of the best climate models, GFDL-ESM2G,
# during 11 years of period from 2002 to 2012 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from math import radians, cos, sin, asin, sqrt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sknn.mlp import Regressor, Layer

pco2_old = pd.read_csv('pco2_ESM2G.csv')
pco2 = pd.DataFrame(pco2_old.values.reshape((131*25235,1)), columns=['pco2'])
pco2_new = pco2.dropna()
plt.plot(pco2_new)
plt.show()
pco2_new.hist()
plt.show()

sst_old = pd.read_csv('sst_ESM2G.csv')
sst = pd.DataFrame(sst_old.values.reshape((131*25235,1)), columns=['sst'])
sst_new = sst.dropna()
plt.plot(sst_new)
plt.show()
sst_new.hist()
plt.show()

sss_old = pd.read_csv('sss_ESM2G.csv')
sss = pd.DataFrame(sss_old.values.reshape((131*25235,1)), columns=['sss'])
sss_new = sss.dropna()
plt.plot(sss_new)
plt.show()
sss_new.hist()
plt.show()

chl_old = pd.read_csv('chl_ESM2G.csv')
chl = pd.DataFrame(chl_old.values.reshape((131*25235,1)), columns=['chl'])
chl_new = chl.dropna()
plt.plot(chl_new)
plt.show()
chl_new.hist()
plt.show()

mld_old = pd.read_csv('mld_ESM2G.csv')
mld = pd.DataFrame(mld_old.values.reshape((131*25235,1)), columns=['mld'])
mld_new = mld.dropna()
plt.plot(mld_new)
plt.show()
mld_new.hist()
plt.show()

print pearsonr(pco2_new, sst_new)
print pearsonr(pco2_new, sss_new)
print pearsonr(pco2_new, chl_new)
print pearsonr(pco2_new, mld_new)

data_old = [pco2_new, sst_new, sss_new, chl_new, mld_new]
data = pd.concat(data_old,axis=1)

data_new = data.dropna(how='any')

feature_cols=['sst','sss','chl','mld']                                                                                             
X=data_new[feature_cols]
y=data_new.pco2
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=2)

#.linear regression %pco2: 9.4
linreg=LinearRegression()
linreg.fit(X_train,y_train)
zip(feature_cols,linreg.coef_)
y_pred=linreg.predict(X_test)
print metrics.mean_absolute_error(y_test,y_pred)

#.Decision tree %pco2: 7.6
np.random.seed(100)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
dt=DecisionTreeRegressor()
dt.fit(X_train,y_train)
y_pred_dt=dt.predict(X_test)
print metrics.mean_absolute_error(y_test,y_pred_dt)

#.random forest %pco2: 6.1
rf=RandomForestRegressor(random_state=42)
rf.fit(X_train,y_train)
y_pred_rf=rf.predict(X_test)
print metrics.mean_absolute_error(y_test,y_pred_rf)

plt.scatter(y_test,y_pred_rf)
plt.xlabel('measured pco2 values (uatm)')
plt.ylabel('predicted pco2 values (uatm)')
plt.title('Random Forest (mean absolute error 6uatm)')
plt.savefig('pco2_rf.eps', format='eps', dpi=1000)








