#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Shree
## Real Estate Sales prediction competition Kaggle
## Dataset is Kaggle data set and not owned by me
## Target is to predict prices of house with minimal RMSE


# In[2]:


# 1.Import Packages

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.utils import shuffle
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns
sns.set()


# Set options for display
pd.options.display.max_rows = 350

print('Packages imported sucessfully')


# In[3]:


# 2 Import Data

# 2.1 Import Data

dfRawREPrices=pd.read_csv('E:/Kaggle/RealEstatePrice/train.csv')
dfRawREPricesTest=pd.read_csv('E:/Kaggle/RealEstatePrice/test.csv')
# 2.2 Check for missing values
dfREPricesFull=dfRawREPrices.copy()
dfREPricesFullTest=dfRawREPricesTest.copy()
dfREPricesFull.isnull()


# In[4]:


# 3 Deleting and adding columns as per requirement

# Delete alley, poolQC,pool area column as it is mostly NA or 0 and won't affect price as such
# Additional columns will be dropped after analysis

# Training Data changes
dfREPrices=dfREPricesFull.drop(columns=['Alley','PoolArea','PoolQC'],inplace=False)
# Add column for overall age and remodeling age
dfREPrices['PropertyAge']=dfREPrices['YrSold']-dfREPrices['YearBuilt']
dfREPrices['RemodeledSinceYears']=dfREPrices['YrSold']-dfREPrices['YearRemodAdd']
lRowstobeChanged1=dfREPrices.RemodeledSinceYears<0
dfREPrices.loc[lRowstobeChanged1,'RemodeledSinceYears']=0


# Replicate Training changes on Testing data

dfREPricesTest=dfREPricesFullTest.drop(columns=['Alley','PoolArea','PoolQC'],inplace=False)

# Add column for overall age and remodeling age
dfREPricesTest['PropertyAge']=dfREPricesTest['YrSold']-dfREPricesTest['YearBuilt']
dfREPricesTest['RemodeledSinceYears']=dfREPricesTest['YrSold']-dfREPricesTest['YearRemodAdd']
lRowstobeChanged2=dfREPricesTest.RemodeledSinceYears<0
dfREPricesTest.loc[lRowstobeChanged2,'RemodeledSinceYears']=0
dfREPricesTest['PropertyAge']=dfREPricesTest['YrSold']-dfREPrices['YearBuilt']

#print(dfREPricesTest.head())


# In[5]:


# 4 Descriptive analysis of Price wrt Categorical features

# 4.1 Price Histogram

fig1,ax=plt.subplots()
plt.hist(dfREPrices['SalePrice'],100,color='blue')
plt.title('Price Histogram')
plt.xlabel('Price in USD')
plt.ylabel('Frequency')
plt.show()

dfREPrices['LandRate']=dfREPrices['SalePrice']/dfREPrices['LotArea']
fig2,ax=plt.subplots()
plt.hist(dfREPrices['LandRate'],100,color='blue')
plt.title('LandRate Histogram')
plt.xlabel('Price in USD')
plt.ylabel('Frequency')
plt.show()

dfREPrices['LvAreaRate']=dfREPrices['SalePrice']/dfREPrices['GrLivArea']
fig2,ax=plt.subplots()
plt.hist(dfREPrices['GrLivArea'],100,color='blue')
plt.title('LandRate Histogram')
plt.xlabel('Price in USD')
plt.ylabel('Frequency')
plt.show()


lPriceQuantile=dfREPrices.SalePrice.quantile(0.99)
#print('Price 99th Percentile')
#print(lPriceQuantile)

# 4.2 Neighbourhoud

# Price
dfRENeighborhood=dfREPrices.groupby('Neighborhood').mean()
dfRENeighborhood.sort_values(by='SalePrice',inplace=True)
lNeighborhood=list(dfRENeighborhood.index.values)
lNeighborhoodSalesPrice=list(dfRENeighborhood['SalePrice'])

# land Rate 
# Living Area Rate


# 4.3 Zoning

dfREMSZoning=dfREPrices.groupby('MSZoning').mean()
dfREMSZoning.sort_values(by='SalePrice',inplace=True)
lREMSZoning=list(dfREMSZoning.index.values)
lSalesPriceZoning=list(dfREMSZoning['SalePrice'])

# 4.4 Utilities

dfREUtilities=dfREPrices.groupby('Utilities').mean()
dfREUtilities.sort_values(by='SalePrice',inplace=True)
lREUtilities=list(dfREUtilities.index.values)
lSalesPriceUtilities=list(dfREUtilities['SalePrice'])

# 4.5 BldgType

dfREBldgType=dfREPrices.groupby('BldgType').mean()
dfREBldgType.sort_values(by='SalePrice',inplace=True)
lREBldgType=list(dfREBldgType.index.values)
lSalesPriceBldgType=list(dfREBldgType['SalePrice'])


# 4.6 House Style
dfREHouseStyle=dfREPrices.groupby('HouseStyle').mean()
dfREHouseStyle.sort_values(by='SalePrice',inplace=True)
lREHouseStyle=list(dfREHouseStyle.index.values)
lSalesPriceHouseStyle=list(dfREHouseStyle['SalePrice'])

# 4.7 YrSold
dfREYrSold=dfREPrices.groupby('YrSold').mean()
dfREYrSold.sort_values(by='SalePrice',inplace=True)
lREYrSold=list(dfREYrSold.index.values)
lSalesPriceYrSold=list(dfREYrSold['SalePrice'])

# 4.8 SaleCondition
dfRESaleType=dfREPrices.groupby('SaleType').mean()
dfRESaleType.sort_values(by='SalePrice',inplace=True)
lRESaleType=list(dfRESaleType.index.values)
lSalesPriceSaleType=list(dfRESaleType['SalePrice'])

# 4.9 SaleType
dfRESaleCondition=dfREPrices.groupby('SaleCondition').mean()
dfRESaleCondition.sort_values(by='SalePrice',inplace=True)
lRESaleCondition=list(dfRESaleCondition.index.values)
lSalesPriceSaleCondition=list(dfRESaleCondition['SalePrice'])

# 4.10 Consolidated Bar-chart for Non-Numeric features
fig8, axes=plt.subplots(4,2,figsize=(12,12))
b1=axes[0,0].barh(lNeighborhood,lNeighborhoodSalesPrice,color='blue')
axes[0,0].set_title('Neighbourhood vs AvgSalePrice')
b2=axes[0,1].barh(lREMSZoning,lSalesPriceZoning,color='orange')
axes[0,1].set_title('MSZoning vs AvgSalePrice')
b3=axes[1,0].barh(lREUtilities,lSalesPriceUtilities,color='blue')
axes[1,0].set_title('PriceUtilities vs AvgSalePrice')
b4=axes[1,1].barh(lREBldgType,lSalesPriceBldgType,color='orange')
axes[1,1].set_title('BldgType vs AvgSalePrice')
b5=axes[2,0].barh(lREHouseStyle,lSalesPriceHouseStyle,color='blue')
axes[2,0].set_title('HouseStyle vs AvgSalePrice')
b6=axes[2,1].barh(lREYrSold,lSalesPriceYrSold,color='orange')
axes[2,1].set_title('YearSold vs AvgSalePrice')
b7=axes[3,0].barh(lRESaleType,lSalesPriceSaleType,color='blue')
axes[3,0].set_title('SaleType vs AvgSalePrice')
b8=axes[3,1].barh(lRESaleCondition,lSalesPriceSaleCondition,color='orange')
axes[3,1].set_title('SaleCondition vs AvgSalePrice')


# In[6]:


# 5 Check relationship of numerical features with price
# 5.1 Create Facet grid for numerical features
# 5.2 Get R^2 for linear regression for all numberical features against sales price


# 5.1 Lot area,GrLivArea,GarageArea

CReg=LinearRegression()
x=dfREPrices[['LotArea','GrLivArea','GarageArea']]
y=dfREPrices['SalePrice']
CReg.fit(x,y)
#print('Intercept',CReg.intercept_)
#print('Coefficients',CReg.coef_)
vR2= CReg.score(x,y)
#dfRegResult=pd.DataFrame(data=vR2,columns='R2Score',index=['LotArea','GrLivArea','GarageArea'])
#print('Regression Results',vR2)

# 5.2 Scatterplot for Lot area and Sale Price

fig9, axes=plt.subplots(3,2,figsize=(12,12))
s1=axes[0,0].scatter(dfREPrices['LotArea'],dfREPrices['SalePrice'],marker='o',color='blue',alpha=0.5)
axes[0,0].set_title('LotArea vs SalePrice')
s2=axes[1,0].scatter(dfREPrices['GrLivArea'],dfREPrices['SalePrice'],marker='o',color='blue',alpha=0.5)
axes[1,0].set_title('GrLivArea vs SalePrice')
s3=axes[2,0].scatter(dfREPrices['GarageArea'],dfREPrices['SalePrice'],marker='o',color='blue',alpha=0.5)
axes[2,0].set_title('GarageArea vs SalePrice')
s4=axes[0,1].scatter(dfREPrices['LotFrontage'],dfREPrices['SalePrice'],marker='o',color='blue',alpha=0.5)
axes[0,1].set_title('LotFrontage vs SalePrice')
s5=axes[1,1].scatter(dfREPrices['BsmtUnfSF'],dfREPrices['SalePrice'],marker='o',color='blue',alpha=0.5)
axes[1,1].set_title('BsmtUnfSF vs SalePrice')
s6=axes[2,1].scatter(dfREPrices['TotalBsmtSF'],dfREPrices['SalePrice'],marker='o',color='blue',alpha=0.5)
axes[2,1].set_title('TotalBsmtSF vs SalePrice')


# In[7]:


# 6 Drop outliers in features
# 6.1 Drop outlier data in lot area
dfREPrices2=pd.DataFrame
dfREPrices2=dfREPrices.copy()
dfREPrices2.drop(dfREPrices2[dfREPrices2['LotArea']>100000].index,inplace=True)
#print(dfREPrices2.head())
# 6.2 Scatterplot for Lot area and Sale Price

fig8, ax=plt.subplots()
plt.scatter(dfREPrices2['LotArea'],dfREPrices2['SalePrice'],marker='o',color='blue',alpha=0.5)
plt.xlabel('LotArea')
plt.ylabel('SalePrice')
plt.title('LotArea vs SalePrice')
plt.show()


# In[8]:


# 7 Data cleaning and pre-processing
#Count null cells in columns for dropping columns or filling cells

lColumnsList=dfREPrices2.isnull().sum()
#print('Blank Cells Listing')
#print(lColumnsList)

# Drop Fence and Misc features columns due to high % of NA cells
dfREPrices3=pd.DataFrame()
dfREPrices3=dfREPrices2.copy()
dfREPrices3=dfREPrices2.drop(columns=['Fence','MiscFeature'],axis=1,inplace=False)

# Fill NA cells assuming 0 for numerical features and No for categorical features
dfREPrices3 = dfREPrices3.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('No'))


# Replicating changes on Test Data
dfREPricesTest2=pd.DataFrame()
dfREPricesTest2=dfREPricesTest.copy()
dfREPricesTest2=dfREPricesTest2.drop(columns=['Fence','MiscFeature'],axis=1,inplace=False)
dfREPricesTest2 = dfREPricesTest2.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('No'))
print(dfREPricesTest2.head())


# In[9]:


# 8 Column merging 
# Using year data to create feature columns for age of house, age of renovation, garage age and deleting year columns

dfREPrices3['GarageAge']=dfREPrices3['YrSold']-dfREPrices3['GarageYrBlt']
lRowstobeChanged=dfREPrices3.GarageAge>100
dfREPrices3.loc[lRowstobeChanged,'GarageAge']=dfREPrices3.loc[lRowstobeChanged,'PropertyAge']
dfREPrices3.drop(columns=['MoSold','YrSold','YearBuilt','YearRemodAdd','LvAreaRate','Id'],inplace=True)
dfREPrices3['1st2ndFlrSF']=dfREPrices3['1stFlrSF']+dfREPrices3['2ndFlrSF']
dfREPrices3.drop(columns=['1stFlrSF','2ndFlrSF','LandRate'],inplace=True)
#print(dfREPrices3.columns)

# Replicating Changes on Test Data

dfREPricesTest2['GarageAge']=dfREPricesTest2['YrSold']-dfREPricesTest2['GarageYrBlt']
lRowstobeChangedTest2=dfREPricesTest2.GarageAge>100
dfREPricesTest2.loc[lRowstobeChangedTest2,'GarageAge']=dfREPricesTest2.loc[lRowstobeChangedTest2,'PropertyAge']
dfREPricesTest2.drop(columns=['MoSold','YrSold','YearBuilt','YearRemodAdd','Id'],inplace=True)
dfREPricesTest2['1st2ndFlrSF']=dfREPricesTest2['1stFlrSF']+dfREPricesTest2['2ndFlrSF']
dfREPricesTest2.drop(columns=['1stFlrSF','2ndFlrSF'],inplace=True)



# In[10]:


# 9 Dummy Generation & Scaling

dfREPrices3['MSSubClass']=dfREPrices3.MSSubClass.astype('object')
dfREPrices3['OverallQual']=dfREPrices3.OverallQual.astype('object')
dfREPrices3['OverallCond']=dfREPrices3.OverallCond.astype('object')
IxCatCol = dfREPrices3.select_dtypes(include=['object', 'bool']).columns
IxNumCol = dfREPrices3.select_dtypes(include=['int64', 'float64']).columns


dfREPrices3Num=dfREPrices3[IxNumCol]
dfREPrices3Cat=dfREPrices3[IxCatCol]

#print(dfREPrices3Num.head())
#print(dfREPrices3Cat.head())

CScaler=StandardScaler()
dfREPrices3NumSca=pd.DataFrame(data=CScaler.fit_transform(dfREPrices3Num),columns=dfREPrices3Num.columns)
#print(dfREPrices3NumSca.head())
dfREPrices3NumScaLog=np.log10(dfREPrices3Num.copy())

#print(dfREPrices3NumSca.head())
dfCatDummies=pd.get_dummies(dfREPrices3Cat,prefix=None,prefix_sep='_')
dfREPrices3ScaTrans=pd.concat([dfREPrices3NumScaLog,dfCatDummies],axis=1)

# Replicating changes for Test Data

dfREPricesTest2['MSSubClass']=dfREPricesTest2.MSSubClass.astype('object')
dfREPricesTest2['OverallQual']=dfREPricesTest2.OverallQual.astype('object')
dfREPricesTest2['OverallCond']=dfREPricesTest2.OverallCond.astype('object')
IxCatColTest = dfREPricesTest2.select_dtypes(include=['object', 'bool']).columns
IxNumColTest = dfREPricesTest2.select_dtypes(include=['int64', 'float64']).columns

dfREPricesTest2Num=dfREPricesTest2[IxNumColTest]
dfREPricesTest2Cat=dfREPricesTest2[IxCatColTest]
dfREPricesTest2NumSca=pd.DataFrame(data=CScaler.fit_transform(dfREPricesTest2Num),columns=dfREPricesTest2Num.columns)
dfREPricesTest2NumScaLog=np.log10(dfREPricesTest2NumSca.copy())
dfCatDummiesTest2=pd.get_dummies(dfREPricesTest2Cat,prefix=None,prefix_sep='_')
dfREPricesTest2ScaTrans=pd.concat([dfREPricesTest2NumScaLog,dfCatDummiesTest2],axis=1)
#print(dfREPricesTest2ScaTrans.head())


# In[11]:


# 10 Pre-processing 

#Shuffle data
dfREPrices3ScaTransShuf=shuffle(dfREPrices3ScaTrans, random_state=5)
#print(dfREPrices3ScaTransShuf.head())
#print(dfREPrices3ScaTransShuf.columns)
dfREPricesTrain,dfREPricesTest=train_test_split(dfREPrices3ScaTransShuf, shuffle=False,test_size=0.2)
#print(dfREPricesTrain.info())
#print(dfREPricesTest.info())

#Replicating changes for Test Data
dfREPricesTest2ScaTransShuf=shuffle(dfREPricesTest2ScaTrans, random_state=5)


# In[12]:


# 11 Data preparation for XGB Regression model

# 11.1 Separate Regressors and target
dfREPricesTrainRegs=dfREPricesTrain.copy()
dfREPricesTrainRegs.drop(columns='SalePrice',inplace=True)
dfREPricesTestRegs=dfREPricesTest.copy()
dfREPricesTestRegs.drop(columns='SalePrice',inplace=True)

dfREPricesTrainTarg=dfREPricesTrain['SalePrice'].copy()
dfREPricesTestTarg=dfREPricesTest['SalePrice'].copy()


#print('All databases')
#print(dfREPricesTrainTarg.head())
#print(dfREPricesTestTarg.head())

# 11.2  Convert df to Numpy arrays

aTrainRegs=np.asarray(dfREPricesTrainRegs)
aTestRegs=np.asarray(dfREPricesTestRegs)
aTrainTarg=np.asarray(dfREPricesTrainTarg)
aTestTarg=np.asarray(dfREPricesTestTarg)



# Compare columns and add missing columns to testing data
lMissingColumns= list(dfREPricesTrainRegs.columns.difference(dfREPricesTest2ScaTransShuf.columns))
#df1.columns.difference(df2.columns)
#print(lMissingColumns)
dfMissingColumns=pd.DataFrame(np.zeros((len(dfREPricesTest2ScaTransShuf),len(lMissingColumns))), columns=lMissingColumns)
print(len(dfMissingColumns))
#print(dfMissingColumns.head())
dfREPricesTest2ScaTransShuf=pd.concat([dfREPricesTest2ScaTransShuf,dfMissingColumns],axis=1)
#print('Testing columns',len(dfREPricesTest2ScaTransShuf.columns))
#print('Training columns',len(dfREPricesTrainRegs.columns))
lTestingColumnsExtra=list(dfREPricesTest2ScaTransShuf.columns.difference(dfREPricesTrainRegs.columns))
#print('Excess Columns in Testing')
#print(lTestingColumnsExtra)

# Drop unknown columsn in Testing
dfREPricesTest2ScaTransShuf=dfREPricesTest2ScaTransShuf.drop(columns=['Exterior1st_No', 'Exterior2nd_No', 'Functional_No',
                                                                      'KitchenQual_No', 'MSSubClass_150', 'MSZoning_No', 
                                                                      'SaleType_No', 'Utilities_No'],inplace=False)

# Replicating changes on Test Data
aPredRegs=np.asarray(dfREPricesTest2ScaTransShuf)


# In[13]:


# 12 XGB model 


# 12.1 Xgb model

dcProgress={}
cModel = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.4, gamma=0,learning_rate=0.03,
                          max_depth=3,min_child_weight=4,n_estimators=30000,                                                            
                          reg_alpha=0.75,reg_lambda=0.45,subsample=0.6,early_stopping_rounds=25000,
                          eval_metric='rmse',verbose=True,seed=34)

cModel.fit(aTrainRegs,aTrainTarg)
#dsProgress=cModel.evals_result()
#print('Epoch Results',dcProgress)

# Predict prices on test data

vPredPrices=cModel.predict(aTestRegs)
vSqErrorRt=np.sqrt(np.square(aTestTarg-vPredPrices))
#print(vPredPrices)

vRMSE=np.sqrt(mean_squared_error(aTestTarg,vPredPrices))

print('Model Error',vRMSE)



fig10,ax=plt.subplots()
plt.scatter(aTestTarg,np.round(vSqErrorRt*100/aTestTarg,2),marker='o',color='blue',alpha=0.5)
ax.text(0.05, 0.95, np.round(vRMSE,1), transform=ax.transAxes, fontsize=14,
        verticalalignment='top')
plt.title('Error vs SalePrice')


# In[14]:


# 13 Model Accuracy analysis and improvement

dfModelAnalysis=pd.DataFrame()
dfModelAnalysis['FeatureScore']=cModel.feature_importances_
#dfModelAnalysis['FeatureName']=cModel.feature_names
#dfModelAnalysis.sort_values(by='FeatureScore',inplace=True)
print(dfModelAnalysis)

# Plot feature importance

#fig11, ax=plt.subplots(figsize=(15,35))
#plt.barh(dfModelAnalysis['FeatureName'],dfModelAnalysis['FeatureScore'],color='blue')
#plt.title('Feature vs Feature Importance')
#plt.show()



#dfREPricesTest2ScaTransShuf('')

# Predict Prices on Prediction Data
#print(len(aPredRegs))
vPredPricesTestLog=cModel.predict(aPredRegs)
vPredPricesTest=10**vPredPricesTestLog


#print(vPredPricesTest)


# In[15]:


# 14 Convert to csv for submission

dfSubmission=pd.DataFrame()
dfSubmission['Id']=dfREPricesFullTest['Id'].copy()
dfSubmission['SalePrice']=vPredPricesTest
dfSubmission.to_csv('E:/Kaggle/RealEstatePrice/ng108.csv')
print('Jai Shree Ram')


# In[ ]:




