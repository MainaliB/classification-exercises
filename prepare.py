#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import acquire
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


# In[41]:


def prep_iris():
    df = acquire.get_iris_data()
    cols_to_drop = ['species_id']
    df = df.drop(columns = cols_to_drop)
    df = df.rename(columns = {"species_name":"species"})
    df_dummies = df_dummies = pd.get_dummies(df[['species']], drop_first = True)
    df = pd.concat([df, df_dummies], axis = 1)
    return df


# In[42]:


def split_data(df):
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=322, 
                                        stratify=df.survived)
    train, validate = train_test_split(train_validate, test_size= 0.2,
                                       random_state = 322, stratify = train_validate.survived)
    return train, test, validate


def prep_titanic():
    titanic = acquire.get_titanic_data()
    titanic = titanic[~ titanic.embarked.isnull()]
    titanic = titanic[~ titanic.embark_town.isnull()]
    cols_to_drop = ['passenger_id','pclass', 'embark_town', 'deck']
    titanic = titanic.drop(columns = cols_to_drop)
    train, test, validate = split_data(titanic)
    return train, test, validate


# In[43]:


def impute(train, test, validate, my_strategy, column_list):
    
    imputer = SimpleImputer(strategy = my_strategy)
    
    imputer = imputer.fit(train[[column_list]])
    
    train[column_list] = imputer.transform(train[[column_list]])
    
    imputer = imputer.fit(test[[column_list]])
    
    test[column_list] = imputer.transform(test[[column_list]]) 
    
    imputer = imputer.fit(validate[[column_list]])
    
    validate[column_list] = imputer.transform(validate[[column_list]])
    
    return train, test, validate

