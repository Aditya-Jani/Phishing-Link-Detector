#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from ExtractingFeatures import urlFeatures


# In[3]:


df = pd.read_csv("url_specifications.csv")
df.head()


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(df[["URL Length", "Non Standard Ports", "HTTPS", "Special Characters", "Numeric Characters", "Number of Redirects", "Shortening of URL"]], df.Phishing, test_size=0.1)


# In[5]:


model = LogisticRegression()


# In[6]:


model.fit(X_train, y_train)


# In[7]:


model.score(X_test, y_test)


# %%
