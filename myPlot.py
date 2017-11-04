
# coding: utf-8

# In[1]:

# database
import pandas as pd
import numpy as np

# plotting 
import seaborn as sns
color = sns.color_palette()
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import matplotlib.dates as mdates
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


# In[2]:

#target variable
def describe_target(data, target, labels=[0,1], explode=[0.1,0], colors = ['lightcoral','lightblue']):
    '''
    The default is the binary classification.
    Change the parameters so that the len of labels, explode, and colors should all be = target.unique()
    '''
    sizes = data[target].value_counts().values
    patches, texts,autotexts= plt.pie(sizes, labels=labels, colors=colors, explode=explode, autopct="%1.1f%%",startangle=90)
    plt.title(target)
    plt.show()


# In[3]:

# For each labels in the feature show the distribution of the target labels.
def stacked_bar_plot(data, feature, target, axis):
    df = data[[feature, target]].groupby([feature, target], as_index=False).size().unstack(fill_value=0)
    df.plot(x=df.index, kind = 'bar', stacked = True, ax=axis)


def cate_plot(train_df, feature, target, order) :
    '''order is the order of the labels of the target'''
    # Pointplot the mean
    sns.factorplot(feature, target, data=train_df, size = 4, aspect = 3)
    fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,5))
    # feature against target
    stacked_bar_plot(train_df, feature, target, axis=axis1)
    # target against feature
    sns.countplot(x=target, hue=feature, data=train_df, order = order, ax=axis2)


# In[4]:

def num_plot(data, feature, target, bins = 10, order=[0, 1, 2]) :
    '''order is the order of the labels of the target'''
    fig, (axis1, axis2, axis3) = plt.subplots(3,1, figsize = (12,24))
    sns.violinplot(x = target, y = feature, data = data, ax = axis1)
    sns.pointplot(x = target, y = feature, data = data, ax = axis2)
    sns.stripplot(data[target],data[feature],jitter=True,order=order)
    plt.title(feature +' Vs ' + target);
    g = sns.FacetGrid(data, col=target, size = 5)
    g.map(plt.hist, feature, bins=bins)

def pivot_plot(data, row, col, target, s=500):
    df=data[[row, col, target]].groupby([row, col], as_index=False).mean()
    plt.scatter(x=df[col], y=df[row], c=df[target], s=s)
