---
layout: page
title: Student Intervention System-EDA
permalink: md/ds/si-mca/
---

### Building a student intervention system: MCA for dimensionality reduction

In previous posts I explained how Multiple Correspondance Analysis (MCA) can be applied to get lower dimensional representation of a categorical data. In this post, I will apply it to a data set comprising behavioral and demographic data collected in 395 students. I will first apply MCA to the student data, and check whats the least number of components required to make comparable predictions. The data can be downloaded from http://tinyurl.com/h2wvk2r. Two previous posts with exploratory analysis and details on MCA can be found at 

- Introduction to MCA: http://wp.me/p7EqYU-1o
- Building Student Intervention System- EDA: http://wp.me/p7EqYU-1N.

If you are not familiar with MCA, Benzécri and Greenacre corrections, please read the introduction to MCA blog before proceeding.


After loading the data, I grouped age into 4 categories, one below 16, other between 16 and 17, other between 17 and 18, and final above 19. I also grouped absences into fewer categories, less than 3 days, between 3 and 7, 7 and 14, 14 and 21, and more than 21. All the groups include upper value of the interval. The main reason for doing this was to have only categorical data, so MCA can be applied. After making dummy variables each categorical variable, there were a total of 104 predictor variables. 

104 features are a lot for any model to overfit, further having 104 variable implies that we will require a lot of data for training, this is also referred as the curse of dimensionality. I therefore applied MCA, a dimensionailty reduction technique for categorical variables. MCA suggested that only 5 dimension were sufficient to explain most of the variance in the data.




**** Important: I wrote functions for data manipulation and plotting and put them in a separate file udacityP2_funs****

First I load important libraries. 


```python
# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
%pylab inline
# Read student data
from udacityP2_funs import * 
from mca import *
from mpl_toolkits.mplot3d import Axes3D

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss, accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_regression,chi2
%load_ext autoreload
%autoreload 2
```

    Populating the interactive namespace from numpy and matplotlib
    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload


    WARNING: pylab import has clobbered these variables: ['clf']
    `%matplotlib` prevents importing * from pylab and numpy


#### Loading data
I next load data in python and modify the age and absence data from numeric to categorical. I grouped age into 4 categories, one below 16, other between 16 and 17, other between 17 and 18, and final above 19. I also grouped absences into fewer categories, less than 3 days, between 3 and 7, 7 and 14, 14 and 21, and more than 21. All the groups include upper value of the interval. The main reason for doing this was to have only categorical data, so MCA can be applied. 


```python
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"

student_data['absentee_class'] = student_data.apply(lambda row: absentee_class(row['absences']), axis=1)
student_data['age_class'] = student_data.apply(lambda row: age_class(row['age']), axis=1)
#student_data = student_data.drop('age',axis=1)
```

    Student data read successfully!



```python
student_data[['absentee_class','absences']].head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>absentee_class</th>
      <th>absences</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&gt;3 &amp; &lt;=7</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&gt;3 &amp; &lt;=7</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>&gt;7 &amp; &lt;=14</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>&lt;=3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>&gt;3 &amp; &lt;=7</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
student_data[['age_class','age']].head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age_class</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&gt;17 &amp; &lt;=18</td>
      <td>18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&gt;16 &amp; &lt;=17</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>&lt;=16</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>&lt;=16</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>&lt;=16</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
student_data = student_data.drop('absences',axis=1)
student_data = student_data.drop('age',axis=1)

target = student_data['passed']
features = student_data.drop('passed',axis=1)
features.columns
```




    Index([u'school', u'sex', u'address', u'famsize', u'Pstatus', u'Medu', u'Fedu',
           u'Mjob', u'Fjob', u'reason', u'guardian', u'traveltime', u'studytime',
           u'failures', u'schoolsup', u'famsup', u'paid', u'activities',
           u'nursery', u'higher', u'internet', u'romantic', u'famrel', u'freetime',
           u'goout', u'Dalc', u'Walc', u'health', u'absentee_class', u'age_class'],
          dtype='object')



#### Making dummy variables

I next make dummy variables to populate the indicator matrix for MCA. After making the dummy variables for each possible score, there were 104 variables. 


```python
var_index = []
val_index = []
col_names_DF = []


unique_names = {'school' : ['GP', 'MS'] ,
        'sex' : ['F','M'] ,
        'address' : ['U' ,'R'] ,
        'famsize' : ['GT3', 'LE3'] ,
        'Pstatus' : ['A' ,'T'] ,
        'Medu' : [0 ,1 ,2, 3 ,4] ,
        'Fedu' : [0, 1, 2 ,3, 4] ,
        'Mjob' : ['at_home', 'health', 
                  'other', 'services', 'teacher'] ,
        'Fjob' : ['teacher' ,'other', 
                  'services', 'health','at_home'] ,
        'reason' : ['course', 'other' ,
                    'home' ,'reputation'] ,
        'guardian' : ['mother', 'father' ,
                      'other'] ,
        'traveltime' : [1,2,3,4] ,
        'studytime' : [1,2,3,4] ,
        'failures' : [0,1,2,3] ,
        'schoolsup' : ['yes', 'no'] ,
        'famsup' : ['yes', 'no'] ,
        'paid' : ['yes', 'no'] ,
        'activities' : ['yes', 'no'] ,
        'nursery' : ['yes', 'no'] ,
        'higher' : ['yes', 'no'] ,
        'internet' : ['yes', 'no'] ,
        'romantic' : ['yes', 'no'] ,
        'famrel' : [1,2,3,4,5] ,
        'freetime' : [1,2,3,4,5] ,
        'goout' : [1,2,3,4,5] ,
        'Dalc' : [1,2,3,4,5] ,
        'Walc' : [1,2,3,4,5] ,
        'health' : [1,2,3,4,5] ,
        'absentee_class' : ['<=3','>3 & <=7', 
                            '>7 & <=14', 
                            '>14 & <=21','>21'] ,
        'age_class' : ['<=16','>16 & <=17',
                       '>17 & <=18','>18'] ,}

col_names = features.columns


print 'Number of features is %d'%len(col_names) 
i = 0
str_row = []
for c_name in col_names:
    print 'Feature value :' ,features[c_name][i] 
    print 'Possible values:' ,unique_names[c_name]
    var_index += [c_name]*len(unique_names[c_name])
    val_index += unique_names[c_name]
    print 'Ind-matrix values:',[1 if (features[c_name][i] == c_val) else 0\
     for c_val in unique_names[c_name]]
    
    col_names_DF+=[c_name+'-'+str(c_val) for c_val in unique_names[c_name]]
    
    str_row+=[1 if (features[c_name][i] == c_val) else 0\
     for c_val in unique_names[c_name]]
    
print 'Number of variables :', len(str_row)
```

    Number of features is 30
    Feature value : GP
    Possible values: ['GP', 'MS']
    Ind-matrix values: [1, 0]
    Feature value : F
    Possible values: ['F', 'M']
    Ind-matrix values: [1, 0]
    Feature value : U
    Possible values: ['U', 'R']
    Ind-matrix values: [1, 0]
    Feature value : GT3
    Possible values: ['GT3', 'LE3']
    Ind-matrix values: [1, 0]
    Feature value : A
    Possible values: ['A', 'T']
    Ind-matrix values: [1, 0]
    Feature value : 4
    Possible values: [0, 1, 2, 3, 4]
    Ind-matrix values: [0, 0, 0, 0, 1]
    Feature value : 4
    Possible values: [0, 1, 2, 3, 4]
    Ind-matrix values: [0, 0, 0, 0, 1]
    Feature value : at_home
    Possible values: ['at_home', 'health', 'other', 'services', 'teacher']
    Ind-matrix values: [1, 0, 0, 0, 0]
    Feature value : teacher
    Possible values: ['teacher', 'other', 'services', 'health', 'at_home']
    Ind-matrix values: [1, 0, 0, 0, 0]
    Feature value : course
    Possible values: ['course', 'other', 'home', 'reputation']
    Ind-matrix values: [1, 0, 0, 0]
    Feature value : mother
    Possible values: ['mother', 'father', 'other']
    Ind-matrix values: [1, 0, 0]
    Feature value : 2
    Possible values: [1, 2, 3, 4]
    Ind-matrix values: [0, 1, 0, 0]
    Feature value : 2
    Possible values: [1, 2, 3, 4]
    Ind-matrix values: [0, 1, 0, 0]
    Feature value : 0
    Possible values: [0, 1, 2, 3]
    Ind-matrix values: [1, 0, 0, 0]
    Feature value : yes
    Possible values: ['yes', 'no']
    Ind-matrix values: [1, 0]
    Feature value : no
    Possible values: ['yes', 'no']
    Ind-matrix values: [0, 1]
    Feature value : no
    Possible values: ['yes', 'no']
    Ind-matrix values: [0, 1]
    Feature value : no
    Possible values: ['yes', 'no']
    Ind-matrix values: [0, 1]
    Feature value : yes
    Possible values: ['yes', 'no']
    Ind-matrix values: [1, 0]
    Feature value : yes
    Possible values: ['yes', 'no']
    Ind-matrix values: [1, 0]
    Feature value : no
    Possible values: ['yes', 'no']
    Ind-matrix values: [0, 1]
    Feature value : no
    Possible values: ['yes', 'no']
    Ind-matrix values: [0, 1]
    Feature value : 4
    Possible values: [1, 2, 3, 4, 5]
    Ind-matrix values: [0, 0, 0, 1, 0]
    Feature value : 3
    Possible values: [1, 2, 3, 4, 5]
    Ind-matrix values: [0, 0, 1, 0, 0]
    Feature value : 4
    Possible values: [1, 2, 3, 4, 5]
    Ind-matrix values: [0, 0, 0, 1, 0]
    Feature value : 1
    Possible values: [1, 2, 3, 4, 5]
    Ind-matrix values: [1, 0, 0, 0, 0]
    Feature value : 1
    Possible values: [1, 2, 3, 4, 5]
    Ind-matrix values: [1, 0, 0, 0, 0]
    Feature value : 3
    Possible values: [1, 2, 3, 4, 5]
    Ind-matrix values: [0, 0, 1, 0, 0]
    Feature value : >3 & <=7
    Possible values: ['<=3', '>3 & <=7', '>7 & <=14', '>14 & <=21', '>21']
    Ind-matrix values: [0, 1, 0, 0, 0]
    Feature value : >17 & <=18
    Possible values: ['<=16', '>16 & <=17', '>17 & <=18', '>18']
    Ind-matrix values: [0, 0, 1, 0]
    Number of variables : 104



```python
col_index = pd.MultiIndex.from_arrays([var_index, val_index], 
                                      names=['variable', 'value'])
ind_data = []
for i in np.arange(0,len(features)):
    ind_data.append(make_row_indicator(features,i,0))
    
table1 = pd.DataFrame(data=ind_data, columns=col_index)
table1.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>variable</th>
      <th colspan="2" halign="left">school</th>
      <th colspan="2" halign="left">sex</th>
      <th colspan="2" halign="left">address</th>
      <th colspan="2" halign="left">famsize</th>
      <th colspan="2" halign="left">Pstatus</th>
      <th>...</th>
      <th>health</th>
      <th colspan="5" halign="left">absentee_class</th>
      <th colspan="4" halign="left">age_class</th>
    </tr>
    <tr>
      <th>value</th>
      <th>GP</th>
      <th>MS</th>
      <th>F</th>
      <th>M</th>
      <th>U</th>
      <th>R</th>
      <th>GT3</th>
      <th>LE3</th>
      <th>A</th>
      <th>T</th>
      <th>...</th>
      <th>5</th>
      <th>&lt;=3</th>
      <th>&gt;3 &amp; &lt;=7</th>
      <th>&gt;7 &amp; &lt;=14</th>
      <th>&gt;14 &amp; &lt;=21</th>
      <th>&gt;21</th>
      <th>&lt;=16</th>
      <th>&gt;16 &amp; &lt;=17</th>
      <th>&gt;17 &amp; &lt;=18</th>
      <th>&gt;18</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 104 columns</p>
</div>




```python
ind_data_DF = pd.DataFrame(ind_data,columns = col_names_DF)
ind_data_DF.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>school-GP</th>
      <th>school-MS</th>
      <th>sex-F</th>
      <th>sex-M</th>
      <th>address-U</th>
      <th>address-R</th>
      <th>famsize-GT3</th>
      <th>famsize-LE3</th>
      <th>Pstatus-A</th>
      <th>Pstatus-T</th>
      <th>...</th>
      <th>health-5</th>
      <th>absentee_class-&lt;=3</th>
      <th>absentee_class-&gt;3 &amp; &lt;=7</th>
      <th>absentee_class-&gt;7 &amp; &lt;=14</th>
      <th>absentee_class-&gt;14 &amp; &lt;=21</th>
      <th>absentee_class-&gt;21</th>
      <th>age_class-&lt;=16</th>
      <th>age_class-&gt;16 &amp; &lt;=17</th>
      <th>age_class-&gt;17 &amp; &lt;=18</th>
      <th>age_class-&gt;18</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 104 columns</p>
</div>




```python
ncols = len(col_names)
mca_ben = MCA(ind_data_DF, ncols=ncols)
```

#### MCA : Eigen values and explained variance. 

As most of the variables are formed by making dummy variables based on the same categorical variable, there is a lot of redundant information in the factors. Therefore, a suitable dimensionality reduction method can identify the main design features. I next apply MCA to identify the important dimensions, and truncate the dataset based on explained variance ratios. I applied MCA with Benzécri and Greenacre corrections, and without. The corresponding eigen values and variance ratios are presented below. With 15 features, more than 95% of the variance with Benzécri and 66% with Greenacre correction is explained.


```python
N_eig = 3
mca_ben = MCA(ind_data_DF, ncols=ncols)
mca_ind = MCA(ind_data_DF, ncols=ncols, benzecri=False)
data = {'Iλ': pd.Series(mca_ind.L[:N_eig]),
        'τI': mca_ind.expl_var(greenacre=False, N=N_eig),
        'Zλ': pd.Series(mca_ben.L[:N_eig]),
        'τZ': mca_ben.expl_var(greenacre=False, N=N_eig),
        'cλ': pd.Series(mca_ben.L[:N_eig]),
        'τc': mca_ben.expl_var(greenacre=True, N=N_eig)}

# 'Indicator Matrix', 'Benzecri Correction', 'Greenacre Correction'
columns = ['Iλ', 'τI', 'Zλ', 'τZ', 'cλ', 'τc']
table2 = pd.DataFrame(data=data, columns=columns).fillna(0)
table2.index += 1
table2.loc['Σ'] = table2.sum()
table2.index.name = 'Factor'

table2
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Iλ</th>
      <th>τI</th>
      <th>Zλ</th>
      <th>τZ</th>
      <th>cλ</th>
      <th>τc</th>
    </tr>
    <tr>
      <th>Factor</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.117244</td>
      <td>0.047531</td>
      <td>0.007535</td>
      <td>0.337490</td>
      <td>0.007535</td>
      <td>0.236254</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.094438</td>
      <td>0.038286</td>
      <td>0.003996</td>
      <td>0.178970</td>
      <td>0.003996</td>
      <td>0.125285</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.077321</td>
      <td>0.031346</td>
      <td>0.002071</td>
      <td>0.092745</td>
      <td>0.002071</td>
      <td>0.064925</td>
    </tr>
    <tr>
      <th>Σ</th>
      <td>0.289003</td>
      <td>0.117163</td>
      <td>0.013601</td>
      <td>0.609205</td>
      <td>0.013601</td>
      <td>0.426464</td>
    </tr>
  </tbody>
</table>
</div>



I next plot explained variance ratios as a function of number of dimensions. With 5 dimensions, more than 70% with Benzécri and 51% with Greenacre correction is explained.


```python
N_eig_all = np.linspace(1,100,100)
Expl_var_nc = []
Expl_var_bn = []
Expl_var_bnga = []
for N_eig in N_eig_all:
    Expl_var_nc.append(np.sum(mca_ind.expl_var(greenacre=False, 
                                        N=N_eig)))
    Expl_var_bn.append(np.sum(mca_ben.expl_var(greenacre=False,
                                               N=N_eig)))
    Expl_var_bnga.append(np.sum(mca_ben.expl_var(greenacre=True, 
                                      N=N_eig)))
```


```python
plt.figure(figsize=(8,5))
plt.plot(N_eig_all,Expl_var_nc, label='No-correction')
plt.plot(N_eig_all,Expl_var_bn, label='Benzecri correction')
plt.plot(N_eig_all,Expl_var_bnga,label='Benzecri & Greenacre correction')
plt.legend(loc='lower right')
plt.ylim(0,1.1)
plt.xlim(5,100)
```



<div class='fig figcenter fighighlight'>
  <img src='/images/si_mca1.png'>
</div>



#### Plotting reduced dimensional space

I next plot the first 3 dimensions against one another. These dimensions explain more than 60% with Benzécri and 42% variance with Greenacre correction. It appears that the reduced dimension space may be a good choice of metric to predict which students are more likely to fail. 


```python
expl_var = mca_ben.expl_var(greenacre=True, N=N_eig)
target_01 = [1.0 if (tar_i=='yes') else 0.0 for tar_i in target]
target_01 = np.asarray(target_01)
color_target = [[0,0,1,.5] if (tar_i=='yes') else [1,0,0,.5] for tar_i in target]

plt.figure(figsize(10,10))
plt.subplot(2,2,1)
plot_low_Dims(plt,features_LD,target_01,0,0,1)

plt.subplot(2,2,2)
plot_low_Dims(plt,features_LD,target_01,0,1,2)

plt.subplot(2,2,3)
plot_low_Dims(plt,features_LD,target_01,0,0,2)
```



<div class='fig figcenter fighighlight'>
  <img src='/images/si_mca2.png'>
</div>




```python
x = np.asarray(features_LD[target_01==0,10])
y = np.asarray(features_LD[target_01==0,11])
z = np.asarray(features_LD[target_01==0,12])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z,  s=30,color = 'r',alpha = 0.5)
x = np.asarray(features_LD[target_01==1,10])
y = np.asarray(features_LD[target_01==1,11])
z = np.asarray(features_LD[target_01==1,12])
ax.scatter(x, y, z,  s=30,color = 'b',alpha = 0.5)
```



<div class='fig figcenter fighighlight'>
  <img src='/images/si_mca3.png'>
</div>




### Conclusion

In this post, I applied Multiple Correspondance Analysis (MCA) to demographic and socioeconomic data collected in 395 students. MCA is a dimensionality reduction techique for data with all categorical variables. After applying MCA, the number of features reduced to less than 15. Preliminary EDA suggests that 5 dimensions may be sufficient to predict which students are at risk of failing the prgoram. 



### Additional links
- Multiple Correpondance Analysis https://www.utdallas.edu/~herve/Abdi-MCA2007-pretty.pdf
- https://github.com/esafak/mca
