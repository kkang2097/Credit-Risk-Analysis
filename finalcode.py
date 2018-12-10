# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 22:04:26 2018

@author: Elliot
"""

import os
os.chdir('C:\\Users\\Elliot\\Desktop\\hw\\sta141b\\proj\\loans')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#To check runtime of the algorithm, multiprocessing could
#speed this up
from time import time
t1 = time()

#Reading in the data
'''
Since 2007 - 2011 was a relatively shaky economic time, we should
take out the relevant csv file.
'''
d0 = pd.read_csv('LoanStats_2007to2011.csv', header = 1)
d1 = pd.read_csv('LoanStats_2012to2013.csv', header = 1)
d2 = pd.read_csv('LoanStats_2014.csv', header = 1)
d3 = pd.read_csv('LoanStats_2015.csv', header = 1)
d4 = pd.read_csv('LoanStats_2016Q1.csv', header = 1)
d5 = pd.read_csv('LoanStats_2016Q2.csv', header = 1)
d6 = pd.read_csv('LoanStats_2016Q3.csv', header = 1)
d7 = pd.read_csv('LoanStats_2016Q4.csv', header = 1)
d8 = pd.read_csv('LoanStats_2017Q1.csv', header = 1)
d9 = pd.read_csv('LoanStats_2017Q2.csv', header = 1)
d10 = pd.read_csv('LoanStats_2017Q3.csv', header = 1)
d11 = pd.read_csv('LoanStats_2017Q4.csv', header = 1)
#d12 = pd.read_csv('LoanStats_2018Q1.csv', header = 1)
#d13 = pd.read_csv('LoanStats_2018Q2.csv', header = 1)
#d14 = pd.read_csv('LoanStats_2018Q3.csv', header = 1)

df = pd.concat([d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11])

def missingness(col, data):
    '''
    
    A simple function to print out the number of missing values by column.
    
    Argument variables:
    
     col - the given dataset column
     data - the given dataframe
     
    '''
    return(data[col].isnull().sum(), col)

#for column in df.columns.values:
#    print(missingness(column, df))

#Number of columns
len(df.columns.values)

#Dropping columns with more than 800000 missing values
t1 = time()
for column in df.columns.values:
    if missingness(column, df)[0] > 800000:
        df.drop(column, axis = 1, inplace = True)
        
print(time() - t1)

#Data Transformation of int_rate column
df['int_rate'] = df['int_rate'].str.rstrip('%').astype('float')/100.0
df['int_rate'].corr(df['inq_last_6mths'])

#Getting rid of A and B rated loans because the interest
#rates are not very favorable
df = df[df['grade'] != 'A']
df = df[df['grade'] != 'B']
df = df.drop(columns = ['emp_title'])
df = df[df['annual_inc'] < 400000]
max(df['annual_inc'])
#Replacing loan_status with 1s for good behavior,
#0s for bad behavior
df['loan_status'] = df['loan_status'].replace('Charged Off', 0.0)
df['loan_status'] =df['loan_status'].replace('Current', 1.0)
df['loan_status'] =df['loan_status'].replace('Default', 0.0)
df['loan_status'] =df['loan_status'].replace('Does not meet the credit policy. Status:Charged Off', 0.0)
df['loan_status'] =df['loan_status'].replace('Does not meet the credit policy. Status:Fully Paid', 1.0)
df['loan_status'] =df['loan_status'].replace('In Grace Period', 1.0)
df['loan_status'] =df['loan_status'].replace('Late (16-30 days)', 1.0)
df['loan_status'] =df['loan_status'].replace('Late (31-120 days)', 0.0)
df['loan_status'] = df['loan_status'].replace('Fully Paid', 1.0)

#Transforming loan term
#3 for 3 year term, 5 for 5 year term
df['term'] = df['term'].replace(' 36 months', 3.0)
df['term'] = df['term'].replace(' 60 months', 5.0)

#Getting rid of data without loan term, since it is essential
df.dropna(subset = ['term', 'last_pymnt_d', 'issue_d'], inplace = True)

#Turning home ownership into an ordinal variable
'''

Invalid values = 0
Rent = 1
Mortgage = 2
Own = 3

'''

df['home_ownership'] = df['home_ownership'].replace(['ANY', 'NONE', 'OTHER'], 0)
df['home_ownership'] = df['home_ownership'].replace(['RENT'], 1.0)
df['home_ownership'] = df['home_ownership'].replace(['MORTGAGE'], 2.0)
df['home_ownership'] = df['home_ownership'].replace(['OWN'], 3.0)


#Transforming employment length
#0,1,2,3,4,5,6,7,8,9,10
#df.groupby('emp_length').size()

df['emp_length'] = df['emp_length'].replace(['< 1 year'], 0)
df['emp_length'] = df['emp_length'].replace('1 year', 1)
df['emp_length'] = df['emp_length'].replace('2 years', 2)
df['emp_length'] = df['emp_length'].replace('3 years', 3)
df['emp_length'] = df['emp_length'].replace('4 years', 4)
df['emp_length'] = df['emp_length'].replace('5 years', 5)
df['emp_length'] = df['emp_length'].replace('6 years', 6)
df['emp_length'] = df['emp_length'].replace('7 years', 7)
df['emp_length'] = df['emp_length'].replace('8 years', 8)
df['emp_length'] = df['emp_length'].replace('9 years', 9)
df['emp_length'] = df['emp_length'].replace('10+ years', 10)

#Finding the lifetime of the loan
import datetime
from datetime import timedelta

months = ['Jan-', 'Feb-', 'Mar-', 'Apr-', 'May-', 'Jun-',
          'Jul-', 'Aug-', 'Sep-', 'Oct-', 'Nov-', 'Dec-']

months_int = ['1-', '2-', '3-', '4-', '5-', '6-',
              '7-', '8-', '9-', '10-', '11-', '12-']

#df['issue_d'] = df['issue_d'].str.replace(months[1], months_int[1])
#df['issue_d'] = df['issue_d'].str.replace(months[2], months_int[2])
#df['issue_d'] = df['issue_d'].str.replace(months[3], months_int[3])
#df['issue_d'] = df['issue_d'].str.replace(months[4], months_int[4])
#df['issue_d'] = df['issue_d'].str.replace(months[5], months_int[5])
#df['issue_d'] = df['issue_d'].str.replace(months[6], months_int[6])
#df['issue_d'] = df['issue_d'].str.replace(months[7], months_int[7])
#df['issue_d'] = df['issue_d'].str.replace(months[8], months_int[8])
#df['issue_d'] = df['issue_d'].str.replace(months[9], months_int[9])
#df['issue_d'] = df['issue_d'].str.replace(months[10], months_int[10])
#df['issue_d'] = df['issue_d'].str.replace(months[11], months_int[11])

#Turning columns into usable variables
for i in range(len(months)):
    df['issue_d'] = df['issue_d'].str.replace(months[i], months_int[i])
    df['last_pymnt_d'] = df['last_pymnt_d'].str.replace(months[i],months_int[i] )

def conv_date(var):
    datetimeFormat = '%m-%Y'
    return(datetime.datetime.strptime(var, datetimeFormat))

#Converting time columns into Timedelta terms
#Units: days
df['last_pymnt_d'].isnull().sum()
df['issue_d'] = df['issue_d'].map(conv_date)
df['last_pymnt_d'] = df['last_pymnt_d'].map(conv_date)
df['loan_life'] = df['last_pymnt_d'] - df['issue_d']



#Getting our loan nondefaults
nondef = df[df['loan_status'] != 0.0]

#Grouping these charged off loans by chargeoff date
df['percent_left'] = df['out_prncp']/df['funded_amnt']


#Getting our loan defaults
#list(set(df['loan_status']))
#df['loan_status'].corr(df['int_rate'])
default = df[df['loan_status'] == 0.0]

#Catching defaults that have not paid out completely
from scipy import stats

#density = stats.kde.gaussian_kde(default['percent_left'])
#x = np.arange(0., 1, 0.02)
#len(default.loc[default['percent_left'] > 0.1])
#plt.plot(x, density(x))
#plt.xlabel('Percent of Principal Unpaid (in decimals) for Grade C')
#plt.ylabel('Density')
#plt.title('Density Plot of Outstanding Principal Unpaid')
#plt.show()

#grouping loans by grade
c_grade = df[df['grade'] == 'C']
d_grade = df[df['grade'] == 'D']
e_grade = df[df['grade'] == 'E']
f_grade = df[df['grade'] == 'F']
g_grade = df[df['grade'] == 'G']

#grouping by grade, and defaults to find density plot of
#"Percentage of Loan left to pay
c_grade_def = c_grade[c_grade['loan_status'] == 0.0]
d_grade_def = d_grade[d_grade['loan_status'] == 0.0]
e_grade_def = e_grade[e_grade['loan_status'] == 0.0]
f_grade_def = f_grade[f_grade['loan_status'] == 0.0]
g_grade_def = g_grade[g_grade['loan_status'] == 0.0]

#Percent of loan left to pay by grade
def density_plot(data, category):
    density = stats.kde.gaussian_kde(data)
    x = np.arange(0.1, 0.4, 0.01)
    plt.plot(x, density(x))
    plt.xlabel('Percent ' + category)
    plt.ylabel('Density (from 0 to 100)')
    plt.title('Density Plot for ' + category)
    plt.show()

density_plot(df['int_rate'], 'Interest Rate')

fig, ax = plt.subplots(1,1)

density1 = stats.kde.gaussian_kde(c_grade_def['percent_left'])
density2 = stats.kde.gaussian_kde(d_grade_def['percent_left'])
density3 = stats.kde.gaussian_kde(e_grade_def['percent_left'])
density4 = stats.kde.gaussian_kde(f_grade_def['percent_left'])
density5 = stats.kde.gaussian_kde(g_grade_def['percent_left'])

x = np.arange(0., 1.0, 0.04)
plt.plot(x, density1(x), label = 'C')
plt.plot(x, density2(x), label = 'D')
plt.plot(x, density3(x), label = 'E')
plt.plot(x, density4(x), label = 'F')
plt.plot(x, density5(x), label = 'G')

plt.xlabel('Percent of Principal Unpaid (in decimals)')
plt.ylabel('Density (from 0 to 100)')
plt.title('Density Plot of Outstanding Principal Unpaid by Grades')
plt.legend()
fig.show()

##
##Now to start the data analysis
##
##

#Checking if verified loans have similar rate of default as
#nonverified loans
n_verif = df[df['verification_status'] == 'Not Verified']
len(n_verif[n_verif['loan_status'] == 0.0])
np.mean(df['int_rate'])
np.mean(n_verif['int_rate'])

#An experiment in heuristics (rules) to get
#high returns for loans.
#
#Result: Didn't work, got 10% but it's not
#so good.

default9 = default[default['int_rate'] > 0.15]
default9 = default9[default9['int_rate'] < 0.17]
nondef9 = nondef[nondef['int_rate'] > 0.15]
nondef9 = nondef9[nondef9['int_rate'] < 0.17]
default9.groupby('purpose').size() / (default9.groupby('purpose').size() + nondef9.groupby('purpose').size())
nondef9.groupby('purpose').size()
vacay = nondef9[nondef9['purpose'] == 'vacation']
np.sum(vacay['funded_amnt'])
wed = nondef9[nondef9['purpose'] == 'wedding']
len(defaults9) /(len(defaults9) + len(nondef9))
len(nondef9)
32/530.0
nondef9['funded_amnt'].sum()
mat = df.corr()
np.mean(nondef9['int_rate'])

#Finding our benchmark to beat
#We find this sum equals 0.35, which is a 35% gain over 10 years.
#Discount the first 5 years, since this data is much more sparse.
#So 1.35^0.2 indicates a 6.1% return
sum(df['total_rec_int'])/sum(df['total_rec_prncp'])

#Correlation of comparing mean interest rate to
#percent_left of charged-off loans
default['percent_left'].corr(default['int_rate'])

#####Random Forest
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics



####Logistic Regression
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

newdf = pd.concat([df['annual_inc'],df['loan_status'],df['int_rate'], df['home_ownership'], df['emp_length'],
                   df['term'], df['dti'], df['purpose']], axis = 1)
newdf.isnull().sum()
newdf.dropna(subset = ['emp_length', 'dti'], inplace = True)    
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(np.array(newdf['purpose']))
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

#reshaping dataframes for ML
newdf.drop(['purpose'], axis = 1, inplace = True)
y = newdf['loan_status']
newdf.drop(['loan_status'], axis = 1, inplace = True)
x = np.hstack((newdf.loc[:, newdf.columns.values].values, onehot_encoded))
y = np.array(y).reshape(len(y), 1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

#Analysis of outcome
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
conf = confusion_matrix(y_test, y_pred)
conf
print(classification_report(y_test, y_pred))

####Random Forests
from sklearn.ensemble import RandomForestClassifier

#y_pred = float64, (num,)
#y_test = float64, (num,1)
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
y_test = y_test.reshape((len(y_test),))

import itertools
list(itertools.chain(y_test))
y_pred = y_pred.reshape(len(y_pred), 1)
list_ypred = list(itertools.chain.from_iterable(y_pred.tolist()))
list_ytest = list(itertools.chain.from_iterable(y_test.tolist()))
conf = confusion_matrix(list_ytest, list_ypred)
conf
print(classification_report(list_ytest, list_ypred))

#Creating an additional rule from the Random Forest
#Now reducing the voting threshold for the classifier
prob = clf.predict_proba(X_test)
prob_good = prob[:,1]
prob_bad = pd.DataFrame(prob[:,0])
np.mean(prob_bad)
len(prob_bad[prob_bad[0]>0.5])
prob_bad.iloc[0,0]

new_ypred = []
for i in range(len(prob_bad)):
    if (prob_bad.iloc[i,0] > 0.2):
        new_ypred.append(0.0)
    else:
        new_ypred.append(1.0)
        
conf = confusion_matrix(list_ytest, new_ypred)
conf
print(classification_report(list_ytest, new_ypred))




