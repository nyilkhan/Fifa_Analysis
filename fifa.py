import numpy as np
import pandas as pd
import math
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#from sklearn import preprocessing, cross_validation, svm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import svm
import matplotlib.pyplot as plt # plotting
import missingno as msno


#Cleaning some of values so that we can interpret them
def value_to_int(df_value):
    try:
        value = float(df_value[1:-1])
        suffix = df_value[-1:]
        if suffix == 'M':
            value = value * 1000000
        elif suffix == 'K':
            value = value * 1000
    except ValueError:
        value = 0.00
    return value

def height_to_inch(df_value):
    try:
        feet = float(df_value[0])
        inch = float(df_value[2:])
        height = (feet)*12 + (inch)
    except TypeError:
        height = -1.00
    return height

def weight_to_lbs(df_value):
    try:
        lbs = float(df_value[0:-3])
    except TypeError:
        lbs = -1.00
    return lbs



df1 = pd.read_csv('data.csv')
numRows, numCol = df1.shape

df1.drop(['Unnamed: 0','Photo','Flag','Club Logo'],axis=1,inplace=True)


df1 = df1[['Name','Age' ,'Overall','Potential','Value','Wage','Skill Moves',
        'Height','Weight','Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing',
        'Volleys',
       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']]
#df1.drop(['Loaned From'],axis=1,inplace=True)
df1['Value'] = df1['Value'].apply(value_to_int)
df1['Wage'] = df1['Wage'].apply(value_to_int)
df1['Height'] = df1['Height'].apply(height_to_inch)
df1['Weight'] = df1['Weight'].apply(weight_to_lbs)
df1 = df1.reset_index()

#print(df1.head(5))


#data analysis
#print('Total number of countries : {0}'.format(df1['Nationality'].nunique()))
#print(df1['Nationality'].value_counts().head(7))
print('Maximum Potential : '+str(df1.loc[df1['Potential'].idxmax()][1]))
print('Maximum Overall Perforamnce : '+str(df1.loc[df1['Overall'].idxmax()][1]))
skill_cols = ['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']
'''
#prints the best in each skill
i = 0
while i < len(skill_cols):
    print('Best {0} : {1}'.format(skill_cols[i], df1.loc[df1[skill_cols[i]].idxmax()][1]))
    i+=1
'''
sns.jointplot(x=df1['Dribbling'], y=df1['Crossing'], kind="hex", color="#4CB391")
#plt.show()

#print(df1['Position'].value_counts())

forecast_col = 'Overall'
df1.fillna(-999.99,inplace = True)
forecast_out = int(math.ceil(.0005*len(df1)))
#df1['label'] = df1[forecast_col].shift(-forecast_out)

df1['label'] = df1[forecast_col]
#df1 = df1.drop('Value',axis=1)


X = np.array(df1.drop(['Name','Overall','label'],1))
y = np.array(df1['label'])

#scale the X values
#X = preprocessing.scale(X)
#y = np.array(df1['label'])
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
#print(X_test.shape,X_train.shape)
#print(y_test.shape,y_train.shape)


clf = LinearRegression()
clf.fit(X_train,y_train,sample_weight=None)
predictions = clf.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error
print('r2 score: '+str(r2_score(y_test, predictions)))
print('RMSE : '+str(np.sqrt(mean_squared_error(y_test, predictions))))


accuracy = clf.score(X_test,y_test)
print(accuracy)
