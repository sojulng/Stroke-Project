# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

# Packages for decision tree model
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Packages for neural network model
from sklearn.neural_network import MLPClassifier

# Packages for accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report



# Stroke Project: If marriage increases the likelihood of stoke
df = pd.read_csv("C:/Users/solju/Downloads/healthcare-dataset-stroke-data.csv")
df.head()

# Checking the 12 variables and missing variables
df.count()

# Data Cleaning: Exclude rows with missing variables
df_new = df.dropna()
df_new.count()

pd.set_option('display.max_columns', None)
print(df_new)

# Data Visualization: Marriage and Stroke
# Fig1: Marriage Status
df_new['ever_married'].value_counts()
result = [3204,1705]
lab = ['Married', 'Not Married']
color1 = ['#84C0C6','#BFADA3']
color2 = ['#F2E2D2','#9FB7B9']

fig1, ax1=plt.subplots()
ax1.set_title('Marriage Status')
ax1.pie(result, labels=lab, autopct='%1.1f%%',pctdistance=0.85, colors=color1)
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Fig2: Stroke Diagnosis
df_new['stroke'].value_counts()
S_results = [4700,209]
S_results_lab = ['No Stroke','Stroke']

fig2, ax2=plt.subplots()
ax2.set_title('Stroke Diagnosis')
ax2.pie(S_results, labels=S_results_lab, autopct='%1.1f%%',startangle=-25, pctdistance=0.85, colors=color2)
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Data for married with stroke
married_with_stroke = df_new[(df_new['ever_married']=='Yes') & (df_new['stroke']==1)]
count_married_with_stroke = married_with_stroke.shape[0]
print("Count of married individuals with stroke:", count_married_with_stroke)
# Data for married without stroke
print('Count of married individuals without stroke:', 3204-(count_married_with_stroke))
# Data for not married with stroke
not_married_with_stroke = df_new[(df_new['ever_married']=='No') & (df_new['stroke']==1)]
count_not_married_with_stroke = not_married_with_stroke.shape[0]
print("Count of not married individuals with stroke:", count_not_married_with_stroke)
# Data for not married without stroke
print("Count of not married individuals without stroke:",1705-(count_not_married_with_stroke))
S2_results_lab = ['No Stroke','Stroke','No Stroke','Stroke']
S2_results = [3018,186,1682,23]


# Combining the two pie charts to one visualization
fig3, ax3=plt.subplots()
ax3.set_title('Marriage Status v. Stroke Diagnosis')
ax3.pie(result, labels=lab, colors=color1,frame=True)
ax3.pie(S2_results,autopct='%1.1f%%', colors=color2, radius=0.75)
centre_circle = plt.Circle((0,0), 0.5,color='black', fc='white', linewidth=0)

fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')
plt.tight_layout()


# Kernel density estimate function
df_new['stroke'].value_counts()
sns.kdeplot(data=df_new, x='age')
#Testing the correlation of avg. glucose lvl to stroke 
sns.kdeplot(data=df_new, x='age', hue='stroke')

# Kernel density estimate function
df_new['stroke'].value_counts()
sns.kdeplot(data=df_new, x='avg_glucose_level')
#Testing the correlation of avg. glucose lvl to stroke 
sns.kdeplot(data=df_new, x='avg_glucose_level', hue='stroke')

# Retreiving the Target Variable
target = df_new.iloc[:,11]
target.head()

# Dummying variables in the feature set
feature_set = df_new[['gender','age','hypertension','heart_disease','ever_married','avg_glucose_level','bmi','smoking_status','work_type','Residence_type']]
feature_set.head()
df_01 = pd.get_dummies(data=feature_set, columns=['gender','ever_married','smoking_status','work_type','Residence_type',], drop_first=True)
df_01.head()

    # CONSTRUCTING THE MODEL

# Creating X and Y for test and training sets
x = df_01.iloc[:,0:16]
x.head()
y = target
y.head()
# Decision Tree
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.2,random_state=46)
model = DecisionTreeClassifier()
model.fit(train_x, train_y)
fig4 = plt.figure(figsize=(25,20)), tree.plot_tree(model, feature_names=list(train_x), filled=True, max_depth=3)

# Accuracy, classificaiton, confusion matrix
pred_y=model.predict(test_x)
accuracy = accuracy_score(test_y, pred_y)
accuracy
rp = classification_report(test_y, pred_y)
print(rp)

cm = confusion_matrix(test_y, pred_y, labels=model.classes_)
cm
cmd = ConfusionMatrixDisplay(cm,display_labels=model.classes_)
cmd.plot()


# 907(true neg) predicted that one did not have strokes and they did not.
# 23(false neg) predicted that one did not have stroke when they did.
# 44(false pos) predicted one had strokes when they did not.
# 8(true pos) predicted one had strokes and they did.



# Neural Networking
clf = MLPClassifier(hidden_layer_sizes=(4,3), random_state=1, solver='lbfgs', alpha= 1e-5)
clf.fit(train_x, train_y)

predicted_test_y = clf.predict(test_x)
acy = accuracy_score(test_y, predicted_test_y)
acy
rp2 = classification_report(test_y, predicted_test_y)
print(rp2)

