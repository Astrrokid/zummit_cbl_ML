#!/usr/bin/env python
# coding: utf-8

# In[62]:


#Importing the necessary libraries
import numpy as np 
import pandas as pd
import tensorflow as tf
import joblib
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[34]:


#Importing and converting the csv file to a Pandas Dataframe for easy manipulation
df = pd.read_csv('loan_approval_dataset.csv')


# In[35]:


#Checking the information of the dataset
df.info()


# In[36]:


#Checking to see if there's any missing datapoint
df.isnull().sum()


# In[37]:


#Displaying the first five datapoints
df.head()


# In[38]:


#There are whitespaces before the letters in the column names
df.columns


# In[39]:


#Removing the whitespaces from the column names and displaying them to make sure they've change accordingly 
for i in df.columns:
    df.rename(columns={i : i.strip()}, inplace=True)
    
df.columns


# In[40]:


#Checking the summarized description of the numerical columns
df.describe()


# ## Exploratory Data Analysis

# In[41]:


sns.countplot(x=df['loan_status'],palette= 'coolwarm')


# In[42]:


#Visualizing the dataset with seaborn pairplot
sns.pairplot(df, hue='loan_status', palette= 'coolwarm')


# In[43]:


'''Creating another column namee 'repayable', this is the sum of the value of the clients assets
    and the the loan they are requesting is deducted from
    it to know if the client assetss can pay for the loan if the they default'''

df['repayable']= (df['residential_assets_value']+df['commercial_assets_value']+df['luxury_assets_value']+df['bank_asset_value'])-df['loan_amount']
df['repayable']


# In[44]:


# Displaying the 'repayable' description
df['repayable'].describe()


# In[45]:


# Distribution plot of 'repayable' with loan status
sns.displot(df, x=df['repayable'],hue=df['loan_status'], kde=True)


# In[46]:


# Count plots for for the categorical columns
# Function to create a rounded rectangle patch

fig = plt.figure(figsize=(15, 15))
# Iterate through each subplot
for e, i in enumerate(['no_of_dependents', 'education', 'self_employed', 'loan_term']):
    ax = plt.subplot(2, 2, e + 1)
    sns.countplot(x=df[i], hue=df['loan_status'], palette='coolwarm', ax=ax)

    plt.xlabel(i)
    plt.ylabel('Count')

# Display the plot
plt.tight_layout()


# In[47]:


# Loan term count
loan_term_count=df[['loan_term','loan_status']]
loan_term_count=pd.DataFrame(loan_term_count.value_counts()).sort_values(by='loan_term')
loan_term_count


# In[48]:


# Plot pie charts for loan terms
plt.figure(figsize=(15,15), constrained_layout=True)
colors=sns.color_palette('pastel')
for i,e in zip(range(1,11), range(2,21,2)):
    plt.subplot(4,3,i)
    plt.pie(loan_term_count.loc[e]['count'].values, explode=(0.2,0), 
            labels=['Accepted', 'Rejected'], autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title(f'{e} years Loan term')
    plt.tight_layout()


# In[49]:


# Strip plots for various continous columns against loan status
fig = plt.figure(figsize=(15,15))
for e, i in enumerate(['income_annum', 'loan_amount', 'cibil_score',
                 'residential_assets_value', 'commercial_assets_value',
                 'luxury_assets_value', 'bank_asset_value', 'repayable']):
    plt.subplot(3,3,e+1)
    sns.stripplot(data=df, x= 'loan_status' , y= i, hue='loan_status', palette='coolwarm')
    plt.xlabel('Loan Status')
    plt.ylabel(i)


# In[50]:


# Filtering the DataFrame for rows with cibil_score <= 550
# Selecting only the 'cibil_score' and 'loan_status' columns
# Counting the occurrences of each loan status in the filtered DataFrame
low_cibil= df[df['cibil_score']<=550]
low_cibil= low_cibil[['cibil_score','loan_status']]
low_cibil['loan_status'].value_counts()


# In[51]:


# Filtering the DataFrame for rows with cibil_score >= 550
# Selecting only the 'cibil_score' and 'loan_status' columns
# Counting the occurrences of each loan status in the filtered DataFrame
high_cibil= df[df['cibil_score']>=550]
high_cibil= high_cibil[['cibil_score','loan_status']]
high_cibil['loan_status'].value_counts()


# In[52]:


# Pie chart for clients with a credit score below 550

plt.figure(figsize=(10,5))
colors=sns.color_palette('pastel')

# First pie chart: Clients with a Credit Score below 550
plt.subplot(1,2,1)
plt.pie([189,1600], explode=(0.2,0), labels=['Accepted','Rejected'], autopct='%1.1f%%',shadow=True, colors=colors)
plt.title('Clients with a Credit Score below 550')

# Second pie chart: Clients with a Credit Score above 550
plt.subplot(1,2,2)
plt.pie([2471,13], explode=(0.2,0), labels=[ 'Accepted','Rejected'], autopct='%1.1f%%',shadow=True, colors=colors)
plt.title('Clients with a Credit Score Above 550')

plt.tight_layout()


'''From the pie chart below, we can deduce that clients with a Credit Score above 550 are usually accepted for the loan
    While clients with a credidt score below 550 are usually denied a loan'''


# In[53]:


# Creating violin plots for each of the continous column against loan status
plt.figure(figsize=(15,15))
for e, i in enumerate(['income_annum', 'loan_amount', 'cibil_score',
                 'residential_assets_value', 'commercial_assets_value',
                 'luxury_assets_value', 'bank_asset_value', 'repayable']):
    plt.subplot(3,3,e+1)
    sns.violinplot(data=df, x= 'loan_status' , y= i, hue='loan_status', palette='coolwarm')
    plt.xlabel('Loan Status')
    plt.ylabel(i)


# In[ ]:





# In[ ]:





# ### Feature Engineering

# In[54]:


# Mapping the categorical values of the 'education' and 'self_employed' column to numerical values
x={' Approved': 1, ' Rejected': 0}
y={' Graduate': 1, ' Not Graduate': 0}
z={' No': 0, ' Yes': 1}
df['loan_status']= df['loan_status'].map(x)
df['education']= df['education'].map(y)
df['self_employed']= df['self_employed'].map(z)
df


# In[55]:


#Calculates the debt-to-income ratio by dividing the loan amount by the annual income for each row in the DataFrame.
#And Dropping the 'loan_id' column because it's unnecessary
df['debt_income_ratio'] = df['loan_amount']/df['income_annum']
df.drop('loan_id', axis=1, inplace=True)
df


# In[56]:


'''Plot the heatmap of correlation matrix
    Calculating the correlation matrix using df.corr()
    and annotating each cell with the correlation value'''

plt.figure(figsize= (15,10))
sns.heatmap(df.corr(), cmap='coolwarm', annot=True)


# In[57]:


# Calculate the correlation of each feature with 'loan_status' and sort them in descending order
# Drop 'loan_status' from the resulting Series
df_loan_corr=pd.DataFrame( df.corr()['loan_status'].sort_values(ascending= False).drop('loan_status'))
df_loan_corr


# In[58]:


#Creating a horizontal bar plot to visualize the correlation between all the features to the loan_status feature(Target)
plt.figure(figsize= (10,5))
sns.barplot(df_loan_corr, x= df_loan_corr['loan_status'], y=df_loan_corr.index, palette='coolwarm')
plt.title('Correlation between Loan Status and all other features')


# In[ ]:





# ## Data Preprocessing
# 
# 

# In[59]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE

X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



# In[60]:


#Balacing the imbalanced training dataset.
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)


# In[61]:


#The training dataset has balanced from the plot below.
sns.countplot(x=y_train_sm,palette= 'coolwarm')


# In[63]:


#Normalization of the feature columns
scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Saving the fitted Normalization model
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)


# In[64]:


#Importing the models 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout


# In[65]:


# Creating all the models objects
# To be used by Grid Search to find the most 

# Linear Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)

# Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
decision_tree_pred = decision_tree.predict(X_test)

# K Nearest Neighbor
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

# Support Vector Machine
svc = SVC()
svc.fit(X_train, y_train)
svc_pred=svc.predict(X_test)

# Random Forest Classifier
rfc= RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)


# In[66]:


predictions =  [log_reg_pred, decision_tree_pred, rfc_pred,knn_pred, svc_pred]


# In[67]:


'''
Importing metrics from sklearn to evaluate the predictions
- Created a for loop to iterate throught the predictions of each model and 
then calculating its accuracy and classification report
'''
from sklearn.metrics import classification_report, confusion_matrix
Accuracy = {'Logistic Regression': 0, 'Decision Tree Classifier': 0,
            'Random Forest Classifier': 0,
          'K Nearest Neighbor': 0, 'Support Vector Machine': 0}
models = ['Logistic Regression', 'Decision Tree Classifier','Random Forest Classifier',
          'K Nearest Neighbor', 'Support Vector Machine']
for acc, model, prediction in zip(list(Accuracy.keys()), models, predictions):
    Accuracy[acc] = np.mean(y_test == prediction)
    print(f'{model} - Accuracy: {Accuracy[acc]} \n')
    print(classification_report(y_test, prediction),confusion_matrix(y_test, prediction), sep='\n')
    print('\n'*3)


# In[70]:


#Displaying the confusion matrix of all the models as a heatmap
fig = plt.figure(figsize=(10,10))
for prediction, e, model in zip(predictions,[1,2,3,4,5], models):
    plt.subplot(3,2,e)
    sns.heatmap(confusion_matrix(y_test, prediction), annot= True, cmap='coolwarm',
                fmt='.0f', square=True, linewidths=2, linecolor="#CCD1ED")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(model)
plt.tight_layout()
    


# In[71]:


'''
Creating a pandas DataFrame with two columns which the Models will have its respective accuracy on the dataset
Using the Dataframe to plot a bar chart of the accuracies agains their respective models
'''
acc = pd.DataFrame({'Models': list(Accuracy.keys()), 'Accuracy': list(Accuracy.values())}).sort_values(by='Accuracy',ascending=False)
fig = plt.figure(figsize=(10,5))
sns.barplot(x= acc['Accuracy'], y=acc['Models'], palette='coolwarm')
plt.tight_layout()
acc


# ### Optimizing the Models with Better Hyperparameters

# In[72]:


# Linear Regression
log_reg_opt = LogisticRegression()

# Decision Tree Classifier
decision_tree_opt = DecisionTreeClassifier(random_state=42)


# K Nearest Neighbor
knn_opt = KNeighborsClassifier(n_neighbors = 2)

# Support Vector Machine
svc_opt = SVC()

# Random Forest Classifier
rfc_opt = RandomForestClassifier(random_state=42)


# ##### Finding the best hyperparameters for each model

# In[73]:


#Decision Tree
param_grid = {'min_samples_split': range(2,10), 'min_samples_leaf':range(1,10)}
grid_search = GridSearchCV(estimator=decision_tree_opt, param_grid=param_grid, scoring='accuracy',verbose=1, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params_dec = grid_search.best_params_
best_params_dec


# In[74]:


# Decision Tree Classifier
decision_tree_opt = DecisionTreeClassifier(min_samples_leaf=1,min_samples_split=2)
decision_tree_opt.fit(X_train, y_train)
decision_tree_opt_predictions = decision_tree_opt.predict(X_test)


# In[75]:


# KNearest Neighbor
param_grid = {'leaf_size': range(1,100,10), 'p':range(1,10)}
grid_search = GridSearchCV(estimator=knn_opt, param_grid=param_grid, scoring='accuracy',verbose=1, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params_knn = grid_search.best_params_
best_params_knn


# In[76]:


knn_opt = KNeighborsClassifier(n_neighbors = 2, leaf_size=1, p=3)
knn_opt.fit(X_train, y_train)
knn_opt_predictions = knn_opt.predict(X_test)


# In[77]:


# Logistic Regressor
param_grid = {'C': range(1,10), 'tol':[0.0001,0.001,0.01,0.1,1,10]}
grid_search = GridSearchCV(estimator=log_reg_opt, param_grid=param_grid, scoring='accuracy',verbose=1, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params_log = grid_search.best_params_
best_params_log


# In[78]:


log_reg_opt = LogisticRegression(C=2,tol=1)
log_reg_opt.fit(X_train, y_train)
log_reg_opt_predictions = log_reg_opt.predict(X_test)


# In[79]:


param_grid = {'n_estimators': range(100,1000,100)}
grid_search = GridSearchCV(estimator=rfc_opt, param_grid=param_grid, scoring='accuracy',verbose=1, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params_rfc = grid_search.best_params_
best_params_rfc


# In[80]:


rfc_opt = RandomForestClassifier(n_estimators=400)
rfc_opt.fit(X_train, y_train)
rfc_opt_predictions = rfc_opt.predict(X_test)


# In[81]:


# Support Vector Machine
param_grid = {'C': range(1,10), 'gamma': [1,0.1,0.01,0.001,0.0001]}
grid_search = GridSearchCV(estimator=svc_opt, param_grid=param_grid, scoring='accuracy',verbose=1)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params_svc = grid_search.best_params_
best_params_svc


# In[ ]:





# In[82]:


svc_opt = SVC(C=7, gamma=0.1)
svc_opt.fit(X_train, y_train)
svc_opt_predictions=svc_opt.predict(X_test)


# In[83]:


model = Sequential()
model.add(Dense(13,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(39,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(19,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')


# In[84]:


model.fit(x=X_train,y=y_train, epochs=100,validation_data=(X_test,y_test),batch_size=256)


# In[85]:


nn_pred = model.predict(X_test)
nn_pred=np.rint(nn_pred)
nn_pred = nn_pred.flatten()


# In[87]:


#Initializing all the optimised model's prediction into a list
predictions_opt = [log_reg_opt_predictions, decision_tree_opt_predictions, 
                   rfc_opt_predictions,knn_opt_predictions, svc_opt_predictions, nn_pred]


# In[88]:


'''
Importing metrics from sklearn to evaluate the predictions
- Created a for loop to iterate throught the predictions of each model and 
then calculating its accuracy and classification report
'''
from sklearn.metrics import classification_report, confusion_matrix
Accuracy_opt = {'Logistic Regression Optimized': 0, 'Decision Tree Classifier Optimized': 0,
            'Random Forest Classifier Optimized': 0,
          'K Nearest Neighbor Optimized': 0, 'Support Vector Machine Optimized': 0, 'Artificial Neural Network': 0}
models_opt = ['Logistic Regression Optimized', 'Decision Tree Classifier Optimized','Random Forest Classifier Optimized',
          'K Nearest Neighbor Optimized', 'Support Vector Machine Optimized', 'Artificial Neural Network']
for acc, model, prediction in zip(list(Accuracy_opt.keys()), models_opt, predictions_opt):
    Accuracy_opt[acc] = np.mean(y_test == prediction)
    print(f'{model} - Accuracy: {Accuracy_opt[acc]} \n')
    print(classification_report(y_test, prediction),confusion_matrix(y_test, prediction), sep='\n')
    print('\n'*3)


# In[89]:


'''
Creating a pandas DataFrame with two columns which the Models will have its respective accuracy on the dataset
Using the Dataframe to plot a bar chart of the accuracies agains their respective models
'''
acc_opt = pd.DataFrame({'Models':list(Accuracy_opt.keys()),
                    'Accuracy': list(Accuracy_opt.values())}).sort_values(by='Accuracy',ascending=False)

fig = plt.figure(figsize=(10,5))
sns.barplot(x= acc_opt['Accuracy'], y=acc_opt['Models'], palette='coolwarm')
plt.tight_layout()
acc


# In[90]:


# Create a combined DataFrame
model_acc = {
    'Models': list(Accuracy.keys()) + list(Accuracy_opt.keys()),
    'Accuracy': list(Accuracy.values()) + list(Accuracy_opt.values()),
    'Type': ['Base'] * len(Accuracy) + ['Optimized'] * len(Accuracy_opt)
}

model_df = pd.DataFrame(model_acc)

# Extract base and optimized models for the same set of models
# Assuming the models in Accuracy and Accuracy_opt are correlated as base and optimized versions
model_df_combined = pd.DataFrame({
    'Models': list(Accuracy.keys()),
    'Base': list(Accuracy.values()),
    'Optimized': list(Accuracy_opt.values())[:5]
})

# Melt the DataFrame to a long format for seaborn
model_df_melted = model_df_combined.melt(id_vars='Models', value_vars=['Base', 'Optimized'], 
                             var_name='Type', value_name='Accuracy')

# Plotting
fig = plt.figure(figsize=(12, 6))
sns.barplot(x='Models', y='Accuracy', hue='Type', data=model_df_melted, palette='coolwarm')
plt.tight_layout()


# Display the DataFrame
model_df_melted


# - From the Plot above we observe that the optimized models had a slight increase in their accuracy, but not a significant one

# In[91]:


# Visualization via confusion matrix Heatmap between the model base accuracy and te accuracy of the optimized model
all_predictions =  [log_reg_pred,log_reg_opt_predictions, decision_tree_pred,decision_tree_opt_predictions,
                rfc_pred,rfc_opt_predictions,knn_pred,knn_opt_predictions,svc_pred,svc_opt_predictions,nn_pred]
all_model_names = list(Accuracy.keys())+ list(Accuracy_opt.keys())
order_indices = [0,2,4,6,8,1,3,5,7,9,10]

all_model_names = [x for _, x in sorted(zip(order_indices, all_model_names))]
#Displaying the confusion matrix of all the models as a heatmap
fig = plt.figure(figsize=(20,20))
for prediction, e, model in zip(all_predictions,range(1,12), all_model_names):
    plt.subplot(6,2,e)
    sns.heatmap(confusion_matrix(y_test, prediction), annot= True, cmap='coolwarm',
                fmt='.0f', square=True, linewidths=2, linecolor="#CCD1ED")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(model)
plt.tight_layout()
    


# # From the models trained with the dataset in the project, the model with the best performance is a tie between 
# - # Decision Tree Classifier 
# - # Random Forest Classifier 
# 
# ### This because these models are best used in dataset where human decision needed to classify a dataset  

# In[92]:


get_ipython().system('jupyter nbconvert --to script your_notebook.ipynb')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




