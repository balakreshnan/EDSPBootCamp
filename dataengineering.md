# Data Engineering and Modelling

## Process of data engineering

## Code

- Import libraries

```
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
#from tensorflow import keras

#from aif360.datasets import BinaryLabelDataset
#from aif360.metrics import BinaryLabelDatasetMetric
#from aif360.metrics import ClassificationMetric
#from aif360.algorithms.preprocessing import Reweighing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import sklearn as sk
```

- Set this to see all columns

```
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```

- now load the data into data frame

```
df = pd.read_csv('Sample_IssueDataset.csv')
df.head()
```

- Now convert categorical data into numberic

```
df1 = pd.get_dummies(df)
```

- Split features and labels

```
y = df1.iloc[:,1]
X = df1.iloc[:,:18]
```

```
X = X.drop(columns=['EmployeeLeft'])
```

```

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(x_scaled)
```

- Counts for labels
- Show the distribution

```
y.value_counts()
```

- Now time to split data set for training and testing

```
from sklearn.model_selection import train_test_split
import sklearn as sk
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

```
LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X, y)
LR.predict(X.iloc[460:,:])
round(LR.score(X,y), 4)
```

```
y_pred = LR.predict(X_test)
```

```
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred))

print("roc_auc_score: ", roc_auc_score(y_test, y_pred))
print("f1 score: ", f1_score(y_test, y_pred))
```

- now more modelling and plotting

```
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
```

- below are models to show which features have impact on the model outcome

```
model = LogisticRegression()
# fit the model
model.fit(X, y)
# get importance
importance = model.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

```
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
# define dataset
# define the model
model = DecisionTreeClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

```
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
# define dataset
# define the model
model = RandomForestClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

```
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
# define dataset
# define the model
model = RandomForestClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    if(v >= 0.04):
        	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

```
RF = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=0)
model = RF.fit(X, y)

y_pred = RF.predict(X_test)
round(RF.score(X,y), 4)

importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    if(v >= 0.04):
        	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

```
agg_employeetarget = df[['EmployeeLeft', 'Activity on Company Forums','Hired through SMTP','National Origin (code)', "Survey, Relative, Peer's Average Review of Employee"]].groupby('EmployeeLeft').mean()
```

```
import seaborn as sns
```

```
mean_found_job = df1['EmployeeLeft'].mean()

fig, ax1 = plt.subplots(figsize=(10, 7))
sns.barplot(x=agg_employeetarget.index, y=agg_employeetarget['National Origin (code)'], ax=ax1).\
set_title('Proportion of total who got attach', fontsize=16, fontweight='bold')
ax1.axhline(mean_found_job, color='k', linestyle=':')
ax1.set(xlabel='', ylabel='')
ax1.xaxis.set_tick_params(labelsize=14)
ax1.yaxis.set_tick_params(labelsize=14)
```

```
mean_found_job = df['EmployeeLeft'].mean()

fig, ax1 = plt.subplots(figsize=(10, 7))
sns.barplot(x=agg_employeetarget.index, y=agg_employeetarget['Hired through SMTP'], ax=ax1).\
set_title('Proportion of total who got attach', fontsize=16, fontweight='bold')
ax1.axhline(mean_found_job, color='k', linestyle=':')
ax1.set(xlabel='', ylabel='')
ax1.xaxis.set_tick_params(labelsize=14)
ax1.yaxis.set_tick_params(labelsize=14)
```

```
mean_found_job = df["Survey, Relative, Peer's Average Review of Employee"].mean()

fig, ax1 = plt.subplots(figsize=(10, 7))
sns.barplot(x=agg_employeetarget.index, y=agg_employeetarget["Survey, Relative, Peer's Average Review of Employee"], ax=ax1).\
set_title('Proportion of total who got attach', fontsize=16, fontweight='bold')
ax1.axhline(mean_found_job, color='k', linestyle=':')
ax1.set(xlabel='', ylabel='')
ax1.xaxis.set_tick_params(labelsize=14)
ax1.yaxis.set_tick_params(labelsize=14)
```

```
mean_found_job = df['National Origin (code)'].mean()

fig, ax1 = plt.subplots(figsize=(10, 7))
sns.barplot(x=agg_employeetarget.index, y=agg_employeetarget['National Origin (code)'], ax=ax1).\
set_title('Proportion of total who got attach', fontsize=16, fontweight='bold')
ax1.axhline(mean_found_job, color='k', linestyle=':')
ax1.set(xlabel='', ylabel='')
ax1.xaxis.set_tick_params(labelsize=14)
ax1.yaxis.set_tick_params(labelsize=14)
```

```

RF = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=0)
model = RF.fit(X, y)

y_pred = RF.predict(X_test)
round(RF.score(X,y), 4)
```

- Save model file

```
import pickle

filename = 'finalized_model.pkl'
pickle.dump(model, open(filename, 'wb'))
```

```
df.groupby('EmployeeLeft').mean().round(1)
```

- score new data set

```

dfholdscore = pd.read_csv('Sample_HoldoutDataset.csv')
dfholdscore.head()
dfholdscore.columns
```

```
import pickle
filename = 'finalized_model.pkl'

dfscore = pd.read_csv('Sample_HoldoutDataset.csv')
```

```
df2 = pd.get_dummies(dfscore)
yscore = df2.iloc[:,1]
Xscore = df2.iloc[:,:18]
Xscore = Xscore.drop(columns=['EmployeeLeft'])
```

```

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(Xscore, yscore)
print(result)

y_pred = loaded_model.predict(Xscore)
round(model.score(Xscore,yscore), 4)
round(loaded_model.score(Xscore,yscore), 4)
```

```
dfoutput = dfholdscore

dfoutput["ypred"] = y_pred
dfoutput[["EmployeeLeft","ypred"]].head(30)
dfoutput[["EmployeeLeft","ypred"]].head()
```

- Now starts the Fairness, Explanation and Error analysis

```
y = df1.iloc[:,1]
X = df1.iloc[:,:18]
```

```
X = X.drop(columns=['EmployeeLeft'])
```

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

```
RF = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=0)
model = RF.fit(X_train, y_train)

y_pred = RF.predict(X_test)
round(RF.score(X_test,y_test), 4)
```

```
from interpret.ext.blackbox import TabularExplainer
```

```
explainer = TabularExplainer(model, 
                             X_train)
```

```
# Passing in test dataset for evaluation examples - note it must be a representative sample of the original data
# X_train can be passed as well, but with more examples explanations will take longer although they may be more accurate
global_explanation = explainer.explain_global(X_test)
```

```
# Sorted SHAP values
print('ranked global importance values: {}'.format(global_explanation.get_ranked_global_values()))
# Corresponding feature names
print('ranked global importance names: {}'.format(global_explanation.get_ranked_global_names()))
# Feature ranks (based on original order of features)
print('global importance rank: {}'.format(global_explanation.global_importance_rank))

# Note: Do not run this cell if using PFIExplainer, it does not support per class explanations
# Per class feature names
print('ranked per class feature names: {}'.format(global_explanation.get_ranked_per_class_names()))
# Per class feature importance values
print('ranked per class feature values: {}'.format(global_explanation.get_ranked_per_class_values()))
```

```
# Print out a dictionary that holds the sorted feature importance names and values
print('global importance rank: {}'.format(global_explanation.get_feature_importance_dict()))
```

```
# feature shap values for all features and all data points in the training data
print('local importance values: {}'.format(global_explanation.local_importance_values))
```

```
# You can pass a specific data point or a group of data points to the explain_local function

# E.g., Explain the first data point in the test set
instance_num = 1
local_explanation = explainer.explain_local(X_test[:instance_num])
```

```
# Get the prediction for the first member of the test set and explain why model made that prediction
prediction_value = RF.predict(X_test)[instance_num]

sorted_local_importance_values = local_explanation.get_ranked_local_values()[prediction_value]
sorted_local_importance_names = local_explanation.get_ranked_local_names()[prediction_value]

print('local importance values: {}'.format(sorted_local_importance_values))
print('local importance names: {}'.format(sorted_local_importance_names))
```

```
from raiwidgets import ExplanationDashboard
```

```
ExplanationDashboard(global_explanation, model, dataset=X_test, true_y=y_test)
```

- Fairness

```
A_test = X_test['National Origin (code)']

from raiwidgets import FairnessDashboard

# A_test contains your sensitive features (e.g., age, binary gender)
# y_true contains ground truth labels
# y_pred contains prediction labels

FairnessDashboard(sensitive_features=A_test,
                  y_true=y_test.tolist(),
                  y_pred=y_pred)
```

```
from raiwidgets import ErrorAnalysisDashboard

ErrorAnalysisDashboard(global_explanation, model, dataset=X_test, true_y=y_test)
```