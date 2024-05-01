# %% [markdown]
# **PARKINSON DISEASE DETECTION USING MACHINE LEARNING**

# %% [markdown]
# **DATA PREPROCESSING**

# %%
import pandas as pd
import numpy as np

# Load the dataset that needs to be cleaned
data = pd.read_excel('data.xlsx')

# %%
#read the data into a pandas dataframe
df = pd.DataFrame(data)
type(df)
df.info()

# %%
df.describe()

# %%
#Drop missing value records and reset to proper index
df = df.dropna()
df = df.reset_index(drop=True)
df

# %%
df = df.sort_values(by=['name'])
df = df.reset_index(drop=True)
df.to_excel("ProcessedPDDdata.xlsx")
df

# %%
df.describe()

# %% [markdown]
# BUILDING THE XGBCLASSIFIER USING THE PROCESSED DATA

# %%
#Seperating the features and target varible for model training
features = df.loc[:,df.columns!='status'].values[:,1:]
label = df.loc[:,'status'].values

print("Features:\nType:"+str(type(features))+"\nShape of array:"+str(features.shape)+"\n"+str(features)+"\n")
print("Label:\nType:"+str(type(label))+"\nShape of array:"+str(label.shape)+"\n"+str(label))

# %%
print(label[label==1].shape[0], label[label==0].shape[0])

# %%
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

#Scaling all the feature values to a similar range bound
scaler = MinMaxScaler((-1,1))
x = scaler.fit_transform(features)
y = label

x

# %%
from sklearn.model_selection import train_test_split,cross_val_score, KFold

#Splitting the dataset into train and test data
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)

print("Shape of training dataset: " + str(x_train.shape))
print("Shape of testing dataset: " + str(x_test.shape))

#create an evaluation dataset to track progress of training process
eval_set = [(x_train, y_train), (x_test, y_test)]

# %%
from xgboost import XGBClassifier, plot_importance

# Defining the classification model [XGBClassifier] with custom parameters
classifier = XGBClassifier(
    eta = 0.3,
    n_estimators = 100,
    max_depth = 2,
    booster = 'gbtree',
    eval_metric = 'error',
    seed = 0
)
classifier.fit(x_train, y_train, eval_set=eval_set)

# %% [markdown]
# **ANALYSING THE TRAINED MODEL**

# %%
import matplotlib.pyplot as pyplot

# retrieve performance metrics
results = classifier.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)

# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error')
pyplot.show()

# %% [markdown]
# KFOLD VALIDATION WITH 5 FOLDS

# %%
# Define the number of folds for cross-validation
n_folds = 5

kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Perform cross-validation on the entire dataset
cv_scores = cross_val_score(classifier, x, y, cv=kf)

print("Cross-validation scores:", cv_scores)
print("Mean CV score:", np.mean(cv_scores))
print("Standard deviation of CV scores:", np.std(cv_scores))

# %%
print("Model Weights:\n" + str(classifier.feature_importances_))

# plot the feature importances and/or model weights
pyplot.bar(range(len(classifier.feature_importances_)), classifier.feature_importances_)
pyplot.show()

# %% [markdown]
# **MODEL PERFOMANCE STATISTICS**

# %%
# Make predictions using the trained model and y_test data
y_pred = classifier.predict(x_test)

print("Accuracy score of the classifier: "+str(accuracy_score(y_test, y_pred)*100))
print("Classification Report:\n" + str(classification_report(y_test, y_pred)))

# %% [markdown]
# **SAVING THE TRAINED MODEL**

# %%
import joblib

# Save the trained model
joblib.dump(classifier, 'ParkinsonsPredictor.sav')
classifier.save_model('xgbclassifier.json')


