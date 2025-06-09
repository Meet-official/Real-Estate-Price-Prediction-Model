# Real-Estate Price Predictor

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# Storing the data
housing = pd.read_csv("data.csv")

# Train-Test Splitting, ahiya Stratified Shuffle Split use karyu 6e, aanathi badha type na data both train and test ma proper rite split thase
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# housing ma training data copy karyo
housing = strat_train_set.copy()

# Combining attributes(columns) to get better results
# housing["TAXRM"] = housing['TAX']/housing['RM']   -> (not needed)

# Checking Correlations, aanathi khabar padse ke kayi column ni value change karva thi price vadi value direct affect thaay 6e
corr_matrix = housing.corr()

housing = strat_train_set.drop("MEDV", axis = 1)  # housing ma training data mathi MEDV-Price vaadi column drop kari
housing_labels = strat_train_set["MEDV"].copy()   # and e MEDV-Price column ne housing_labels ma label tarike store kari

# Filling null values with the median
median = housing["RM"].median()

# Imputer a statistical method, used to replace missing values in a dataset with estimated values
imputer = SimpleImputer(strategy = "median")    
imputer.fit(housing)    # Imputer use karine easily median fit kari sakiye 6e

# X ma just training data without null values store karyu
X = imputer.transform(housing)    

# Have aapda X ma badhi columns 6e train dataset ni, eno DataFrame banaay didho
housing_tr = pd.DataFrame(X, columns = housing.columns)  

# Creating Pipeline:  (To Preprocess Data)
# We can add multiple preprocessing steps like transformers or estimators (like imputer, scaler, encoder) into a single object.
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = my_pipeline.fit_transform(housing)


# Selecting a desired Model for Real-Estates

# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor() 
model.fit(housing_num_tr, housing_labels)

# Evaluating the model
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)

# Using better Evaluation technique - Corss Validation
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring = "neg_mean_squared_error", cv = 10)
rmse_scores = np.sqrt(-scores)

# Creating a function to get scores in one click
def print_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())

print_scores(rmse_scores)


# Saving the Model using joblib

from joblib import dump, load
dump(model, 'Real-Estate.joblib')


# Testing the model on test data

X_test = strat_test_set.drop("MEDV", axis = 1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
# print(final_predictions, list(Y_test))

print(f"Final RMSE on Test Set: {final_rmse}")


# Using the Model

from joblib import dump, load

model = load('Real-Estate.joblib')

input = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.2536354 , -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])

prediction = model.predict(input)

print("Predicted Price:", prediction)
