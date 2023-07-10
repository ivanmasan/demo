import pickle

import numpy as np
import pandas as pd
from clearml import Task, Logger, OutputModel
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


task = Task.init(project_name="demo", task_name="Console Run")

train_data = pd.read_csv('train.csv')
valid_data = pd.read_csv('valid.csv')

train_y = train_data[['price']].values
valid_y = valid_data[['price']].values

train_X = train_data.drop(['price', 'area'], axis=1)
valid_X = valid_data.drop(['price', 'area'], axis=1)


one_hot_cols = [
    'bedrooms',
    'bathrooms',
    'stories',
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'parking',
    'prefarea',
    'furnishingstatus'
]

extractor = ColumnTransformer(
    transformers=[('one_hot', OneHotEncoder(sparse=False), one_hot_cols)],
    remainder='drop'
)

X_transformer = Pipeline(
    steps=[
        ('extractor', extractor),
        ('scaler', StandardScaler())
    ]
)

y_transformer = StandardScaler()

train_X = X_transformer.fit_transform(train_X)
transformed_train_y = y_transformer.fit_transform(train_y)

valid_X = X_transformer.transform(valid_X)

model = Lasso(alpha=1)

model.fit(train_X, transformed_train_y)

pred_valid_y = model.predict(valid_X).reshape(-1, 1)
pred_valid_y = y_transformer.inverse_transform(pred_valid_y)

pred_train_y = model.predict(train_X).reshape(-1, 1)
pred_train_y = y_transformer.inverse_transform(pred_train_y)

valid_loss = np.sqrt(((pred_valid_y - valid_y) ** 2).sum())
train_loss = np.sqrt(((pred_train_y - train_y) ** 2).sum())

logger = Logger.current_logger()

logger.report_single_value("Train Loss", train_loss)
logger.report_single_value("Valid Loss", valid_loss)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

task.upload_artifact('model', artifact_object='model.pkl')

task.close()
