import pandas as pd
from clearml import Task

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


task = Task.init(project_name="demo", task_name="Console Run")

data = pd.read_csv('Housing.csv')
y = data[['price']].values
X = data.drop(['price', 'area'], axis=1)


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

X = X_transformer.fit_transform(X)
y = y_transformer.fit_transform(y)

model = LinearRegression()

model.fit(X, y)

score = model.score(X, y)

task.close()