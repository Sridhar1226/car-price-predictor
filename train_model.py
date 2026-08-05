import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("quikr_car.csv")

# Data Cleaning
df = df[df['year'].str.isnumeric()]
df['year'] = df['year'].astype(int)

df = df[df['Price'] != 'Ask For Price']
df['Price'] = df['Price'].str.replace(',', '')
df['Price'] = df['Price'].astype(int)

df = df[df['kms_driven'] != 'Petrol']

df['kms_driven'] = df['kms_driven'].str.replace(',', '')
df['kms_driven'] = df['kms_driven'].str.replace(' kms', '')
df['kms_driven'] = df['kms_driven'].astype(int)

df = df.dropna()

# Features and Target
X = df.drop('Price', axis=1)
y = df['Price']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Model
ohe = OneHotEncoder(handle_unknown='ignore')

column_trans = make_column_transformer(
    (ohe, ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

lr = LinearRegression()

pipe = make_pipeline(column_trans, lr)

# Train Model
pipe.fit(X_train, y_train)

# Save Model
pickle.dump(pipe, open('CarPricePredictor.pkl', 'wb'))

print("Model saved successfully!")