from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open('CarPricePredictor.pkl', 'rb'))

# Load dataset
car = pd.read_csv('quikr_car.csv')

# Clean dataset
car = car[car['year'].astype(str).str.isnumeric()]
car['year'] = car['year'].astype(int)

car = car[car['Price'] != 'Ask For Price']
car['Price'] = car['Price'].str.replace(',', '')
car['Price'] = car['Price'].astype(int)

car = car[car['kms_driven'] != 'Petrol']
car['kms_driven'] = car['kms_driven'].str.replace(',', '')
car['kms_driven'] = car['kms_driven'].str.replace(' kms', '')
car['kms_driven'] = car['kms_driven'].astype(int)

car = car.dropna()

companies = sorted(car['company'].unique())
car_models = sorted(car['name'].unique())
years = sorted(car['year'].unique(), reverse=True)
fuel_types = sorted(car['fuel_type'].dropna().unique())


@app.route('/')
def index():
    return render_template(
        'index.html',
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_types=fuel_types
    )


@app.route('/predict', methods=['POST'])
def predict():

    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kms_driven'))

    data = pd.DataFrame(
        [[car_model, company, year, kms_driven, fuel_type]],
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
    )

    prediction = model.predict(data)

    return str(round(prediction[0]))


if __name__ == "__main__":
    app.run(debug=True)