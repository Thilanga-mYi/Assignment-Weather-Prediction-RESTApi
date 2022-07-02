from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

weather_dataset = pd.read_csv('bataAssignmentPy.csv')
# print(weather_dataset.head())

data_dummies = pd.get_dummies(weather_dataset["pedictedWeather"])
print(data_dummies.head(122))

concated_dataset = pd.concat([weather_dataset,data_dummies],axis=1)
concated_dataset_drop_predictionWeather = concated_dataset.drop(concated_dataset.columns[[5]],axis = 1)

new_cols_order = ["waterLevel","humidity","temperature","windSpeed",'cloudness','Clouds','Rains','Sunny','predictingToFull']
preprocessed_dataset = concated_dataset_drop_predictionWeather.reindex(columns=new_cols_order)
# print(preprocessed_dataset.head(10))

X = preprocessed_dataset.iloc[:, 0:8].values
y = preprocessed_dataset.iloc[:, 8].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(y_pred)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

y_pred_test1 = regressor.predict([[11,24,41,27,70,0,0,1]])
print(y_pred_test1)

@app.route('/prediction', methods=['GET','POST'])
def prediction():

    clouds = 0
    rains = 0
    sunny = 0

    from cerberus import Validator

    preictionWaterLevel_rule = {'waterLevel': {'type': 'float', 'required': True, 'min': 0, 'max': 100}}
    predictionWeather_rule = {'predictedWeather': {'type': 'string', 'required': True, 'allowed': ['sunny', 'rains', 'clouds']}}

    validation_schema = {
        preictionWaterLevel_rule,
        predictionWeather_rule
    }

    validate = Validator(validation_schema)

    document = {
        'waterLevel': request.args.get("waterLevel"),
        'predictedWeather': request.args.get("predictedWeather")
    }

    print(validate.validate(document))

    if validate.validate(document) == False:
        return jsonify(
            predictedToFull_value = validation_schema
        )

    wl = request.args.get("waterLevel")
    hum = request.args.get("humidity")
    temp = request.args.get("temperature")
    ws = request.args.get("windSpeed")
    cl = request.args.get("cloudness")
    pw = request.args.get("predictedWeather")

    if pw == 'sunny':
        sunny = 1
    elif pw == 'rains':
        rains = 1
    elif pw == 'clouds':
        clouds = 1

    data = [wl, hum, temp, ws, cl, clouds, rains, sunny]

    return jsonify(
        predictedToFull_value = regressor.predict([data])[0]
    )

@app.route('/')
def hello():
    return "Weather Prediction"
