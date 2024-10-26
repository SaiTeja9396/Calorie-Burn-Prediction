
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import pickle
import warnings
warnings.filterwarnings("ignore")


df_x = pd.read_csv('C:/Users/saite/Downloads/exercise.csv')

df_x['Gender'] = LabelEncoder().fit_transform(df_x['Gender'])

y = df_x['Calories']
x = df_x.drop(['Calories', 'User_ID'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

model_1 = LinearRegression()
model_1.fit(x_train, y_train)

train_score_1 = model_1.score(x_train, y_train)
test_score_1 = model_1.score(x_test, y_test)

y_predict = model_1.predict(x_test)
mae_1 = mean_absolute_error(y_test, y_predict)

accuracy_1 = (1 - mae_1 / np.mean(y_test)) * 100

print(f'Linear Regression R2 Training Score: {train_score_1:.4f}')
print(f'Linear Regression R2 Testing Score: {test_score_1:.4f}')
print(f'Linear Regression Accuracy: {accuracy_1:.2f}%')


model_2 = XGBRegressor()
model_2.fit(x_train, y_train)

train_score_2 = model_2.score(x_train, y_train)
test_score_2 = model_2.score(x_test, y_test)

y_predict_2 = model_2.predict(x_test)
mae_2 = mean_absolute_error(y_test, y_predict_2)

accuracy_2 = (1 - mae_2 / np.mean(y_test)) * 100

print(f'XGBoost R2 Training Score: {train_score_2:.4f}')
print(f'XGBoost R2 Testing Score: {test_score_2:.4f}')
print(f'XGBoost Accuracy: {accuracy_2:.2f}%')


input_data = ['0', '27', '154.0', '58.0', '10.0', '81.0', '39.8'] 

input_array = np.asarray(input_data).reshape(1, -1)
cal_x = sc.transform(input_array)

predicted_calories = model_2.predict(cal_x)
print('Calories Prediction:', predicted_calories[0])


with open('Calories_model.pkl', 'wb') as f:
    pickle.dump(model_2, f)

with open('scaler.save', 'wb') as f:
    pickle.dump(sc, f)
