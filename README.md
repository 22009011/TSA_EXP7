# Developed By : THANJIYAPPAN K
# Register Number : 212222240108
# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM:
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

file_path = 'ev.csv'  # Replace with the correct file path
data = pd.read_csv(file_path)

yearly_data = data.groupby('year')['value'].sum()

diff_data = yearly_data.diff().dropna()

result = adfuller(diff_data)
print('ADF Statistic (After Differencing):', result[0])
print('p-value (After Differencing):', result[1])

train_data = diff_data[:int(0.8 * len(diff_data))]
test_data = diff_data[int(0.8 * len(diff_data)):]

lag_order = min(5, len(train_data) - 1)  # Ensure lag order < number of observations
model = AutoReg(train_data, lags=lag_order)
model_fit = model.fit()

max_lags = len(diff_data) // 2

plt.figure(figsize=(10, 6))
plot_acf(diff_data, lags=max_lags, alpha=0.05)
plt.title('Autocorrelation Function (ACF) - Differenced Value')
plt.show()

plt.figure(figsize=(10, 6))
plot_pacf(diff_data, lags=max_lags, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF) - Differenced Value')
plt.show()

predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

mse = mean_squared_error(test_data, predictions)
print('Mean Squared Error (MSE):', mse)

plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data.values, label='Test Data - Differenced Value', color='blue', linewidth=2)
plt.plot(test_data.index, predictions, label='Predictions - Differenced Value', color='orange', linestyle='--', linewidth=2)
plt.xlabel('Year')
plt.ylabel('Differenced Value')
plt.title('AR Model Predictions vs Test Data (Differenced Value)')
plt.legend()
plt.grid(True)
plt.show()

```

### OUTPUT:

GIVEN DATA
![image](https://github.com/user-attachments/assets/b1221f82-4cc3-41dd-b47c-aa5759062192)



PACF - ACF
![image](https://github.com/user-attachments/assets/a816602d-ee44-4b68-9421-a71be87b526d)

![image](https://github.com/user-attachments/assets/932c5185-ef2a-4aa1-8c56-3bb4c3fea7bd)


PREDICTION
![image](https://github.com/user-attachments/assets/ca931c06-67ac-42b4-9c9d-4383265c5d13)



### RESULT:
Thus ,program was successfully implemented the auto regression function using python.
