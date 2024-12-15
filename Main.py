# 1. Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 2. Load and preprocess data
try:
    # 2.1 Load the dataset from a CSV file
    data = pd.read_csv('StressPredict/DATA1.csv')
    print("Data loaded successfully. Columns in the dataset are:")
    print(data.columns)

    # 2.2 Remove leading and trailing spaces from column names
    data.columns = data.columns.str.strip()

    # 2.3 Check for expected columns
    expected_columns = ['Cyberbullying', 'Screen_Time', 'Mental_Health']
    missing_columns = [col for col in expected_columns if col not in data.columns]
    if missing_columns:
        print(f"Error: Columns {', '.join(missing_columns)} not found in data.")
        exit()

    # 2.4 Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data[expected_columns] = imputer.fit_transform(data[expected_columns])

    # 2.5 Handle outliers using IQR (Interquartile Range)
    for col in expected_columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
        data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])

except FileNotFoundError:
    # 2.6 Error handling if the file is not found
    print("Error: Could not find DATA1.csv file.")
    exit()

# 3. Extract features and target
X = data[['Cyberbullying', 'Screen_Time']]  # Features
y = data['Mental_Health']  # Target

# 4. Feature scaling (standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Train the model
model = LinearRegression()  # Using Multiple Regression model
model.fit(X_train, y_train)

# 7. Make predictions
y_pred = model.predict(X_test)

# 8. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 9. Scatter plot for actual vs predicted values
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color='red', label='Predicted')
plt.scatter(y_test, y_test, color='blue', label='Actual', alpha=0.5)  # To differentiate actual values
plt.xlabel('Actual Mental Health')
plt.ylabel('Predicted Mental Health')       
plt.title('Actual vs Predicted Mental Health')
plt.legend()
plt.show()
