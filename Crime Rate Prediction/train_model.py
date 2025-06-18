# train_model.py

import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

# Dummy training data: [year, city_code, population, crime_code]
X = np.array([
    [2011, 0, 63.5, 0],
    [2012, 1, 85.8, 1],
    [2013, 2, 89.2, 2],
    [2014, 3, 21.9, 3],
    [2015, 4, 165.7, 4]
])

# Dummy target: crime rates
y = np.array([2.5, 3.2, 4.0, 1.5, 5.5])

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
with open('Model/model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved successfully.")