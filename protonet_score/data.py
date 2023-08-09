import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
# Reading and pre-processing data
def load_data():
  data_u = pd.read_csv(r'your file')
  data_use = data_u.loc[:,['Enter your data characteristics']]

  X = data_use.copy()
  y = X.pop('target')

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

  X_train = np.asarray(X_train)
  X_test = np.asarray(X_test)
  y_train = np.asarray(y_train)
  y_test = np.asarray(y_test)

  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.fit_transform(X_test)

  # Returns preprocessed data
  return X_train_scaled, y_train, X_test_scaled, y_test
