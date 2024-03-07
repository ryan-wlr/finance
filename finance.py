import numpy as np 
from keras.models import Sequential
from keras.layers import LSTM, Dense
import yfinance as yf
from copy import deepcopy
from sklearn.model_selection import train_test_split

# Spaceship titanic train_test_split
# Load historical stock data 
ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2020-12-31")
print(data.head())
X = deepcopy(data[['Close']])
y = deepcopy(data[['Open']])
print(X)
# Preprocess data
# Code to normalize, split into train/test sets, etc.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# Build LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], 1))) 
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# Train model on data
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Evaluate on test set
test_loss = model.evaluate(X_test, y_test)
print(f'Test loss: {test_loss}')
