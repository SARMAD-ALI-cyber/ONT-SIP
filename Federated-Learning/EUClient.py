import tensorflow as tf, keras
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from plot_keras_history import plot_history
import numpy as np
import pandas as pd
import flwr as fl

# data loading this data is considered as data of client
train_data = pd.read_csv('EU_train_class.csv')
test_data = pd.read_csv('EU_test_class.csv')
# splitting the data set
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]
# =========converting to array===================
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
# =========reshaping=============================
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
# ===========scaling================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model initialization like initalizing it's parameters only
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_initializer='he_normal'),
    tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dense(16, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='he_normal')
], name='USA-Teacher')

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])


# =================================Starting Federated Learning==================

# here model which was previously initialized will train on user data

class ONTClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        history = model.fit(X_train, y_train, epochs=100, batch_size=X_train.shape[0],
                            validation_data=(X_test, y_test))
        return model.get_weights(), len(X_train), {"Train_loss": history.history['loss'][-1]}

    def evaluate(self, parameters, config):
        # model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test)
        return loss, len(X_test), {"accuracy": accuracy}


# start flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080",
                             client=ONTClient()
)


