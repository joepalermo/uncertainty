import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda, Embedding, Dense, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt

# constants
model_filepath = 'monte_carlo_dropout_networks/models/model.h5'

# load information about categorical columns
with open('data/bulldozer/categorical_sizes.json') as f:
    categorical_sizes_str = f.read()
    categorical_sizes = json.loads(categorical_sizes_str)

# prepare training inputs and targets
train_df = pd.read_csv('data/bulldozer/processed_train.csv')
train_inputs = train_df.drop('target', axis=1)
all_cols_set = set(train_inputs.columns)
categorical_cols_set = set(list(categorical_sizes.keys()))
non_categorical_cols = list(all_cols_set - categorical_cols_set)
column_order = sorted(list(categorical_cols_set)) + sorted(non_categorical_cols)
# normalize non-categorical columns
non_categorical_train_mean = train_inputs[non_categorical_cols].mean(axis=0)
non_categorical_train_std = train_inputs[non_categorical_cols].std(axis=0)
train_inputs[non_categorical_cols] -= non_categorical_train_mean
train_inputs[non_categorical_cols] /= non_categorical_train_std
# ensure that inputs are presented in the right order
train_inputs = train_inputs[column_order]
x_train = train_inputs.values
y_train = train_df['target'].values

# split training and validation
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.1)

# load test data
test_df = pd.read_csv('data/bulldozer/processed_test.csv')
test_inputs = test_df.drop('target', axis=1)
# normalize non-categorical columns
test_inputs[non_categorical_cols] -= non_categorical_train_mean
test_inputs[non_categorical_cols] /= non_categorical_train_std
# ensure that inputs are presented in the right order
test_inputs = test_inputs[column_order]
x_test = test_inputs.values
y_test = test_df['target'].values

print("target range")
print(f"train: {y_train.min(), y_train.mean(), y_train.max()}")
print(f"test: {y_test.min(), y_test.mean(), y_test.max()}")

# plot target distribution
# plt.hist(y_train)
# plt.hist(y_test)
# plt.show()

# define parameters
n_epochs = 100
batch_size = 1024
input_size = x_train.shape[1]
hidden_sizes = [128]
load_model_flag = False

# either load model or train one from scratch
if load_model_flag and os.path.isfile(model_filepath):
    model = load_model(model_filepath)
else:
    # define model architecture
    inputs = keras.Input(shape=(input_size,))
    embedding_layers = list()
    for i, col_name in enumerate(sorted(list(categorical_sizes.keys()))):
        categorical_size = categorical_sizes[col_name]
        embedding_size = int(categorical_size**(0.5))
        ith_input_slice = Lambda(lambda x: x[:,i])(inputs)
        embedding = Embedding(categorical_size, embedding_size, input_length=1)(ith_input_slice)
        embedding_layers.append(embedding)
    numeric_inputs_slice = Lambda(lambda x: x[:,len(categorical_sizes):])(inputs)
    to_concat = embedding_layers + [numeric_inputs_slice]
    all_inputs = Concatenate(axis=1)(to_concat)
    hidden_input = all_inputs
    for hidden_size in hidden_sizes:
        hidden_output = Dense(hidden_size, activation='relu')(hidden_input)
        hidden_input = hidden_output
    outputs = Dense(1, activation='linear')(hidden_output)
    model = Model(inputs, outputs)

    # define optimization procedure
    lr_annealer = ReduceLROnPlateau(monitor='mean_squared_error', factor=0.1, patience=2)
    early_stopper = EarlyStopping(monitor='mean_squared_error', min_delta=0, patience=3)
    model.compile(optimizer=Adam(lr=0.0001),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])
    model.fit(x_train, y_train, epochs=n_epochs,
                                batch_size=batch_size,
                                validation_data=(x_validation, y_validation),
                                callbacks=[lr_annealer, early_stopper],
                                verbose=True)
    model.save(model_filepath)

# get model predictions
model_preds = model.predict(x_test).flatten()
# get baseline predictions
baseline_preds = y_train.mean() * np.ones(len(y_test))

# evaluate
model_test_rmse = np.sqrt(mean_squared_error(y_test, model_preds))
baseline_test_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
model_mae = mean_absolute_error(y_test, model_preds)
baseline_mae = mean_absolute_error(y_test, baseline_preds)
print()
print(f"model test RMSE: {model_test_rmse}")
print(f"baseline test RMSE: {baseline_test_rmse}")
print()
print(f"model test MAE: {model_mae}")
print(f"baseline test MAE: {baseline_mae}")
