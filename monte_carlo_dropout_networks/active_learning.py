import json
import pandas as pd
from utils import sample_without_replacement
from monte_carlo_dropout_networks.network import MCDNN
from active_learning.utils import run_experiments

train_df = pd.read_csv('data/bulldozer/processed_train.csv')
test_df = pd.read_csv('data/bulldozer/processed_test.csv')
# load information about categorical columns
with open('data/bulldozer/categorical_sizes.json') as f:
    categorical_sizes_str = f.read()
    categorical_sizes = json.loads(categorical_sizes_str)

# active learning parameters
experiment_params = {'n_experiments': 2,
                     'n_selection_rounds': 10,
                     'starting_size': 10000,
                     'selection_size': 10000}

# model hyperparameters
hyperparameters = {'n_epochs': 100,
                   'batch_size': 1024,
                   'hidden_sizes': [128],
                   'validation_percentage': 0.1}

def train_model(train_df):
    input_size = train_df.shape[1] - 1 # target is not included input size
    model = MCDNN(input_size, categorical_sizes, hyperparameters)
    model.train(train_df)
    return model

def evaluate_model(test_df, model):
    metric = model.evaluate(test_df)
    return metric

def selection_function(df, selection_size):
    return sample_without_replacement(df, selection_size)

run_experiments(train_df, test_df, sample_without_replacement, train_model, evaluate_model, experiment_params)