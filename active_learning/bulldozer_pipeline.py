import json
import pandas as pd
from monte_carlo_dropout_networks.utils import train_model, evaluate_model, selection_function
from active_learning.utils import run_experiments
from utils import sample_without_replacement

train_df = pd.read_csv('data/bulldozer/train.csv')
test_df = pd.read_csv('data/bulldozer/test.csv')
# load information about categorical columns
with open('data/bulldozer/categorical_sizes.json') as f:
    categorical_sizes_str = f.read()
    categorical_sizes = json.loads(categorical_sizes_str)

# active learning parameters
experiment_params = {'n_experiments': 2,
                     'n_selection_rounds': 10,
                     'starting_size': 1000,
                     'selection_size': 100}

# baseline experiment using random selection
run_experiments(train_df, test_df, sample_without_replacement, train_model, evaluate_model, experiment_params)

# active-learning experiment
run_experiments(train_df, test_df, selection_function, train_model, evaluate_model, experiment_params)
