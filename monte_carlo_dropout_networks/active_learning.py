import json
import pandas as pd
from utils import sample_without_replacement
from monte_carlo_dropout_networks.network import MCDNN
import matplotlib.pyplot as plt

original_remaining_train_df = pd.read_csv('data/bulldozer/processed_train.csv')
test_df = pd.read_csv('data/bulldozer/processed_test.csv')
# load information about categorical columns
with open('data/bulldozer/categorical_sizes.json') as f:
    categorical_sizes_str = f.read()
    categorical_sizes = json.loads(categorical_sizes_str)

# active learning parameters
n_experiments = 2
starting_size = 10000
selection_batch_size = 10000
n_selection_rounds = 10

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

all_experiments = list()
# run active learning experiments
for experiment_i in range(n_experiments):
    experiment_metrics = list()
    # re-initialize whole training data set
    remaining_train_df = original_remaining_train_df.copy()
    # sample starting population
    selected_population_df, remaining_train_df = sample_without_replacement(starting_size, remaining_train_df)
    for selection_round in range(n_selection_rounds):
        print(f"running selection round #{selection_round}")
        # select a new batch randomly
        new_selection_df, remaining_train_df = sample_without_replacement(selection_batch_size, remaining_train_df)
        selected_population_df = pd.concat([selected_population_df, new_selection_df])
        # train a model on the samples selected thus far
        model = train_model(selected_population_df)
        # evaluate trained model
        metric = evaluate_model(test_df, model)
        experiment_metrics.append(metric)
        print(experiment_metrics)
        all_experiments.append(experiment_metrics)

# plot results
for experiment_metrics in all_experiments:
    plt.hist(experiment_metrics)
plt.show()