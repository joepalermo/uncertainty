import numpy as np
import pandas as pd
from utils import sample_without_replacement

original_remaining_train_df = pd.read_csv('data/bulldozer/train.csv')
test_df = pd.read_csv('data/bulldozer/test.csv')

# active learning parameters
n_experiments = 1
starting_size = 100
selection_batch_size = 10
n_selection_rounds = 100

def train_model(train_df):
    pass
    return model


def evaluate_model(test_df, model):
    pass
    return metric

all_metrics = list()
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
    all_metrics.append(experiment_metrics)

