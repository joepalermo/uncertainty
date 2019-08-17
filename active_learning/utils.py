import pandas as pd
import matplotlib.pyplot as plt
from utils import sample_without_replacement

def run_experiments(train_df, test_df, selection_function, train_model, evaluate_model, experiment_params):
    n_experiments = experiment_params['n_experiments']
    n_selection_rounds = experiment_params['n_selection_rounds']
    starting_size = experiment_params['starting_size']
    selection_size = experiment_params['selection_size']
    all_experiments = list()
    # run active learning experiments
    for experiment_i in range(n_experiments):
        experiment_metrics = list()
        # re-initialize whole training data set
        remaining_train_df = train_df.copy()
        # sample starting population randomly
        selected_population_df, remaining_train_df = sample_without_replacement(remaining_train_df, starting_size)
        for selection_round in range(n_selection_rounds):
            print(f"running selection round #{selection_round}")
            # select a new batch randomly
            new_selection_df, remaining_train_df = selection_function(remaining_train_df, selection_size)
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