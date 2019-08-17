import json
from monte_carlo_dropout_networks.network import MCDNN


def train_model(train_df):
    hyperparameters = {'n_epochs': 100,
                       'batch_size': 1024,
                       'hidden_sizes': [128],
                       'validation_percentage': 0.1}
    with open('data/bulldozer/categorical_sizes.json') as f:
        categorical_sizes_str = f.read()
        categorical_sizes = json.loads(categorical_sizes_str)
    input_size = train_df.shape[1] - 1  # target is not included input size
    model = MCDNN(input_size, hyperparameters, categorical_sizes)
    model.train(train_df)
    return model


def evaluate_model(test_df, model):
    metric = model.evaluate(test_df)
    return metric


def selection_function(train_df, selection_size):
    pass

