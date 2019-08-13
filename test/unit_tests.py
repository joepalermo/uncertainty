import pandas as pd
from utils import *

def test_sample_without_replacement():
    starting_size_of_remaining_train_df = 100
    selection_batch_size = 10
    remaining_train_df = pd.DataFrame(np.zeros((starting_size_of_remaining_train_df,2)))
    new_selection_df, remaining_train_df = sample_without_replacement(selection_batch_size, remaining_train_df)
    assert len(new_selection_df) == selection_batch_size
    assert len(remaining_train_df) == starting_size_of_remaining_train_df - selection_batch_size

