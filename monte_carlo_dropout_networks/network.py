from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate
from tensorflow.keras.models import Model

class Model:

    def __init__(self, input_size, hidden_sizes, output_size, categorical_sizes):
        inputs = keras.Input(shape=(input_size,))

# how you do dropout at inference time
# outputs = keras.layers.Dropout(0.5)(x, training=True)
