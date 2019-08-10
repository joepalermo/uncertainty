from tensorflow import keras


class Model:

    def __init__(self, input_size, hidden_sizes, output_size, categorical_sizes):
    inputs = keras.Input(shape=(10,))
    x = keras.layers.Dense(3)(inputs)
    outputs = keras.layers.Dropout(0.5)(x, training=True)
    model = keras.Model(inputs, outputs)