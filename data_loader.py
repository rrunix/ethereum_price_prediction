import pandas as pd
import numpy as np

class PreprocessData:

    def __init__(self, df):
        self.feature_pos = dict(list(zip(df.columns, range(len(df.columns)))))
        self.inverse_transform = None

    def matrix_view(self, df, feature, index=None):
        return df[index, self.feature_pos[feature]]

    def inverse_normalise_feature(self, feature_name, values):
        observations = np.zeros(shape=(len(values), len(self.feature_pos)))
        feature_position = self.feature_pos[feature_name]
        observations[:, feature_position] = values
        return self.inverse_trasform(observations)[:, feature_position]


def load_data(filename, window_size, max_row = 1000000, remove_features = [], preprocess_args={}):
    df = pd.read_json(filename)
    df = df.head(max_row)

    # Remove the features in remove_features aray
    select_features(df, remove_features)

    dfs = []
    target_values = []
    preprocess_data = []

    for batch in create_window(df, window_size):
        processed_batch, target_value, data = preprocess_batch(batch, **preprocess_args)

        dfs.append(processed_batch)
        target_values.append(target_value)
        preprocess_data.append(data)

    dfs = np.array(dfs)
    dfs = np.reshape(dfs, newshape=(dfs.shape[0], dfs.shape[1], len(df.columns)))

    target_values = np.reshape(target_values, newshape=(len(target_values), 1))

    return dfs, target_values, preprocess_data


def create_window(df, window_size):
    for index in range(df.shape[0] - window_size - 1):
        yield df[index: index+window_size + 1].copy()


def preprocess_batch(df_input, target_feature="close", normaliser=None):
    # Is not necessary to create a new copy because it has been already done by the window function
    df = df_input

    preprocess_data = PreprocessData(df)

    # Normalise the features (if normaliser is provided)
    df = normalise_vectorized(df, preprocess_data, normaliser)

    # Get the features we want to predict from the last row and remove it.
    target_value = df[target_feature].iloc[-1]
    df.drop(df.index[df.shape[0] - 1], inplace=True)

    # Convert the dataframe to a matrix
    df = df.as_matrix()

    return df, target_value, preprocess_data


def select_features(df, remove_features):
    for field in remove_features:
        if field in df:
            del df[field]


def normalise_vectorized(df, preprocess_data, normaliser):
    if normaliser:
        df.iloc[:,:] = normaliser.fit_transform(df)
        preprocess_data.inverse_transform = normaliser.inverse_transform

    return df