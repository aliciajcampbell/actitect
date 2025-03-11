from typing import Union, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

__all__ = ['Hopkins']


class Hopkins:

    def __init__(self, data: Union[np.ndarray, pd.DataFrame], n_samples: int, random_state: Optional[int] = 42) -> float:
        """
        Adapted with slight changes from https://github.com/lachhebo/pyclustertend to allow calculations for 1d data.

        Assess the clusterability of a dataset as a score between 0 and 1 by evaluating similarity to a uniform
         distribution. A score around 0.5 expresses no clusterability and a score tending to 0 express a high
         cluster tendency.

        Parameters:
            :param data: (np.ndarray/pd.DataFrame) Input data of shape (samples, dim), where dim can be 1 or greater.
            :param n_samples: (int) How many samples to use for the reference uniform dataset. Must be smaller than
            the 'samples' dimension in 'data'.
            :param random_state: (int or None, default=42) Random seed for reproducibility. If None, the global random state is used.

        Returns:
            :return: (float) The hopkins statistic score between 0 and 1.
        """
        self.score = None

        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        rng = np.random.default_rng(random_state)  # set random seed (can be None)

        # get n_samples from the input dataset:
        if n_samples > data.shape[0]:
            raise ValueError(f"'n_samples' ({n_samples}) is larger than the dataset samples {data.shape[0]}")
        dataset_samples = data.sample(n=n_samples, random_state=rng)

        # calculate the distances to nearest neighbors (NNs) in input data:
        nn_distances_data = self.calc_nn_distances(data, dataset_samples)

        # create uniform reference data with same variation:
        uniform_reference_data = self.generate_uniform_reference_data(data, n_samples, rng)

        # calculate the nn distance between input and reference dataset
        nn_distances_data_to_ref = self.nn_distances_data_to_ref(data, uniform_reference_data)

        x = sum(nn_distances_data)
        y = sum(nn_distances_data_to_ref)

        if x + y == 0:
            raise Exception("The denominator of the hopkins statistics is null")

        else:
            self.score = x / (x + y)[0]

    @staticmethod
    def calc_nn_distances(df: pd.DataFrame, data_frame_sample: pd.DataFrame) -> np.ndarray:
        tree = BallTree(df, leaf_size=2)
        dist, _ = tree.query(data_frame_sample, k=2)
        return dist[:, 1]

    @staticmethod
    def generate_uniform_reference_data(df: pd.DataFrame, n_samples: int, rng: np.random.Generator) -> pd.DataFrame:
        max_data_frame, min_data_frame = df.max(), df.min()
        _reference_data = None

        for i in range(len(max_data_frame)):  # generate data for each dimension of
            uniform_values_i = rng.uniform(min_data_frame[i], max_data_frame[i], n_samples)
            if _reference_data is None:
                _reference_data = uniform_values_i.reshape(-1, 1)
            else:
                _reference_data = np.column_stack((_reference_data, uniform_values_i))

        return pd.DataFrame(_reference_data)

    @staticmethod
    def nn_distances_data_to_ref(data_df: pd.DataFrame, reference_df: pd.DataFrame) -> np.ndarray:
        tree = BallTree(data_df, leaf_size=2)
        dist, _ = tree.query(reference_df, k=1)
        return dist
