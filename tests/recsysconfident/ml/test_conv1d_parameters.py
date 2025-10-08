import unittest
from pandas import DataFrame
import numpy as np

from recsysconfident.ml.conv1d_parameters import Conv1DParameters


class TestConv1DParameters(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.USER_COL = "userCol"
        self.ITEM_COL = "itemCol"
        self.RATING_COL = "ratingCol"
        self.ratings_df = DataFrame({self.USER_COL: np.random.randint(0, 61, size=100),
                                     self.ITEM_COL: np.random.randint(0, 61, size=100),
                                     self.RATING_COL: np.random.uniform(0, 60, size=100)})
        self.len_unique_users = 100
        self.len_unique_items = 100

    def test_get_config(self):
        conv1d_parameters = Conv1DParameters(input_size=3706, output_size=256, min_channels=16, max_channels=256)
        configs = conv1d_parameters.get_configs(input_channels=1, kernel_size=5)

        assert configs[-1]['output_size'] == 256

    def test_get_config_quantity(self):
        user_blocks_config = Conv1DParameters(
            input_size=22,
            output_size=1,
            min_channels=16,
            max_channels=128
        ).get_configs(input_channels=1, kernel_size=3, down_target=512, start_stride=1)
        print(user_blocks_config)

    def test_get_config_256(self):
        conv1d_parameters = Conv1DParameters(input_size=128, output_size=1, min_channels=16, max_channels=256)
        configs = conv1d_parameters.get_configs(input_channels=1, kernel_size=5)
        assert configs[-1]['output_size'] == 1

    def test_get_configs_large_input(self):
        conv1d_parameters = Conv1DParameters(input_size=6040, output_size=256, min_channels=16, max_channels=256)
        configs = conv1d_parameters.get_configs(input_channels=1, kernel_size=5, down_target=640)

        assert configs[-1]['output_size'] == 256

    def test_get_configs_610(self):

        conv1d_parameters = Conv1DParameters(input_size=610, output_size=256, min_channels=16, max_channels=256)
        configs_610 = conv1d_parameters.get_configs(input_channels=1, kernel_size=5)

        assert configs_610[-1]['output_size'] == 256

    def test_get_configs_340_556(self):

        conv1d_parameters = Conv1DParameters(input_size=340_556, output_size=256, min_channels=16, max_channels=256)
        configs = conv1d_parameters.get_configs(input_channels=1, kernel_size=5, down_target=520)

        assert configs[-1]['output_size'] == 256

    def test_get_configs_105283(self):

        conv1d_parameters = Conv1DParameters(input_size=105283, output_size=256, min_channels=16, max_channels=256)
        configs = conv1d_parameters.get_configs(input_channels=1, kernel_size=5, down_target=520)

        assert configs[-1]['output_size'] == 256

    def test_get_configs_previous_channels(self):

        conv1d_parameters = Conv1DParameters(input_size=100, output_size=1, min_channels=3, max_channels=256)
        configs = conv1d_parameters.get_configs(input_channels=1, kernel_size=2, down_target=1000)

        assert configs[0]['previous_channels'] == 1

    def test_get_configs_last_previous_channels(self):

        conv1d_parameters = Conv1DParameters(input_size=100, output_size=1, min_channels=3, max_channels=256)
        configs = conv1d_parameters.get_configs(input_channels=1, kernel_size=2, down_target=1000)

        assert configs[-1]['previous_channels'] == configs[-2]['channels']
