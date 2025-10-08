import unittest
import os

from recsysconfident.utils.files import setup_and_model_exists


class TestFiles(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        script_dir = str(os.path.dirname(os.path.abspath(__file__)))
        self.run_folder = f'{str(script_dir[0:script_dir.index("tests")])}/tests/static'
        self.empty_folder = f"{self.run_folder}/sub_empty_folder"

    def test_setup_and_model_exists(self):

        exists = setup_and_model_exists(self.run_folder)
        assert exists, "The model and setup files exists."

    def test_setup_and_model_exists_not(self):

        exists = setup_and_model_exists(self.empty_folder)
        assert not exists, "The model or setup files do not exist."

