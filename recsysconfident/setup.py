import json
import os
import time


class Setup:

    def __init__(self, model_name: str,
                 database_name: str,
                 conf_calibration: bool = False,
                 instance_dir: str = None,
                 split_position: int = 0,
                 fit_mode: int = 0,
                 batch_size: int = 1024,
                 learning_rate: float = 0.001,
                 patience: int = 5,
                 rate_range: list = None,
                 timestamp: str = None,
                 min_inter_per_user: int =72,
                 reevaluate:bool = False,
                 learn_to_rank:bool=True):

        self.model_name = model_name
        self.database_name = database_name
        self.conf_calibration = conf_calibration
        self.split_position = split_position
        self.fit_mode = fit_mode
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.min_inter_per_user = min_inter_per_user
        self.reevaluate = reevaluate
        self.learn_to_rank = learn_to_rank

        self.set_rate_range(rate_range)
        self.setup_instance_dir(instance_dir)

    def set_split_position(self, split_position):
        self.split_position = split_position
        self.setup_instance_dir(None)

    def set_rate_range(self, rate_range: list[float]):

        if not rate_range:
            if os.path.isfile(f"./data/{self.database_name}/info.json"):
                with open(f"./data/{self.database_name}/info.json") as f:
                    info = json.load(f)
                self.rate_range = info.get("rate_range", None)

                if not self.rate_range:
                    raise Exception(f"No rate_range specified in {self.database_name}")
            else:
                raise Exception(f"No info.json found for {self.database_name}")
        else:
            self.rate_range = rate_range

    def setup_instance_dir(self, instance_dir: str):

        if instance_dir is None:
            self.work_dir = f"./runs/{self.database_name}-{self.model_name}"
            instance_dir = f"{self.work_dir}-{self.split_position}"

        self.instance_dir = instance_dir
        os.makedirs(name=instance_dir, exist_ok=True)

    def to_dict(self) -> dict:
        return {
            'model_name': self.model_name,
            'database_name': self.database_name,
            'instance_dir': self.instance_dir,
            'split_position': self.split_position,
            'conf_calibration': self.conf_calibration,
            'fit_mode': self.fit_mode,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'patience': self.patience,
            'rate_range': self.rate_range,
            'min_inter_per_user': self.min_inter_per_user,
            'reevaluate': self.reevaluate,
            'timestamp': time.strftime('%Y-%m-%d-%H-%M-%S'),
            "learn_to_rank": self.learn_to_rank
        }
