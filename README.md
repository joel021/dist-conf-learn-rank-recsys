# setup for development

# Install dependencies</h1>
    
    * run ```$pip install -r requirements.txt```

    * Install pytorch
    * pip install torch-geometric torch-sparse torch-scatter torch-cluster torch-spline-conv pyg-lib -f https://data.pyg.org/whl/torch-2.5.0+cpu.html

# Run setup.py for dev environment:

    ```$python setup.py develop ```

# Run the tests:

    ```$pytest tests``` run all tests

    ```$pytest tests/.../file.py``` run especific python file


# Configure poetry to create virtual environment locally
poetry config virtualenvs.in-project true


# Project structure

- recsysconfident: the main package
  - data_handling: responsible for any data preprocessing, dataset buildings, and dataloaders
  - ml: everything related to machine learning
    - fitting: responsible for perform the fit and evaluation of the models
    - models: Where the models implementations are. Each model implementation should have its on class on its on file. Additionally, the method of instantiating the model and its compatible dataloader should be implemented along with the model class.
  - utils: utilities scripts

- files:
  - data/{database_name}/info.json: Describes the database parameters, such as its columns, which one are used and how many columns are in the dataset
  - setups.json: Describes the supported setups of experiments.

- supported datasets and models:
  - recsysconfident/environment.Environment.database_name_fn: Defines which databases are supported to perform experiments. Add or remove instances from this dictionary to control the supported datasets.
  - recsysconfident/environment.Environment.model_name_fn: Defines which models are supported to perform experiments.

- setups: Are the set of configurations that describes an experiment.
- when running main.py with --setup_instance: means that you probably want to rexecute an experiment, also provide --fit_mode to specify whether you want to fit the model or just rerun the evaluation.


** Setups Examples**

- python main.py --setups ./setups-conf-benchmark.json --setup_name k_folds --k_folds 5 
- 