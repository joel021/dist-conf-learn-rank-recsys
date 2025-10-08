import argparse
import glob
import json

import torch

from recsysconfident.environment import Environment
from recsysconfident.ml.eval.inference_error_analysis import export_elementwise_error
from recsysconfident.ml.eval.ranking_evaluation import evaluate
from recsysconfident.setup import Setup
from recsysconfident.utils.files import export_metrics, export_setup, read_json, \
    setup_and_model_exists, setup_model_results_exists
from recsysconfident.setup_manager import setup_fit

def run_all_setups(setups: dict, split_position: int=0, shuffle: bool=False):

    for value in setups.values():
        setup = Setup(**value)
        setup.set_split_position(split_position)
        main(setup, shuffle)

def run_k_folds(setups: dict, split_position: int, k: int):
    for i in range(split_position, k):
        print(f"Running fold {i}.")
        run_all_setups(setups, i, not i == 0) #Use sorted splitting for the first fold.


def main(setup: Setup, shuffle_train_split: bool = False):
    """
    shuffle_train_split: whether shuffle the train split or use sorted by timestamp
    """
    print(setup.to_dict())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    environ = Environment(model_name=setup.model_name,
                          database_name=setup.database_name,
                          instance_dir=setup.instance_dir,
                          split_position=setup.split_position,
                          batch_size=setup.batch_size,
                          conf_calibration=setup.conf_calibration,
                          min_inter_per_user=setup.min_inter_per_user,
                          learn_to_rank=setup.learn_to_rank
                          ).read_split_datasets(shuffle_train_split)

    model, fit_dl, val_dl, test_dl = environ.get_model_dataloaders()

    if setup.fit_mode == 0 and not setup_and_model_exists(setup.instance_dir):

        model = setup_fit(setup, model, fit_dl, val_dl, environ, device)

    if setup_model_results_exists(setup.instance_dir) and not setup.reevaluate:
        print("All results already obtained. Skip.")
        return

    export_setup(environ, setup.to_dict())
    eval_df, test_df = export_elementwise_error(model, environ, device)
    conf_10_3threshold = (None, None)
    eval_metrics = evaluate(eval_df, environ, conf_10_3threshold)
    if 'conf_threshold@10' in eval_metrics:
        conf_10_3threshold = (float(eval_metrics['conf_threshold@10']), float(eval_metrics['conf_threshold@3']))
    test_metrics = evaluate(test_df, environ, conf_10_3threshold)
    
    export_metrics(environ, {"eval": eval_metrics, "test": test_metrics})

def handle_setup_instance(setup_instance_path):
    """Run a specific setup instance from a provided JSON file."""
    setup_json = read_json(setup_instance_path)
    setup = Setup(**setup_json)
    main(setup)

def handle_reevaluate(runs_folder: str):
    """Re-evaluate all setups found in the runs directory."""
    setups_uri_list = glob.glob(f"{runs_folder}/**/setup-[0-9].json")
    for setup_uri in setups_uri_list:
        with open(setup_uri, 'r') as f:
            setup_data = json.load(f)

        setup_data['instance_dir'] = setup_uri[:setup_uri.rindex("/")]
        setup_data['reevaluate'] = True
        setup = Setup(**setup_data)

        main(setup)

def handle_all_setups(setups):
    """Run all predefined setups from the setups JSON file."""
    run_all_setups(setups)

def handle_k_folds(setups, split_position, k_folds):
    """Run k-fold cross-validation for all predefined setups."""
    if k_folds <= 0:
        raise ValueError("k_folds must be greater than 0")
    run_k_folds(setups, split_position, k_folds)

def handle_single_setup(setup_data):
    """Run a single predefined setup from the setups JSON file."""
    setup = Setup(**setup_data)
    main(setup)

def handle_custom_setup(args):
    """Run a custom setup using command-line arguments."""
    setup = Setup(
        model_name=args.model_name,
        database_name=args.database_name,
        split_position=args.split_position,
        fit_mode=args.fit_mode,
        batch_size=args.batch_size,
        conf_calibration=bool(args.conf_calibration),
        learning_rate=args.learning_rate,
        learn_to_rank=bool(args.learn_to_rank)
        # Include additional parameters as needed
    )
    main(setup)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup_name", type=str, default="none",
                        help="Predefined setup name, 'all', 'k_folds', 'reevaluate', or '{database_name}-{model_name} from setups.json'")
    parser.add_argument("--k_folds", type=int, default=0,
                        help="Number of folds for cross-validation")
    parser.add_argument("--database_name", type=str, default="ml-1m")
    parser.add_argument("--fit_mode", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="mf")
    parser.add_argument("--setup_instance", type=str, default=None,
                        help="Path to a specific setup JSON file")
    parser.add_argument("--split_position", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--conf_calibration", type=int, default=0)
    parser.add_argument("--setups", type=str, default="setups.json",
                        help="Path to predefined setups JSON file")
    parser.add_argument("--runs_folder", type=str, default="./runs",
                        help="Path to the runs folder")
    parser.add_argument("--learn_to_rank", type=int, default=0,
                        help="Whether is learn to rank or rating prediction.")

    args = parser.parse_args()
    setups = read_json(args.setups)

    if args.setup_instance:
        handle_setup_instance(args.setup_instance)
    elif args.setup_name == "reevaluate":
        handle_reevaluate(args.runs_folder)
    elif args.setup_name == "all":
        handle_all_setups(setups)
    elif args.setup_name == "k_folds":
        handle_k_folds(setups, args.split_position, args.k_folds)
    elif args.setup_name in setups:
        handle_single_setup(setups[args.setup_name])
    else:
        handle_custom_setup(args)
