import os
import glob
import json
import re
import time
from datetime import datetime


def setup_model_results_exists(run_folder: str):

    setup_exists = setup_and_model_exists(run_folder)

    eval_errors = glob.glob(f"{run_folder}/eval_error_conf-[0-9]*.csv")
    test_errors = glob.glob(f"{run_folder}/test_error_conf-[0-9]*.csv")
    metrics_models = glob.glob(f"{run_folder}/metrics-[0-9]*.json")

    return setup_exists and len(eval_errors) > 0 and len(test_errors) > 0 and len(metrics_models) > 0

def setup_and_model_exists(run_folder: str):

    existent_setups = glob.glob(f"{run_folder}/setup-[0-9]*.json" )
    existent_models = glob.glob(f"{run_folder}/model-[0-9]*.pth")
    return len(existent_setups) > 0 and len(existent_models) > 0

def scan_folder_for_files(root_folder: str, start_with:str= "ranking", end_with: str= ".json") -> dict:
    """
    look for files with that match {start_with}*{end_with} in subfolders of root_folder
    """
    subfolder_files = {}

    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        if os.path.isdir(subfolder_path):
            try:
                for filename in os.listdir(subfolder_path):
                    if filename.startswith(start_with) and filename.endswith(end_with):
                        filepath = os.path.join(subfolder_path, filename)
                        with open(filepath, 'r') as f:
                            target_data = json.load(f)

                        subfolder_files[subfolder] = target_data
                        break

            except (ValueError, IndexError):
                print(f"Error parsing subfolder name: {subfolder}")
                continue

    return subfolder_files


def read_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def export_metrics(environ, metrics: dict):
    metrics_uri = f"{environ.instance_dir}/metrics-{environ.split_position}.json"
    if os.path.isfile(f"{environ.instance_dir}/metrics-{environ.split_position}.json"):
        metrics_uri = f"{environ.instance_dir}/metrics-{environ.split_position}-{time.strftime('%Y-%m-%d-%H-%M-%S')}.json"
    with open(metrics_uri, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def export_setup(environ, setup: dict):
    setup['fit_mode'] = 1
    setup_uri = f"{environ.instance_dir}/setup-{environ.split_position}.json"
    if os.path.isfile(f"{environ.instance_dir}/setup-{environ.split_position}.json"):
        setup_uri = f"{environ.instance_dir}/setup-{environ.split_position}-{time.strftime('%Y-%m-%d-%H-%M-%S')}.json"

    with open(setup_uri, "w", encoding="utf-8") as f:
        json.dump(setup, f, indent=4)

def extract_datetime(path: str):
    match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', path)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d-%H-%M-%S")
    else:
        return None

def sort_paths_by_datetime(paths: list[str]):
    sorted_paths = sorted(paths, key=lambda x: (extract_datetime(x) is not None, extract_datetime(x)))
    return sorted_paths
