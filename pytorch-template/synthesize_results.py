"""Aggregate results from all experiments of hypermarameter search"""

import argparse
import os

import csv
import json


parser = argparse.ArgumentParser()
parser.add_argument("--parent_dir", default="experiments", help="")


def aggregate_metrics(parent_dir: str, metrics: dict[str, dict[str, float]]):  
    """
    Aggregate metrics from all subfolders of experiments by recursion. The structure of parent_dir should be:

    ```
    parent_dir/
        base_model/
            metrics_val_best_weights.json
        search_hyperparameter_1/
            job_name_1/
                metrics_val_best_weights.json
            job_name_2/
                metrics_val_best_weights.json
            ...
    ```
    
    Args:
        * parent_dir: (str) directory containing ALL subfolders of experiments
        * metrics: (dict) sub_dir -> {metric_name -> metric_value}
    """          
    
    # check whether the parent_dir itself contains results
    metrics_path = os.path.join(parent_dir, "metrics_val_best_weights.json")
    if os.path.isfile(metrics_path):
        with open(metrics_path, 'r') as json_file:
            metrics[parent_dir] = json.load(json_file)
    
    # check every subfolder by recursion
    for sub_dir in os.listdir(parent_dir):
        metrics_folder = os.path.join(parent_dir, sub_dir)
        if os.path.isdir(metrics_folder):
            aggregate_metrics(metrics_folder, metrics)
    

def metrics_to_csv(parent_dir: str, metrics: dict[str, dict[str, float]]):
    """
    Export double dictionary metrics to csv file under parent_dir. 

    Args:
        * parent_dir: (str) directory containing ALL subfolders of experiments
        * metrics: (dict) sub_dir -> {metric_name -> metric_value}
    """

    csv_path = os.path.join(parent_dir, "results.csv")
    with open(csv_path, "wb") as csv_file:
        header = metrics[parent_dir].keys()
        csv_writer = csv.DictWriter(csv_file, header)
        csv_writer.writeheader(header)
        for sub_dir in metrics:
            csv_writer.writerow({metric: metrics[sub_dir][metric] for metric in header})


if __name__ == "__main__":
    args = parser.parse_args()
    metrics: dict[str, dict[str, float]] = {}
    aggregate_metrics(args.parent_dir, metrics)
    metrics_to_csv(args.parent_dir, metrics)
