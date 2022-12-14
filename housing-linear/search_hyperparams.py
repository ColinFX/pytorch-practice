import argparse
import os
import sys
from subprocess import check_call

import utils


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/usa_housing", help="")
parser.add_argument("--parent_dir", default="experiments/learning_rate", help="")


def launch_training_job(parent_dir: str, data_dir: str, job_name: str, params: utils.Params):
    """
    Launch the training-evaluation of the model with a set of hyperparameters params and save the configuration in 
    `parent_dir/job_name/params.json`. 

    Args:
        * parent_dir: (str) parent directory of a set of folders containing config files
        * data_dir: (str) directory of the dataset
        * job_name: (str) unique name of one configuration of hyperparameters as one job
        * params: (utils.Params) hyperparameters
    """

    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    json_path = os.path.join(model_dir, "params.json")
    params.save(json_path)

    # launch training with this config
    cmd = "{} train.py --model_dir {} --data_dir {}".format(PYTHON, model_dir, data_dir) 
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    """Launch several jobs with different configuration of hyperparameters"""

    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, "params.json")    # default configuration of hyperparameters
    assert os.path.isfile(json_path), "No json file found at {}.".format(json_path)
    params = utils.Params(json_path)

    # search over one hyperparameter and overwrite default configuration
    learning_rates = [1e-8, 1e-7, 1e-6, 1e-5]
    for learning_rate in learning_rates:
        params.learning_rate = learning_rate
        job_name = "learning_rate_{}".format(learning_rate)
        launch_training_job(args.parent_dir, args.data_dir, job_name, params)
