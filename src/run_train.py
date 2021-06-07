import argparse
from collections import namedtuple

import yaml

import common
import experiment

parser = argparse.ArgumentParser(
    description='')
parser.add_argument('--conf', metavar='conf', type=str, required=True,
    help='Path to config file')
parser.add_argument('--exp_id', type=str, required=True,
    help='Experiment ID')

cli_args = parser.parse_args()

args = {}
with open(cli_args.conf) as conf_file:
    Config = namedtuple('Config', common.OPTS)
    args = Config(**yaml.load(conf_file, Loader=yaml.FullLoader)[f'exp_{cli_args.exp_id}'])

dataset_group = common.make_dataset_group(args)
model = common.make_model(args.model_id)

print(f'Running with params: model={type(model)},dataset_group={dataset_group}, epochs={args.epochs}, learning_rate={args.learning_rate}, exp_id={cli_args.exp_id}')
experiment.run(model, dataset_group, args.epochs, args.learning_rate, cli_args.exp_id, args.iter_per_epoch, args.weight_decay)
