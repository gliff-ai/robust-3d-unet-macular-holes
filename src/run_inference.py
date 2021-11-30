import argparse
from collections import namedtuple

import yaml

import common
import inference


parser = argparse.ArgumentParser(
    description='')
parser.add_argument('--conf', metavar='conf', type=str, required=True,
    help='Path to config file')
parser.add_argument('--exp_id', type=str, required=True,
    help='Experiment ID')
parser.add_argument('--weights_filename', type=str, default='best_validation_full_size',
    help="e.g - 'best_validation'")

cli_args = parser.parse_args()

args = {}
with open(cli_args.conf) as conf_file:
    Config = namedtuple('Config', common.OPTS)
    args = Config(**yaml.load(conf_file, Loader=yaml.FullLoader)[f'exp_{cli_args.exp_id}'])

dataset_group = common.make_dataset_group(args)
model = common.make_model(args.model_id)

print(f'Running inference with params: model={type(model)},dataset_group={dataset_group},exp_id={cli_args.exp_id}')
inference.run(model, dataset_group, cli_args.exp_id, cli_args.weights_filename)
