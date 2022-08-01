from argparse import ArgumentParser

from experiments.experiment_dc_gan import conduct_experiment_dc_gan
from experiments.experiment_github_gan import conduct_experiment_github_gan
from experiments.experiment_one_dim_gan import conduct_experiment_one_dim_gan
from experiments.experiment_conditional_gan import conduct_experiment_cond_gan
from experiments.experiment_ls_gan import conduct_experiment_ls_gan

from json import loads


def _parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--ganType',
        default='dcgan',
        type=str,
        help='Type of the gan you want to learn'
    )

    parser.add_argument(
        'experiment_config',
        type=str,
        help='experiment configuration'
    )

    arguments = parser.parse_args()
    return arguments


experiment = {
    'dc_gan': conduct_experiment_dc_gan,
    'github': conduct_experiment_github_gan,
    'one_dim': conduct_experiment_one_dim_gan,
    'cond_gan': conduct_experiment_cond_gan,
    'ls_gan': conduct_experiment_ls_gan,
}

args = _parse_args()
params = loads(args.experiment_config)
experiment[args.ganType](**params)
