from argparse import ArgumentParser
from lib.utils import load_config, setup_seed
from easydict import EasyDict as edict
setup_seed(0)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/mbesdata_test.yaml', help='Path to config file')
    args = parser.parse_args()
    config = load_config(args.config)
    config = edict(config)
    print(config)