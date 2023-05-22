from argparse import ArgumentParser
from lib.utils import load_config, setup_seed
from easydict import EasyDict as edict
from datasets import mbes_data
from torch.utils.data import DataLoader
setup_seed(0)

def get_datasets(config: edict):
    if(config.dataset=='multibeam'):
        train_set, val_set = mbes_data.get_multibeam_train_datasets(config)
        test_set = mbes_data.get_multibeam_test_datasets(config)
    else:
        raise NotImplementedError

    return train_set, val_set, test_set

def test(config: edict):
    _, _, test_set = get_datasets(config)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)
    for i, data in enumerate(test_loader):
        print(data['matching_inds'].shape)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/mbesdata_test.yaml', help='Path to config file')
    args = parser.parse_args()
    config = load_config(args.config)
    config = edict(config)
    print(config)
    test(config)