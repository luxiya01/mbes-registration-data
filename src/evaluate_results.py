import argparse
from mbes_data.lib.evaluations import compute_metrics_from_results_folder
import numpy as np
import logging
import os
from typing import List

def _get_logger(results_root: str) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    f_handler = logging.FileHandler(os.path.join(results_root, 'metrics.log'))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)
    return logger

def main(results_root: str, use_transforms: List[str]=['pred', 'null', 'gt']):
    logger = _get_logger(results_root)

    for root, dirs, files in os.walk(results_root):
        # Compute metrics for folders without further subfolders, i.e.
        # leaf folders containing the *_res.npz results files
        if len(dirs) > 0:
            continue
        logger.info('='*60)
        logger.info(f'Computing results for {root}')
        for use_transform in use_transforms:
            logger.info(f'Using transform {use_transform}')
            compute_metrics_from_results_folder(logger,
                                                results_folder=root,
                                                use_transform=use_transform,
                                                print=True)
        logger.info('='*60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root', type=str, required=True,
                        help='Root folder containing the results folders')
    parser.add_argument('--use_transforms', type=str, nargs='+',
                        default=['pred', 'null', 'gt'],
                        help='Which transforms to use for computing metrics')
    args = parser.parse_args()
    main(args.results_root, args.use_transforms)