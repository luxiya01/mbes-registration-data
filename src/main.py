from argparse import ArgumentParser
from collections import defaultdict
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
import torch
import numpy as np
import open3d as o3d
import copy
import pandas as pd
from tqdm import tqdm
import logging

from mbes_data.lib.utils import load_config, setup_seed
from mbes_data.lib.benchmark_utils import to_o3d_pcd, to_tsfm
from mbes_data.lib.evaluations import (compute_metrics, update_metrics_dict,
                             summarize_metrics, print_metrics)
from mbes_data.datasets.mbes_data import get_multibeam_datasets
setup_seed(0)

def draw_registration_results(source, target, transform_gt, transform_pred):
    source_gt_trans = copy.deepcopy(source)
    source_gt_trans.transform(transform_gt)

    source_pred_trans = copy.deepcopy(source)
    source_pred_trans.transform(transform_pred)

    # Paint source_gt_trans to yellow and source_pred_trans to blue
    source.paint_uniform_color([1., 0, 0])
    source_gt_trans.paint_uniform_color([1, 0.706, 0])
    source_pred_trans.paint_uniform_color([0, 0.651, 0.929])

    o3d.visualization.draw_geometries([source, source_gt_trans, source_pred_trans, target])

def generalized_icp(config: edict,
                    data: dict,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()):
    points_src = to_o3d_pcd(data['points_src'])
    points_ref = to_o3d_pcd(data['points_ref'])
    transform_gt = to_tsfm(data['transform_gt_rot'], data['transform_gt_trans'])

    predicted = o3d.pipelines.registration.registration_generalized_icp(
        points_src, points_ref, config.overlap_radius, np.eye(4),
        estimation_method,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))

    if config.draw_registration_results:
        draw_registration_results(points_src, points_ref, transform_gt, predicted.transformation)
    return predicted.transformation

def test(config: edict):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.info('Start testing...')

    _, _, test_set = get_multibeam_datasets(config)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)

    all_gt_metrics = {'fmr_wrt_distances': defaultdict(list),
                  'fmr_inlier_ratio': defaultdict(list),
                  'fmr_wrt_inlier_ratio': defaultdict(list),
                  'registration_rmse': [],
                  'consistency': [],
                  'std_of_mean': [],
                  'std_of_points': [],
                  'hit_by_one': [],
                  'hit_by_both': [],
                  'r_mse': [], 't_mse': [], 'r_mae': [], 't_mae': [],
                  'err_r_deg': [], 'err_t': [], 'chamfer_dist': []}
    all_pred_metrics = copy.deepcopy(all_gt_metrics)

    for _, data in tqdm(enumerate(test_loader), total=len(test_set)):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].squeeze(0)
        transform_pred = generalized_icp(config, data)
        pred_metrics = compute_metrics(data, transform_pred, resolution=config.voxel_size*2)
        all_pred_metrics = update_metrics_dict(all_pred_metrics, pred_metrics)

        transform_gt = to_tsfm(data['transform_gt_rot'], data['transform_gt_trans'])
        gt_metrics = compute_metrics(data, transform_gt, resolution=config.voxel_size*2)
        all_gt_metrics = update_metrics_dict(all_gt_metrics, gt_metrics)

    for key, val in all_gt_metrics.items():
        if isinstance(val, dict):
            for k, v in val.items():
                all_gt_metrics[key][k] = np.array(v)
                all_pred_metrics[key][k] = np.array(all_pred_metrics[key][k])
        else:
            all_gt_metrics[key] = np.array(val)
            all_pred_metrics[key] = np.array(all_pred_metrics[key])

    print('Using ground truth GT:')
    summary_all_gt_metrics = summarize_metrics(all_gt_metrics)
    print_metrics(logger, summary_all_gt_metrics)
    
    print('==========================================')
    print('Using predicted transformation:')
    summary_all_pred_metrics = summarize_metrics(all_pred_metrics)
    print_metrics(logger, summary_all_pred_metrics)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/mbesdata_test_meters.yaml', help='Path to config file')
    args = parser.parse_args()
    config = load_config(args.config)
    config = edict(config)
    print(config)
    test(config)