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

from mbes_data.common.math import so3
from mbes_data.lib.utils import load_config, setup_seed
from mbes_data.lib.benchmark_utils import to_o3d_pcd, to_tsfm
from mbes_data.lib.evaluations import (compute_metrics, update_metrics_dict,
                                       summarize_metrics, print_metrics,
                                       ALL_METRICS_TEMPLATE)
from mbes_data.datasets.mbes_data import get_multibeam_datasets
setup_seed(0)

def draw_registration_results(source, target, transform_gt, transform_pred):
    source_gt_trans = copy.deepcopy(source)
    source_gt_trans.transform(transform_gt)

    source_pred_trans = copy.deepcopy(source)
    source_pred_trans.transform(transform_pred)

    # Paint source_gt_trans to green and source_pred_trans to red
    source_gt_trans.paint_uniform_color([0, 1, 0])
    source_pred_trans.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([source, source_gt_trans, source_pred_trans, target])

def icp(config: edict,
        data: dict,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()):
    points_src = to_o3d_pcd(data['points_src'])
    points_ref = to_o3d_pcd(data['points_ref'])
    if config.icp_variant == 'icp_point_to_plane':
        points_src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=config.overlap_radius*2, max_nn=30))
        points_ref.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=config.overlap_radius*2, max_nn=30))
    transform_gt = to_tsfm(data['transform_gt_rot'], data['transform_gt_trans'])

    predicted = o3d.pipelines.registration.registration_icp(
        points_src, points_ref, config.overlap_radius, np.eye(4),
        estimation_method,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))

    if config.draw_registration_results:
        draw_registration_results(points_src, points_ref, transform_gt, predicted.transformation)
    return predicted.transformation

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

    all_gt_metrics = copy.deepcopy(ALL_METRICS_TEMPLATE)
    all_pred_metrics = copy.deepcopy(ALL_METRICS_TEMPLATE)
    all_null_metrics = copy.deepcopy(ALL_METRICS_TEMPLATE)

    for _, data in tqdm(enumerate(test_loader), total=len(test_set)):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].squeeze(0)
        if config.icp_variant == 'gicp':
            transform_pred = generalized_icp(config, data)
        elif config.icp_variant == 'icp_point_to_point':
            transform_pred = icp(config, data, estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
        elif config.icp_variant == 'icp_point_to_plane':
            transform_pred = icp(config, data, estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane())
        else:
            raise NotImplementedError(f'Unknown icp variant: {config.icp_variant}')
        pred_metrics = compute_metrics(data, transform_pred, config)
        all_pred_metrics = update_metrics_dict(all_pred_metrics, pred_metrics)

        transform_gt = to_tsfm(data['transform_gt_rot'], data['transform_gt_trans'])
        logger.info(f'gt rotation: {so3.dcm2euler(data["transform_gt_rot"].unsqueeze(0))}')
        gt_metrics = compute_metrics(data, transform_gt, config)
        all_gt_metrics = update_metrics_dict(all_gt_metrics, gt_metrics)

        null_metrics = compute_metrics(data, np.eye(4), config)
        all_null_metrics = update_metrics_dict(all_null_metrics, null_metrics)

    for key, val in all_gt_metrics.items():
        if isinstance(val, dict):
            for k, v in val.items():
                all_gt_metrics[key][k] = np.array(v)
                all_pred_metrics[key][k] = np.array(all_pred_metrics[key][k])
                all_null_metrics[key][k] = np.array(all_null_metrics[key][k])
        else:
            all_gt_metrics[key] = np.array(val)
            all_pred_metrics[key] = np.array(all_pred_metrics[key])
            all_null_metrics[key] = np.array(all_null_metrics[key])

    print('Using ground truth GT:')
    summary_all_gt_metrics = summarize_metrics(all_gt_metrics)
    print_metrics(logger, summary_all_gt_metrics)
    
    print('==========================================')
    print('Using predicted transformation:')
    summary_all_pred_metrics = summarize_metrics(all_pred_metrics)
    print_metrics(logger, summary_all_pred_metrics)

    print('==========================================')
    print('Using null transformation:')
    summary_all_null_metrics = summarize_metrics(all_null_metrics)
    print_metrics(logger, summary_all_null_metrics)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mbes_config', type=str, default='mbes_data/configs/mbesdata_test_meters.yaml', help='Path to config file')
    parser.add_argument('--network_config', type=str, default='network_config.yaml', help='Path to config file')
    args = parser.parse_args()
    mbes_config = edict(load_config(args.mbes_config))
    network_config = edict(load_config(args.network_config))
    config = copy.deepcopy(mbes_config)
    for k, v in network_config.items():
        if k not in config:
            config[k] = v
    print(config)
    test(config)
