from argparse import ArgumentParser
from collections import defaultdict
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
import torch
import numpy as np
import open3d as o3d
import copy
from tqdm import tqdm
import logging
import os

from mbes_data.lib.utils import load_config, setup_seed
from mbes_data.lib.benchmark_utils import to_o3d_pcd, to_tsfm, ransac_pose_estimation
from mbes_data.datasets.mbes_data import get_multibeam_datasets
from mbes_data.lib.evaluations import save_results_to_file, update_results
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
    if config.method == 'icp_point_to_plane':
        points_src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=config.overlap_radius*2, max_nn=30))
        points_ref.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=config.overlap_radius*2, max_nn=30))
    transform_gt = to_tsfm(data['transform_gt_rot'], data['transform_gt_trans'])

    predicted = o3d.pipelines.registration.registration_icp(
        points_src, points_ref, config.overlap_radius, np.eye(4),
        estimation_method,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=config.max_iteration))

    if config.draw_registration_results:
        draw_registration_results(points_src, points_ref, transform_gt, predicted.transformation)
    return predicted

def generalized_icp(config: edict,
                    data: dict,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()):
    points_src = to_o3d_pcd(data['points_src'])
    points_ref = to_o3d_pcd(data['points_ref'])
    transform_gt = to_tsfm(data['transform_gt_rot'], data['transform_gt_trans'])

    predicted = o3d.pipelines.registration.registration_generalized_icp(
        points_src, points_ref, config.overlap_radius, np.eye(4),
        estimation_method,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=config.max_iteration))

    if config.draw_registration_results:
        draw_registration_results(points_src, points_ref, transform_gt, predicted.transformation)
    return predicted

def registration_with_fpfh(config: edict,
                           data: dict):
    points_src = to_o3d_pcd(data['points_src'])
    points_ref = to_o3d_pcd(data['points_ref'])
    transform_gt = to_tsfm(data['transform_gt_rot'], data['transform_gt_trans'])

    radius_normal = config.overlap_radius * 2
    points_src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=config.overlap_radius*2, max_nn=30))
    points_ref.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=config.overlap_radius*2, max_nn=30))

    radius_feature = config.feature_radius
    fpfh_src = o3d.pipelines.registration.compute_fpfh_feature(
        points_src,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    fpfh_ref = o3d.pipelines.registration.compute_fpfh_feature(
        points_ref,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    predicted = ransac_pose_estimation(points_src, points_ref, fpfh_src, fpfh_ref,
                                       distance_threshold=config.voxel_size*1.5,
                                       ransac_n=config.ransac_n,
                                       ransac_iterations=config.ransac_iterations,)
    if config.draw_registration_results:
        draw_registration_results(points_src, points_ref, transform_gt, predicted.transformation)

    extras = {'feat_src_points': np.array(points_src.points),
              'feat_ref_points': np.array(points_ref.points),
              'feat_src': fpfh_src.data.T,
              'feat_ref': fpfh_ref.data.T}
    return predicted, extras


def test(config: edict):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.info('Start testing...')

    _, _, test_set = get_multibeam_datasets(config)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)

    results = defaultdict(dict)
    outdir = os.path.join(config.exp_dir, config.method)
    os.makedirs(outdir, exist_ok=True)
    for i, data in tqdm(enumerate(test_loader), total=len(test_set)):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].squeeze(0)
        if config.method == 'fpfh':
            prediction, extras = registration_with_fpfh(config, data)
            data.update(extras)
        elif config.method == 'gicp':
            prediction = generalized_icp(config, data)
        elif config.method == 'icp_point_to_point':
            prediction = icp(config, data, estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
        elif config.method == 'icp_point_to_plane':
            prediction = icp(config, data, estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane())
        else:
            raise NotImplementedError(f'Unknown method: {config.method}')

        data['success'] = len(prediction.correspondence_set) > 0

        transform_pred = prediction.transformation
        data['success'] = not (np.allclose(transform_pred,
                                      np.eye(4)))
        results = update_results(results, data, transform_pred,
                                 config, outdir, logger)
    # save results of the last MBES file
    save_results_to_file(logger, results, config, outdir)


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
    os.makedirs(config.exp_dir, exist_ok=True)
    config.results_path = f'{config.exp_dir}/results.npz'
    print(config)
    test(config)
