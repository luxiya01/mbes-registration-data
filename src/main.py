from argparse import ArgumentParser
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
import torch
import numpy as np
import open3d as o3d
import copy
import pandas as pd
from tqdm import tqdm

from lib.utils import load_config, setup_seed
from lib.benchmark_utils import to_o3d_pcd, to_tsfm
from lib.evaluations import fmr_wrt_distances, fmr_wrt_inlier_ratio, get_predicted_correspondence_gt_distance
from datasets import mbes_data
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

    gt_all_distances = []
    pred_all_distances = []

    for _, data in tqdm(enumerate(test_loader), total=len(test_set)):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].squeeze(0)
        transform_pred = generalized_icp(config, data)
        pred_distances = get_predicted_correspondence_gt_distance(data, transform_pred)
        pred_all_distances.append(pred_distances)

        transform_gt = to_tsfm(data['transform_gt_rot'], data['transform_gt_trans'])
        gt_distances = get_predicted_correspondence_gt_distance(data, transform_gt)
        gt_all_distances.append(gt_distances)

    print('Using ground truth GT:')
    fmr_wrt_distances(gt_all_distances)
    fmr_wrt_inlier_ratio(gt_all_distances)
    print('==========================================')
    print('Using predicted transformation:')
    fmr_wrt_distances(pred_all_distances)
    fmr_wrt_inlier_ratio(pred_all_distances)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/mbesdata_test_meters.yaml', help='Path to config file')
    args = parser.parse_args()
    config = load_config(args.config)
    config = edict(config)
    print(config)
    test(config)