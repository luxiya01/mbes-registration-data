from argparse import ArgumentParser
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
import numpy as np
import open3d as o3d
import copy
import pandas as pd

from lib.utils import load_config, setup_seed
from lib.benchmark_utils import to_o3d_pcd, to_tsfm
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
    points_src = to_o3d_pcd(data['points_src'].squeeze(0))
    points_ref = to_o3d_pcd(data['points_ref'].squeeze(0))
    transform_gt = to_tsfm(data['transform_gt_rot'].squeeze(0), data['transform_gt_trans'].squeeze(0))

    predicted = o3d.pipelines.registration.registration_generalized_icp(
        points_src, points_ref, config.overlap_radius, np.eye(4),
        estimation_method,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))

    eval_gt = o3d.pipelines.registration.evaluate_registration(points_src, points_ref, config.overlap_radius, transform_gt)
    eval_pred = o3d.pipelines.registration.evaluate_registration(points_src, points_ref, config.overlap_radius, predicted.transformation)
    print(f'Ground truth: {eval_gt}\nPredicted: {eval_pred}')
    draw_registration_results(points_src, points_ref, transform_gt, predicted.transformation)
    return {'gt': eval_gt, 'pred': eval_pred}


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

    gt_fitness, gt_inlier_rmse, gt_corr = [], [], []
    pred_fitness, pred_inlier_rmse, pred_corr = [], [], []

    for i, data in enumerate(test_loader):
        res = generalized_icp(config, data)
        gt_fitness.append(res['gt'].fitness)
        gt_inlier_rmse.append(res['gt'].inlier_rmse)
        gt_corr.append(np.array(res['gt'].correspondence_set).shape[0])
        pred_fitness.append(res['pred'].fitness)
        pred_inlier_rmse.append(res['pred'].inlier_rmse)
        pred_corr.append(np.array(res['pred'].correspondence_set).shape[0])
        #print(data['matching_inds'].shape)
    df = pd.DataFrame.from_dict({'gt_fitness': gt_fitness, 'gt_inlier_rmse': gt_inlier_rmse, 'gt_corr': gt_corr,
                    'pred_fitness': pred_fitness, 'pred_inlier_rmse': pred_inlier_rmse, 'pred_corr': pred_corr})
    df.to_csv(f'{config.noise_type}_{config.subset_test}_gicp_res.csv', index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/mbesdata_test_meters.yaml', help='Path to config file')
    args = parser.parse_args()
    config = load_config(args.config)
    config = edict(config)
    print(config)
    test(config)