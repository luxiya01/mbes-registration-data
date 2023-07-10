from collections import defaultdict
import json
from typing import List
import numpy as np
import open3d as o3d
from mbes_data.lib.benchmark_utils import to_o3d_pcd, to_tsfm, to_array, to_tensor
import torch
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import copy
import logging
from tqdm import tqdm

from easydict import EasyDict as edict

from mbes_data.common.math_torch import se3
from mbes_data.common.math.so3 import dcm2euler
import os


def update_results(results: dict, data: dict, transform_pred: torch.Tensor, config: edict, outdir: str,
                   logger: logging.Logger):
    """Helper function to update the results dict with the new data and predicted transform.
    If the length of the results dict exceeds the threshold given in config, then we store the
    existing results dict under patches-{idxmin}-{idxmax}.npz in outdir, and create a new results dict with
    the new data and predicted transform. This is to avoid memory overflow."""

    if len(results) >= config.results_length:
        save_results_to_file(logger, results, config, outdir)
        results = defaultdict(dict)

    idx = int(data['idx'])
    results[idx] = {
        'idx': data['idx'],
        'src_idx': data['src_idx'],
        'ref_idx': data['ref_idx'],
        'points_src': to_array(data['points_src']),
        'points_ref': to_array(data['points_ref']),
        'transform_gt': to_array(data['transform_gt']),
        'transform_gt_rot': to_array(data['transform_gt_rot']),
        'transform_gt_trans': to_array(data['transform_gt_trans']),
        'transform_pred': to_array(transform_pred),
        'success': data['success']
    }
    return results

def save_results_to_file(logger: logging.Logger, results: dict, config: edict, outdir: str):
    """ Save the results to file, separated by data filenames."""
    pair_indices = set(results.keys())
    min_idx = min(pair_indices)
    max_idx = max(pair_indices)

    logger.info(f'Saving results for pairs ({min_idx}, {max_idx}) to {outdir}')
    np.savez(f'{outdir}/pairs_{min_idx}_to_{max_idx}_res.npz',
                results=results,
                config=config,
                allow_pickle=True)

def compute_metrics_from_results_folder(logger: logging.Logger, results_folder: str, use_transform: str = 'null', print: bool = True):
    """ Compute metrics from the results files and log to logger."""
    if use_transform not in ['null', 'pred', 'gt']:
        raise ValueError('Invalid use_transform arg: {}. Supports [null, pred, gt]'.format(use_transform))
    logger.info('Computing metrics using {} tranform from results folder: {}'
                 .format(use_transform, results_folder))

    all_metrics = copy.deepcopy(ALL_METRICS_TEMPLATE)
    config = None
    results_files = sorted([x for x in os.listdir(results_folder) if x.endswith('.npz')
                            and 'metrics' not in x])
    for results_file in results_files:
        logger.info('Loading results from {}'.format(results_file))
        file_content = np.load(os.path.join(results_folder, results_file), allow_pickle=True)
        results = file_content['results'].item()
        if not config:
            config = file_content['config'].item()

        for _, data in tqdm(results.items(), total=len(results)):
            # Do not include unsuccessful predictions into metrics computation for pred transform,
            # but add whether the prediction is successful to the metrics dict
            all_metrics['success'].append(data['success'])
            if not data['success'] and use_transform == 'pred':
                continue
            if use_transform == 'pred':
                pred_transform = data['transform_pred']
            elif use_transform == 'gt':
                pred_transform = to_tsfm(data['transform_gt_rot'], data['transform_gt_trans'])
            else:
                pred_transform = np.eye(4) # null transform
            data_metric = compute_metrics(data, pred_transform, config)
            all_metrics = update_metrics_dict(all_metrics, data_metric)
    summary = summarize_metrics(all_metrics)
    if print:
        print_metrics(logger, summary, title=f'{use_transform.upper()} Metrics')

    summary_and_all_metrics = {'all_metrics': all_metrics, 'summary': summary}
    np.savez(os.path.join(results_folder, f'{use_transform}_metrics.npz'),
                **summary_and_all_metrics)
    return summary_and_all_metrics

#========================================================================================
# Below: Point cloud registration metrics computation
#========================================================================================

ALL_METRICS_TEMPLATE = {
  'fmr_wrt_distances': defaultdict(list),
  'fmr_inlier_ratio': defaultdict(list),
  'fmr_wrt_inlier_ratio': defaultdict(list),
  'registration_mse': [],
  'consistency': [],
  'std_of_mean': [],
  'std_of_points': [],
  'hit_by_one': [],
  'hit_by_both': [],
  'r_mse': [],
  't_mse': [],
  'r_mae': [],
  't_mae': [],
  'err_r_deg': [],
  'err_t': [],
  'success': []
}

def _construct_kd_tree_from_xy_values(points: np.ndarray,
                                      transform: np.ndarray = None) -> tuple:
    """ Construct a KD tree from the x, y values of the points.
    Optionally transform the points using the provided transform."""
    pcd = to_o3d_pcd(points)
    if transform is not None:
        pcd.transform(transform)
    transformed_points = np.array(pcd.points)
    return transformed_points, KDTree(transformed_points[:, :2])

def _construct_query_tree(min_x: float, min_y: float,
                          num_rows: float, num_cols: float,
                          resolution: float):
    x = np.linspace(min_x + (0.5 * resolution),
                    min_x + (num_rows - 1 + 0.5) * resolution, num_rows)
    y = np.linspace(min_y + (0.5 * resolution),
                    min_y + (num_cols - 1 + 0.5) * resolution, num_cols)
    xv, yv = np.meshgrid(x, y)
    queries = np.stack((xv.flatten(), yv.flatten()), axis=-1)
    queries_tree = KDTree(queries)
    return queries, queries_tree

def compute_consistency_metrics(data: dict,
                                transform_pred: np.array,
                                resolution: float = 1) -> np.ndarray:
    """ Compute the consistency metrics at a specified resolution."""
    src_points, src_tree = _construct_kd_tree_from_xy_values(
        points=data['points_src'], transform=transform_pred)
    ref_points, ref_tree = _construct_kd_tree_from_xy_values(
        points=data['points_ref'], transform=None)

    # Get range of x and y
    max_x = max(np.max(src_points[:, 0]), np.max(ref_points[:, 0]))
    min_x = min(np.min(src_points[:, 0]), np.min(ref_points[:, 0]))
    max_y = max(np.max(src_points[:, 1]), np.max(ref_points[:, 1]))
    min_y = min(np.min(src_points[:, 1]), np.min(ref_points[:, 1]))

    num_rows = int((max_x - min_x) / resolution) + 1
    num_cols = int((max_y - min_y) / resolution) + 1

    consistency_metric = np.zeros((num_rows, num_cols))
    std_of_mean_metric = np.zeros((num_rows, num_cols))
    std_of_points_metric = np.zeros((num_rows, num_cols))
    hit_by_both = np.zeros((num_rows, num_cols))
    hit_by_one = np.zeros((num_rows, num_cols))

    queries, queries_tree = _construct_query_tree(min_x=min_x, min_y=min_y,
                                                  num_rows=num_rows, num_cols=num_cols,
                                                  resolution=resolution)

    std_src = queries_tree.query_ball_tree(src_tree, resolution * .5)
    std_ref = queries_tree.query_ball_tree(ref_tree, resolution * .5)

    consistency_src = queries_tree.query_ball_tree(src_tree, resolution * 1.5)
    consistency_ref = queries_tree.query_ball_tree(ref_tree, resolution * 1.5)

    for i, query in enumerate(queries):
        row = int((query[0] - min_x) / resolution)
        col = int((query[1] - min_y) / resolution)
        hits_consistency_src = consistency_src[i]
        hits_consistency_ref = consistency_ref[i]
        std_src_idx = std_src[i]
        std_ref_idx = std_ref[i]

        maxmin_dist_src_to_ref, maxmin_dist_ref_to_src = 0, 0
        if len(hits_consistency_src) > 0 and len(std_ref_idx) > 0:
            maxmin_dist_ref_to_src = np.min(cdist(ref_points[std_ref_idx], src_points[hits_consistency_src]), axis=1).max()
        if len(hits_consistency_ref) > 0 and len(std_src_idx) > 0:
            maxmin_dist_src_to_ref = np.min(cdist(src_points[std_src_idx], ref_points[hits_consistency_ref]), axis=1).max()
        consistency_metric[row, col] = np.max([maxmin_dist_src_to_ref,
                                               maxmin_dist_ref_to_src])


        if len(std_src_idx) == 0 and len(std_ref_idx) == 0:
            continue
        std_of_points_metric[row, col] = np.std(
            np.concatenate((src_points[std_src_idx, 2], ref_points[std_ref_idx,
                                                                   2])))
        if len(std_src_idx) > 0 and len(std_ref_idx) > 0:
            std_of_mean_metric[row, col] = np.std([
                np.mean(src_points[std_src_idx, 2]),
                np.mean(ref_points[std_ref_idx, 2])
            ])
        hit_by_both[row, col] = len(std_src_idx) > 0 and len(std_ref_idx) > 0
        hit_by_one[row, col] = len(std_src_idx) > 0 or len(std_ref_idx) > 0

    mean_of_grid_with_values = lambda grid: np.mean(grid[grid > 0]) if np.sum(
        grid > 0) > 0 else 0

    return {
        'consistency': mean_of_grid_with_values(consistency_metric),
        'std_of_mean': mean_of_grid_with_values(std_of_mean_metric),
        'std_of_points': mean_of_grid_with_values(std_of_points_metric),
        'hit_by_both': hit_by_both.mean(),
        'hit_by_one': hit_by_one.mean()
    }


def get_mutual_nearest_neighbor(points_src, points_ref, trans):
    points_src.transform(trans)
    pcd_tree_src = o3d.geometry.KDTreeFlann(points_src)
    pcd_tree_ref = o3d.geometry.KDTreeFlann(points_ref)

    correspondences = []
    for i, point_ref in enumerate(points_ref.points):
        # Get the nearest neighbor of the reference point in the source point cloud
        [_, idx_src,
         dist_src] = pcd_tree_src.search_knn_vector_3d(point_ref, 1)
        # Get the nearest neighbor of points_src[idx_src] in the reference point cloud
        [_, idx_ref, dist_ref
         ] = pcd_tree_ref.search_knn_vector_3d(points_src.points[idx_src[0]],
                                               1)
        # Check if they are mutual nearest neighbors
        if i == idx_ref[0]:
            correspondences.append([idx_src[0], idx_ref[0]])
    return np.array(correspondences)


def get_predicted_correspondence_gt_distance(
        data: dict, transform_pred: np.ndarray) -> np.ndarray:
    """ Compute the mutual nearest neighbor correspondences under the predicted transformation,
    then compute the Euclidean distance between the GT-transformed source points and the
    reference points. The returned distances are used for feature match recall evaluation."""

    # Compute correspondences under predicted transformation
    # corr_pred.shape = [num_cor, 2]
    # corr_pred[i] = (idx_src, idx_ref)
    corr_pred = get_mutual_nearest_neighbor(to_o3d_pcd(data['points_src']),
                                            to_o3d_pcd(data['points_ref']),
                                            transform_pred)

    # Compute distances under ground truth transformation
    points_ref = to_o3d_pcd(data['points_ref'])
    points_src_gt_trans = to_o3d_pcd(data['points_src'])
    gt_trans = to_tsfm(data['transform_gt_rot'], data['transform_gt_trans'])
    points_src_gt_trans.transform(gt_trans)

    corr_points_src_gt_trans = np.array(
        points_src_gt_trans.points)[corr_pred[:, 0]]
    corr_points_ref = np.array(points_ref.points)[corr_pred[:, 1]]

    # Compute Euclidean distance between GT_t((corr[src])) and corr[ref]
    distances = np.linalg.norm(corr_points_src_gt_trans - corr_points_ref,
                               axis=1)

    return distances


def fmr_wrt_distances(distances: np.ndarray, inlier_ratio_thresh=0.05) -> dict:
    """
    Args:
        distances: Euclidean distances between 1 pair of GT-transformed source points and reference points
        inlier_ratio_thresh: Inlier ratio threshold
                             default = 5%
    """
    fmr_wrt_distances = defaultdict(int)
    fmr_inlier_ratios = defaultdict(int)
    for distance_threshold in range(
            0, 21):  # from 0.0 to 2.0 meters with 0.1 m step
        distance_threshold = distance_threshold / 10.
        inlier_percentage = (distances < distance_threshold).mean()
        success = inlier_percentage > inlier_ratio_thresh

        fmr_inlier_ratios[distance_threshold] = inlier_percentage
        fmr_wrt_distances[distance_threshold] = success

    return {
        'fmr_wrt_distances': fmr_wrt_distances,
        'fmr_inlier_ratios': fmr_inlier_ratios
    }


def fmr_wrt_inlier_ratio(distances: np.ndarray, distance_threshold=1.) -> dict:
    """
    Args:
        distances: Euclidean distances between 1 pair of GT-transformed source points and reference points
                    for all pairs of point clouds in the dataset
        distance_thresh: Inlier distance threshold
                         default = 1.m
    """
    fmr_wrt_inlier_ratio = defaultdict(int)
    for inlier_ratio in range(0, 21):  # 0% to 20% with 1% step
        inlier_ratio = inlier_ratio / 100.
        inlier_percentage = (distances < distance_threshold).mean()
        success = inlier_percentage > inlier_ratio
        fmr_wrt_inlier_ratio[inlier_ratio] = success

    #print(f'FMR wrt inlier ratios @{distance_threshold}m:\n {fmr_wrt_inlier_ratio}')
    return fmr_wrt_inlier_ratio


def registration_mse(data: dict, transform_pred: np.ndarray) -> float:
    gt_trans = to_tsfm(data['transform_gt_rot'], data['transform_gt_trans'])
    gt_corr = get_mutual_nearest_neighbor(to_o3d_pcd(data['points_src']),
                                          to_o3d_pcd(data['points_ref']),
                                          gt_trans)

    # Compute distances under ground truth transformation
    points_ref = to_o3d_pcd(data['points_ref'])
    points_src_pred_trans = to_o3d_pcd(data['points_src'])
    points_src_pred_trans.transform(transform_pred)

    corr_points_src_pred_trans = np.array(
        points_src_pred_trans.points)[gt_corr[:, 0]]
    corr_points_ref = np.array(points_ref.points)[gt_corr[:, 1]]

    mse = np.mean((corr_points_src_pred_trans - corr_points_ref)**2)
    return mse


def compute_recall_metrics(data: dict, transform_pred: np.ndarray) -> dict:
    """ Compute the recall metrics for the predicted transformation. """
    distances = get_predicted_correspondence_gt_distance(data, transform_pred)
    recall_metrics = {}
    fmr_wrt_distance = fmr_wrt_distances(distances)
    recall_metrics['fmr_wrt_distances'] = fmr_wrt_distance['fmr_wrt_distances']
    recall_metrics['fmr_inlier_ratio'] = fmr_wrt_distance['fmr_inlier_ratios']
    recall_metrics['fmr_wrt_inlier_ratio'] = fmr_wrt_inlier_ratio(distances)
    recall_metrics['registration_mse'] = registration_mse(data, transform_pred)
    return recall_metrics


def compute_metrics(data: dict, transform_pred: np.ndarray,
                    config: edict, has_scaled: bool = False) -> dict:
    """ Compute the metrics for the predicted transformation,
        including the recall metrics, the registration MSE and the
        metrics included in OverlapPredator.
    """
    # scale data back to meter scales for metrics computation
    # if config.scale = True (i.e. data was scaled into [-1, 1])
    if config.scale:
        scale = data['labels']['max_dist']
        if isinstance(scale, torch.Tensor):
            scale = scale.float().numpy()

        # has_scaled is only used for RPMNet that feeds in multiple transforms...
        if not has_scaled:
            # scale points
            for k in ['points_src', 'points_ref']:
                data[k] *= scale
            # scale translations in GT transform
            data['transform_gt_trans'] *= scale
            data['transform_gt'][:, 3][:, None] = data['transform_gt_trans']

        # scale translations in predicted transform
        transform_pred = np.array(transform_pred)
        transform_pred[:3, 3] *= scale

    recall = compute_recall_metrics(data, transform_pred)
    predator_metrics = compute_overlap_predator_metrics(data, transform_pred)

    resolution = config.voxel_size*2
    consistency_metrics = compute_consistency_metrics(data,
                                                      transform_pred,
                                                      resolution=resolution)
    return {**recall, **predator_metrics, **consistency_metrics}


def update_metrics_dict(metrics_dict: dict, new_metrics: dict) -> dict:
    """ Update the metrics dictionary with the new metrics. """
    for k, v in new_metrics.items():
        if isinstance(v, dict):
            metrics_dict[k] = update_metrics_dict(metrics_dict[k], v)
        elif isinstance(v, np.ndarray):
            metrics_dict[k].extend(v.tolist())
        else:
            metrics_dict[k].append(v)
    return metrics_dict


#========================================================================================
# Below: compute metrics functions from OverlapPredator
#========================================================================================
def compute_overlap_predator_metrics(data, pred_transforms):
    """
    Compute metrics required in the paper (from OverlapPredator)
    """

    def square_distance(src, dst):
        return torch.sum((src[:, :, None, :] - dst[:, None, :, :])**2, dim=-1)

    with torch.no_grad():
        pred_transforms = torch.from_numpy(pred_transforms).float().unsqueeze(
            0)
        gt_transforms = to_tensor(data['transform_gt']).unsqueeze(0)
        points_src = to_tensor(data['points_src'][..., :3]).unsqueeze(0)
        points_ref = to_tensor(data['points_ref'][..., :3]).unsqueeze(0)

        # Euler angles, Individual translation errors (Deep Closest Point convention)
        # TODO Change rotation to torch operations
        r_gt_euler_deg = dcm2euler(gt_transforms[:, :3, :3].numpy(), seq='xyz')
        r_pred_euler_deg = dcm2euler(pred_transforms[:, :3, :3].numpy(),
                                     seq='xyz')
        t_gt = gt_transforms[:, :3, 3]
        t_pred = pred_transforms[:, :3, 3]
        r_mse = np.mean((r_gt_euler_deg - r_pred_euler_deg)**2, axis=1)
        r_mae = np.mean(np.abs(r_gt_euler_deg - r_pred_euler_deg), axis=1)
        t_mse = torch.mean((t_gt - t_pred)**2, dim=1)
        t_mae = torch.mean(torch.abs(t_gt - t_pred), dim=1)

        # Rotation, translation errors (isotropic, i.e. doesn't depend on error
        # direction, which is more representative of the actual error)
        concatenated = se3.concatenate(se3.inverse(gt_transforms),
                                       pred_transforms)
        rot_trace = concatenated[:, 0,
                                 0] + concatenated[:, 1,
                                                   1] + concatenated[:, 2, 2]
        residual_rotdeg = torch.acos(
            torch.clamp(0.5 *
                        (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
        residual_transmag = concatenated[:, :, 3].norm(dim=-1)

        metrics = {
            'r_mse': r_mse,
            'r_mae': r_mae,
            't_mse': to_array(t_mse),
            't_mae': to_array(t_mae),
            'err_r_deg': to_array(residual_rotdeg),
            'err_t': to_array(residual_transmag)
        }

    return metrics


def print_metrics(logger,
                  summary_metrics,
                  losses_by_iteration=None,
                  title='Metrics'):
    """Prints out formated metrics to logger (From OverlapPredator)"""

    logger.info('=' * 60)
    logger.info(title + ':')
    logger.info('=' * (len(title) + 1))

    if losses_by_iteration is not None:
        losses_all_str = ' | '.join(
            ['{:.5f}'.format(c) for c in losses_by_iteration])
        logger.info('Losses by iteration: {}'.format(losses_all_str))

    logger.info('----------------')
    logger.info(
        'DeepCP metrics:{:.4f}(rot-rmse) | {:.4f}(rot-mae) | {:.4g}(trans-rmse) | {:.4g}(trans-mae)'
        .format(
            summary_metrics['r_rmse'],
            summary_metrics['r_mae'],
            summary_metrics['t_rmse'],
            summary_metrics['t_mae'],
        ))
    logger.info('Rotation error {:.4f}(deg, mean) | {:.4f}(deg, rmse)'.format(
        summary_metrics['err_r_deg_mean'], summary_metrics['err_r_deg_rmse']))
    logger.info('Translation error {:.4g}(mean) | {:.4g}(rmse)'.format(
        summary_metrics['err_t_mean'], summary_metrics['err_t_rmse']))

    # Log registration RMSE and % pairs with <= x meters RMSE
    logger.info('----------------')
    logger.info('Registration RMSE: {:.4f}(meters)'.format(
        summary_metrics['registration_rmse']))
    logger.info('%pairs with <= x meters RMSE|{}'.format(' | '.join(
        ['{:.2f}m'.format(c) for c in np.arange(0, 2, 0.1)])))
    logger.info('values                      |{}'.format(' | '.join([
        '{:.2f}%'.format(
            np.mean(summary_metrics['registration_rmse_per_pointcloud'] < c) * 100)
        for c in np.arange(0, 2, 0.1)
    ])))

    # Log FMR wrt distances (meters)
    logger.info('----------------')
    logger.info('FMR wrt distances thresholds (m)|{}'.format(' | '.join([
        '{:.2f}m'.format(c)
        for c in summary_metrics['fmr_wrt_distances'].keys()
    ])))
    logger.info('FMR values                      |{}'.format(' | '.join([
        '{:.2f}%'.format(c * 100)
        for c in summary_metrics['fmr_wrt_distances'].values()
    ])))
    logger.info('Inlier ratio                    |{}'.format(' | '.join([
        '{:.2f}%'.format(c * 100)
        for c in summary_metrics['fmr_inlier_ratio'].values()
    ])))

    # Log FMR wrt inlier ratio
    logger.info('----------------')
    logger.info('Inlier ratio thresholds (%)|{}'.format(' | '.join([
        '{:.2f}%'.format(c * 100)
        for c in summary_metrics['fmr_wrt_inlier_ratio'].keys()
    ])))
    logger.info('FMR wrt inlier ratio       |{}'.format(' | '.join([
        '{:.2f}%'.format(c * 100)
        for c in summary_metrics['fmr_wrt_inlier_ratio'].values()
    ])))

    # Log consistency metrics
    logger.info('----------------')
    logger.info('Consistency metrics: {:.4f}'.format(
        summary_metrics['consistency']))
    logger.info('std of mean(z) values: {:.4f}'.format(
        summary_metrics['std_of_mean']))
    logger.info('std of points(z) values: {:.4f}'.format(
        summary_metrics['std_of_points']))
    logger.info('grids hit by both: {:.2f}%'.format(
        summary_metrics['hit_by_both'] * 100))
    logger.info('grids hit by at least one: {:.2f}%'.format(
        summary_metrics['hit_by_one'] * 100))
    logger.info(
        'perc.overlapping grids (hit by both/hit by one): {:.2f}%'.format(
            summary_metrics['hit_by_both'] / summary_metrics['hit_by_one'] *
            100))
    logger.info('----------------')

    # Log success rate
    logger.info(
        'Success rate: {:.2f}%'.format(summary_metrics['success'] * 100))
    logger.info('END OF METRICS LOGGING!')
    logger.info('=' * 60)


def summarize_metrics(metrics):
    """Summaries computed metrices by taking mean over all data instances (From OverlapPredator)"""

    # Convert everything to numpy arrays
    for key, value in metrics.items():
        if isinstance(value, dict):
            for k, v in value.items():
                metrics[key][k] = np.array(v)
        else:
            metrics[key] = np.array(value)

    summarized = {}
    for k in metrics:
        if k == 'registration_mse':
            summarized['registration_rmse_per_pointcloud'] = np.sqrt(metrics[k])
            summarized['registration_rmse'] = np.sqrt(np.mean(metrics[k]))

        elif k.endswith('mse'):
            summarized[k[:-3] + 'rmse'] = np.sqrt(np.mean(metrics[k]))
        elif k.startswith('err'):
            summarized[k + '_mean'] = np.mean(metrics[k])
            summarized[k + '_rmse'] = np.sqrt(np.mean(metrics[k]**2))
        elif k.startswith('fmr'):
            if k not in summarized:
                summarized[k] = {}
            for thresh, v in metrics[k].items():
                summarized[k][thresh] = np.mean(v)
        else:
            summarized[k] = np.mean(metrics[k])

    return summarized
