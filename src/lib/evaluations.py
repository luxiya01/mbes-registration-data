from collections import defaultdict
import numpy as np
import copy
import open3d as o3d
from lib.benchmark_utils import to_o3d_pcd, to_tsfm

def get_mutual_nearest_neighbor(points_src, points_ref, trans):
    points_src.transform(trans)
    pcd_tree_src = o3d.geometry.KDTreeFlann(points_src)
    pcd_tree_ref = o3d.geometry.KDTreeFlann(points_ref)

    correspondences = []
    for i, point_ref in enumerate(points_ref.points):
        # Get the nearest neighbor of the reference point in the source point cloud
        [_, idx_src, dist_src] = pcd_tree_src.search_knn_vector_3d(point_ref, 1)
        # Get the nearest neighbor of points_src[idx_src] in the reference point cloud
        [_, idx_ref, dist_ref] = pcd_tree_ref.search_knn_vector_3d(points_src.points[idx_src[0]], 1)
        # Check if they are mutual nearest neighbors
        if i == idx_ref[0]:
            correspondences.append([idx_src[0], idx_ref[0]])
    return np.array(correspondences)

def get_predicted_correspondence_gt_distance(data:dict, transform_pred: np.ndarray) -> np.ndarray:
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

    corr_points_src_gt_trans = np.array(points_src_gt_trans.points)[corr_pred[:, 0]]
    corr_points_ref = np.array(points_ref.points)[corr_pred[:, 1]]
    
    # Compute Euclidean distance between GT_t((corr[src])) and corr[ref]
    distances = np.linalg.norm(corr_points_src_gt_trans-corr_points_ref, axis=1)

    return distances

def fmr_wrt_distances(distances: np.ndarray,
                         inlier_ratio_thresh=0.05) -> float:
    """
    Args:
        distances: Euclidean distances between GT-transformed source points and reference points
                    for all pairs of point clouds in the dataset
        inlier_ratio_thresh: Inlier ratio threshold
                             default = 5%
    """
    fmr_wrt_distances = defaultdict(list)
    fmr_inlier_ratios = defaultdict(list)
    for distance_threshold in range(0, 20): # from 0.0 to 2.0 meters with 0.1 m step
        distance_threshold = distance_threshold / 10.
        for distance in distances:
            inlier_percentage = (distance < distance_threshold).mean()
            success = inlier_percentage > inlier_ratio_thresh

            fmr_inlier_ratios[distance_threshold].append(inlier_percentage)
            fmr_wrt_distances[distance_threshold].append(success)

    fmr_wrt_distances = {k: np.mean(v) for k, v in fmr_wrt_distances.items()}
    fmr_inlier_ratios = {k: np.mean(v) for k, v in fmr_inlier_ratios.items()}
    print(f'FMR wrt distance @inlier ratio = {inlier_ratio_thresh}%:\n {fmr_wrt_distances}')
    print(f'FMR wrt distances inlier ratio: {fmr_inlier_ratios}')

def fmr_wrt_inlier_ratio(distances: np.ndarray,
                         distance_threshold=0.5) -> float:
    """
    Args:
        distances: Euclidean distances between GT-transformed source points and reference points
                    for all pairs of point clouds in the dataset
        distance_thresh: Inlier distance threshold
                         default = 0.5m
    """
    fmr_wrt_inlier_ratio = defaultdict(list)
    for inlier_ratio in range(0, 21): # from 0.0 to 10.0 meters with 0.1 m step
        inlier_ratio = inlier_ratio / 100.
        for distance in distances:
            inlier_percentage = (distance < distance_threshold).mean()
            success = inlier_percentage > inlier_ratio
            fmr_wrt_inlier_ratio[inlier_ratio].append(success)

    fmr_wrt_inlier_ratio = {k: np.mean(v) for k, v in fmr_wrt_inlier_ratio.items()}
    print(f'FMR wrt inlier ratios @{distance_threshold}m:\n {fmr_wrt_inlier_ratio}')