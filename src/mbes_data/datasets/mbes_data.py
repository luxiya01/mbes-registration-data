"""Data loader
"""
import argparse, os, torch, h5py, torchvision
from typing import List

import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
from dataclasses import dataclass, asdict

from tqdm import tqdm

import mbes_data.datasets.transforms as Transforms
import mbes_data.common.math.se3 as se3
from mbes_data.lib.benchmark_utils import get_correspondences, to_o3d_pcd, to_tsfm
from mbes_data.lib.utils import setup_seed
import MinkowskiEngine as ME

setup_seed(0)


def get_multibeam_datasets(args: argparse.Namespace):
    if (args.dataset == 'multibeam'):
        train_set, val_set = get_multibeam_train_datasets(args)
        test_set = get_multibeam_test_datasets(args)
    else:
        raise NotImplementedError
    return train_set, val_set, test_set


def get_multibeam_train_datasets(args: argparse.Namespace):
    train_transforms, val_transforms = get_transforms(
        args.noise_type, args.rot_mag_z, args.trans_mag_x, args.trans_mag_y,
        args.trans_mag_z, args.scale_x, args.scale_y, args.scale_z,
        args.clip_x, args.clip_y, args.clip_z, args.num_points, args.partial)
    train_transforms = torchvision.transforms.Compose(train_transforms)
    val_transforms = torchvision.transforms.Compose(val_transforms)

    if args.dataset_type == 'multibeam_npy':
        train_data = MultibeamNpy(args,
                                  args.root,
                                  subset=args.subset_train,
                                  transform=train_transforms)
        val_data = MultibeamNpy(args,
                                args.root,
                                subset=args.subset_val,
                                transform=val_transforms)
    elif args.dataset_type == 'multibeam_npy_for_fcgf':
        train_data = MultibeamNpyForFCGFTraining(args,
                                                 args.root,
                                                 subset=args.subset_train,
                                                 transform=train_transforms)
        val_data = MultibeamNpyForFCGFTraining(args,
                                               args.root,
                                               subset=args.subset_val,
                                               transform=val_transforms)
    elif args.dataset_type == 'multibeam_npy_for_dgr':
        train_data = MultibeamNpyForDGR(args,
                                        args.root,
                                        subset=args.subset_train,
                                        transform=train_transforms)
        val_data = MultibeamNpyForDGR(args,
                                      args.root,
                                      subset=args.subset_val,
                                      transform=val_transforms)
    elif args.dataset_type == 'multibeam_npy_for_overlap_predator':
        train_data = MultibeamNpyForOverlapPredator(args,
                                                    args.root,
                                                    subset=args.subset_train,
                                                    transform=train_transforms)
        val_data = MultibeamNpyForOverlapPredator(args,
                                                    args.root,
                                                    subset=args.subset_val,
                                                    transform=val_transforms)
    elif args.dataset_type == 'multibeam_npy_for_bathynn':
        train_data = MultibeamNpyForBathyNN(args,
                                            args.root,
                                            subset=args.subset_train,
                                            transform=train_transforms)
        val_data = MultibeamNpyForBathyNN(args,
                                            args.root,
                                            subset=args.subset_val,
                                            transform=val_transforms)
    else:
        raise NotImplementedError

    return train_data, val_data


def get_multibeam_test_datasets(args: argparse.Namespace):
    if 'enforce_forbidden_direction' not in args:
        args.enforce_forbidden_direction = False

    _, test_transforms = get_transforms(
        args.noise_type, args.rot_mag_z, args.trans_mag_x, args.trans_mag_y,
        args.trans_mag_z, args.scale_x, args.scale_y, args.scale_z,
        args.clip_x, args.clip_y, args.clip_z, args.num_points, args.partial,
        args.enforce_forbidden_direction)
    test_transforms = torchvision.transforms.Compose(test_transforms)

    if args.dataset_type == 'multibeam_npy':
        test_data = MultibeamNpy(args,
                                 args.root,
                                 subset=args.subset_test,
                                 transform=test_transforms)
    elif args.dataset_type == 'multibeam_npy_for_fcgf':
        test_data = MultibeamNpyForFCGFTraining(args,
                                                args.root,
                                                subset=args.subset_test,
                                                transform=test_transforms)
    elif args.dataset_type == 'multibeam_npy_for_dgr':
        test_data = MultibeamNpyForDGR(args,
                                      args.root,
                                      subset=args.subset_test,
                                      transform=test_transforms)
    elif args.dataset_type == 'multibeam_npy_for_overlap_predator':
        test_data = MultibeamNpyForOverlapPredator(args,
                                                    args.root,
                                                    subset=args.subset_test,
                                                    transform=test_transforms)
    elif args.dataset_type == 'multibeam_npy_for_bathynn':
        test_data = MultibeamNpyForBathyNN(args,
                                            args.root,
                                            subset=args.subset_test,
                                            transform=test_transforms)
    else:
        raise NotImplementedError

    return test_data


def get_transforms(noise_type: str,
                   rot_mag_z: float = 360.0,
                   trans_mag_x: float = 0.5,
                   trans_mag_y: float = 0.5,
                   trans_mag_z: float = 0.05,
                   scale_x: float = 0.01,
                   scale_y: float = 0.01,
                   scale_z: float = 0.001,
                   clip_x: float = 0.05,
                   clip_y: float = 0.05,
                   clip_z: float = 0.005,
                   num_points: int = 1024,
                   partial_p_keep: List = None,
                   enforce_forbidden_direction: bool = True):
    """Get the list of transformation to be used for training or evaluating RegNet

    Args:
        noise_type: Either 'clean', 'jitter', 'crop'.
          Depending on the option, some of the subsequent arguments may be ignored.
        rot_mag_z (float): Maximum rotation around z-axis in degrees.
          Default: 360.0
        trans_mag_x (float): Maximum translation in x-direction
          Default: 0.5
        trans_mag_y (float): Maximum translation in y-direction
          Default: 0.5
        trans_mag_z (float): Maximum translation in z-direction
          Default: 0.05
        scale_x (float): Standard deviation of perturbation along x axis
          Default: 0.01
        scale_y (float): Standard deviation of perturbation along y axis
          Default: 0.01
        scale_z (float): Standard deviation of perturbation along z axis
          Default: 0.001
        clip_x (float): Maximum magnitude of perturbation along x axis
          Default: 0.05
        clip_y (float): Maximum magnitude of perturbation along y axis
          Default: 0.05
        clip_z (float): Maximum magnitude of perturbation along z axis
          Default: 0.005
        num_points: Number of points to uniformly resample to.
          Note that this is with respect to the full point cloud. The number of
          points will be proportionally less if cropped
        partial_p_keep: Proportion to keep during cropping, [src_p, ref_p]
          Default: [0.7, 0.7], i.e. Crop both source and reference to ~70%
        enforce_forbidden_direction: Used for crop noise type. If True, the sampled
          crop direction will NOT be parallel to the 2nd principle axis of the point cloud
          (assumed travel direction of the vehicle).

    Returns:
        train_transforms, test_transforms: Both contain list of transformations to be applied
    """

    partial_p_keep = partial_p_keep if partial_p_keep is not None else [
        0.7, 0.7
    ]

    if noise_type == "clean":
        # 1-1 correspondence for each point (resample first before splitting), no noise
        train_transforms = [
            Transforms.Resampler(num_points),
            Transforms.RandomTransformSE3_euler_MBES(rot_mag_z=rot_mag_z,
                                                     trans_mag_x=trans_mag_x,
                                                     trans_mag_y=trans_mag_y,
                                                     trans_mag_z=trans_mag_z),
            Transforms.ShufflePoints()
        ]

        #TODO: double check FixedResampler implementation!
        test_transforms = [
            Transforms.SetDeterministic(),
            Transforms.FixedResampler(num_points),
            Transforms.RandomTransformSE3_euler_MBES(rot_mag_z=rot_mag_z,
                                                     trans_mag_x=trans_mag_x,
                                                     trans_mag_y=trans_mag_y,
                                                     trans_mag_z=trans_mag_z),
            Transforms.ShufflePoints()
        ]

    elif noise_type == "jitter":
        # Points randomly sampled (might not have perfect correspondence), gaussian noise to position
        train_transforms = [
            Transforms.RandomTransformSE3_euler_MBES(rot_mag_z=rot_mag_z,
                                                     trans_mag_x=trans_mag_x,
                                                     trans_mag_y=trans_mag_y,
                                                     trans_mag_z=trans_mag_z),
            Transforms.Resampler(num_points),
            Transforms.RandomJitterMBES(scale_x=scale_x,
                                        scale_y=scale_y,
                                        scale_z=scale_z,
                                        clip_x=clip_x,
                                        clip_y=clip_y,
                                        clip_z=clip_z),
            Transforms.ShufflePoints()
        ]

        test_transforms = [
            Transforms.SetDeterministic(),
            Transforms.RandomTransformSE3_euler_MBES(rot_mag_z=rot_mag_z,
                                                     trans_mag_x=trans_mag_x,
                                                     trans_mag_y=trans_mag_y,
                                                     trans_mag_z=trans_mag_z),
            Transforms.Resampler(num_points),
            Transforms.RandomJitterMBES(scale_x=scale_x,
                                        scale_y=scale_y,
                                        scale_z=scale_z,
                                        clip_x=clip_x,
                                        clip_y=clip_y,
                                        clip_z=clip_z),
            Transforms.ShufflePoints()
        ]

    elif noise_type == "crop":
        # Both source and reference point clouds cropped, plus same noise in "jitter"
        train_transforms = [
            Transforms.RandomCropMBES(
                partial_p_keep,
                enforce_forbidden_direction=enforce_forbidden_direction),
            Transforms.RandomTransformSE3_euler_MBES(rot_mag_z=rot_mag_z,
                                                     trans_mag_x=trans_mag_x,
                                                     trans_mag_y=trans_mag_y,
                                                     trans_mag_z=trans_mag_z),
            Transforms.Resampler(num_points),
            Transforms.RandomJitterMBES(scale_x=scale_x,
                                        scale_y=scale_y,
                                        scale_z=scale_z,
                                        clip_x=clip_x,
                                        clip_y=clip_y,
                                        clip_z=clip_z),
            Transforms.ShufflePoints()
        ]

        test_transforms = [
            Transforms.SetDeterministic(),
            Transforms.RandomCropMBES(
                partial_p_keep,
                enforce_forbidden_direction=enforce_forbidden_direction),
            Transforms.RandomTransformSE3_euler_MBES(rot_mag_z=rot_mag_z,
                                                     trans_mag_x=trans_mag_x,
                                                     trans_mag_y=trans_mag_y,
                                                     trans_mag_z=trans_mag_z),
            Transforms.Resampler(num_points),
            Transforms.RandomJitterMBES(scale_x=scale_x,
                                        scale_y=scale_y,
                                        scale_z=scale_z,
                                        clip_x=clip_x,
                                        clip_y=clip_y,
                                        clip_z=clip_z),
            Transforms.ShufflePoints()
        ]
    else:
        raise NotImplementedError

    return train_transforms, test_transforms


class MultibeamNpy(Dataset):

    def __init__(self, args, root: str, subset: str = 'train', transform=None):
        """Multibeam .npy dataset.
        Implemented similarly to the original ModelNetHdf dataset class.

        Args:
            root (str): Folder containing processed dataset
            subset (str): Dataset subset, either 'train' or 'test'
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.config = args
        self._subset = subset
        patches_folder_name = (
            f'patches-{args.nbr_pings_per_patch}pings-{args.nbr_beams_per_patch}beams-'
            f'{args.pings_step}pings_step-{args.beams_step}beams_step')
        self._subset_root = os.path.join(
            root, os.path.join(subset, patches_folder_name))
        self._transform = transform
        self.n_in_feats = args.in_feats_dim
        self.overlap_radius = args.overlap_radius
        self.voxel_size = args.voxel_size
        self.pair_overlap_ratio = args.pair_overlap_ratio
        self.draw_items = args.draw_items
        self._pairs = self._get_pairs()

    def _get_pairs(self):
        patches = sorted(
            [x for x in os.listdir(self._subset_root) if x.endswith('.npz')],
            key=lambda x: int(x.split('.')[0]))
        pairs = []

        # The patches were segmented with 80% overlap consecutively, so to achieve a e.g. 20% overlap
        # we would need to skip 4 patches in between
        if isinstance(self.pair_overlap_ratio, float):
            self.pair_overlap_ratio = [self.pair_overlap_ratio]
        for pair_overlap_ratio in self.pair_overlap_ratio:
            index_step = np.ceil((1 - pair_overlap_ratio) / 0.2).astype(int)
            print(f'index_step: {index_step}')

            for i in range(len(patches) - index_step):
                pairs.append((os.path.join(self._subset_root, patches[i]),
                              os.path.join(self._subset_root,
                                           patches[i + index_step])))
        return pairs

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, item):
        # TODO: implement custom collate_fn to handle variable number of matching_inds and allow for batch_size > 1!
        sample = self._load_and_center_patch_pair(item)

        if self._transform:
            sample = self._transform(sample)
        # transform to our format
        src_pcd = sample['points_src'][:, :3]
        tgt_pcd = sample['points_ref'][:, :3]
        rot = sample['transform_gt'][:, :3]
        trans = sample['transform_gt'][:, 3][:, None]
        matching_inds = get_correspondences(to_o3d_pcd(src_pcd),
                                            to_o3d_pcd(tgt_pcd),
                                            to_tsfm(rot, trans),
                                            self.overlap_radius)

        if (self.n_in_feats == 1):
            src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
            tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
        elif (self.n_in_feats == 3):
            src_feats = src_pcd.astype(np.float32)
            tgt_feats = tgt_pcd.astype(np.float32)

        if self.draw_items:
            src_pcd_o3d = to_o3d_pcd(src_pcd)
            tgt_pcd_o3d = to_o3d_pcd(tgt_pcd)
            o3d.visualization.draw_geometries([src_pcd_o3d, tgt_pcd_o3d])

        # Filter out pairs with no matching inds
        if matching_inds.shape[0] == 0:
            return None

        # TODO: implement custom collate_fn to handle variable number of matching_inds
        return {
            'points_src': src_pcd,
            'points_ref': tgt_pcd,
            'features_src': src_feats,
            'features_ref': tgt_feats,
            'transform_gt': sample['transform_gt'],
            'transform_gt_rot': rot,
            'transform_gt_trans': trans,
            'matching_inds': matching_inds,
            'idx': sample['idx'],
            'src_idx': self._pairs[item][0],
            'ref_idx': self._pairs[item][1]
        }

    def _voxel_down_sample(self, patch):
        pcd = to_o3d_pcd(patch.reshape(-1, 3))
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        patch = np.asarray(pcd.points, dtype=np.float32)
        return patch

    def _process_one_patch(self, patch, centroid):
        patch = patch - centroid
        patch = self._voxel_down_sample(patch)
        return patch

    def _load_and_center_patch_pair(self, item):
        patch_src = np.load(self._pairs[item][0])['submap']
        patch_ref = np.load(self._pairs[item][1])['submap']

        centroid = np.concatenate([patch_src,
                                   patch_ref]).mean(axis=0, dtype=np.float64)
        patch_src = self._process_one_patch(patch_src, centroid)
        patch_ref = self._process_one_patch(patch_ref, centroid)
        return {
            'points_src': patch_src,
            'points_ref': patch_ref,
            'centroid': centroid,
            'idx': item
        }

    def _convert_to_ME(self, data):
        _, sel0 = ME.utils.sparse_quantize(data['points_src'] /
                                           self.voxel_size,
                                           return_index=True)
        _, sel1 = ME.utils.sparse_quantize(data['points_ref'] /
                                           self.voxel_size,
                                           return_index=True)

        xyz0 = torch.from_numpy(data['points_src'][sel0])
        xyz1 = torch.from_numpy(data['points_ref'][sel1])

        feats0 = torch.from_numpy(data['features_src'][sel0])
        feats1 = torch.from_numpy(data['features_ref'][sel1])

        gt_trans = to_tsfm(data['transform_gt_rot'],
                           data['transform_gt_trans'])
        matching_inds = get_correspondences(to_o3d_pcd(xyz0), to_o3d_pcd(xyz1),
                                            gt_trans, self.overlap_radius).to(dtype=torch.int32)

        # Filter out pairs with no matching inds after ME sparse quantize
        if matching_inds.shape[0] == 0:
            return None

        # Get coords
        coords0 = np.floor(xyz0 / self.voxel_size)
        coords1 = np.floor(xyz1 / self.voxel_size)

        return (xyz0, xyz1, coords0, coords1, feats0, feats1, matching_inds,
                gt_trans)

    def reset_seed(self, seed):
        setup_seed(seed)


class MultibeamNpyForFCGFTraining(MultibeamNpy):

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        # Filter out pairs with no matching inds
        if data is None:
            print(f'No matching inds for pair {idx}!')
            return None
        return self._convert_to_ME(data)

class MultibeamNpyForDGR(MultibeamNpy):
    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        # Filter out pairs with no matching inds
        if data is None:
            print(f'No matching inds for pair {idx}!')
            return None

        extra_package = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                extra_package[k] = torch.from_numpy(v).float()
            else:
                extra_package[k] = v
        return *self._convert_to_ME(data), data

class MultibeamNpyForOverlapPredator(MultibeamNpy):
    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        if data is None:
            print(f'No matching inds for pair {idx}!')
            return None

        return (data['points_src'], data['points_ref'],
                data['features_src'], data['features_ref'],
                data['transform_gt_rot'], data['transform_gt_trans'],
                data['matching_inds'],
                data['points_src'], data['points_ref'],
                data)

class MultibeamNpyForBathyNN(MultibeamNpy):
    def __getitem__(self, item):
        data = super().__getitem__(item)

        if data is None:
            print(f'No matching inds for pair {item}!')
            return None

        src_pose = torch.from_numpy(np.mean(data['points_src'], axis=0).reshape(1, 3)).float()
        tgt_pose = torch.from_numpy(np.mean(data['points_ref'], axis=0).reshape(1, 3)).float()

        src_cloud = torch.from_numpy(data['points_src']).float()
        tgt_cloud = torch.from_numpy(data['points_ref']).float()
        src_cloud_centered = src_cloud - src_pose
        tgt_cloud_centered = tgt_cloud - tgt_pose

        src_tgt_pose = torch.cat((src_pose, tgt_pose), dim=1)
        data_src = torch.cat((src_cloud_centered, src_cloud), dim=1)
        data_tgt = torch.cat((tgt_cloud_centered, tgt_cloud), dim=1)
        indices = torch.tensor([item, item])

        return {'data_src': data_src,
                'data_tgt': data_tgt,
                'src_tgt_pose': src_tgt_pose,
                'indices': indices,
                'sample': data}