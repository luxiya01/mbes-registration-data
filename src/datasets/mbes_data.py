"""Data loader
"""
import argparse, os, torch, h5py, torchvision
from typing import List

import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
from dataclasses import dataclass

import datasets.transforms as Transforms
import common.math.se3 as se3
from lib.benchmark_utils import get_correspondences, to_o3d_pcd, to_tsfm
  

def get_multibeam_train_datasets(args: argparse.Namespace):
    train_transforms, val_transforms = get_transforms(args.noise_type,
                                                      args.rot_mag_z,
                                                      args.trans_mag_x, args.trans_mag_y, args.trans_mag_z,
                                                      args.scale_x, args.scale_y, args.scale_z,
                                                      args.clip_x, args.clip_y, args.clip_z,
                                                      args.num_points, args.partial)
    train_transforms = torchvision.transforms.Compose(train_transforms)
    val_transforms = torchvision.transforms.Compose(val_transforms)

    if args.dataset_type == 'multibeam_npy':
        train_data = MultibeamNpy(args, args.root, subset=args.subset_train,
                                 transform=train_transforms)
        val_data = MultibeamNpy(args, args.root, subset=args.subset_val,
                               transform=val_transforms)
    else:
        raise NotImplementedError

    return train_data, val_data


def get_multibeam_test_datasets(args: argparse.Namespace):
    _, test_transforms = get_transforms(args.noise_type,
                                        args.rot_mag_z,
                                        args.trans_mag_x, args.trans_mag_y, args.trans_mag_z,
                                        args.scale_x, args.scale_y, args.scale_z,
                                        args.clip_x, args.clip_y, args.clip_z,
                                        args.num_points, args.partial)
    test_transforms = torchvision.transforms.Compose(test_transforms)

    if args.dataset_type == 'multibeam_npy':
        test_data = MultibeamNpy(args, args.root, subset=args.subset_test,
                                transform=test_transforms)
    else:
        raise NotImplementedError

    return test_data


def get_transforms(noise_type: str,
                   rot_mag_z: float = 360.0,
                   trans_mag_x: float = 0.5, trans_mag_y: float = 0.5, trans_mag_z: float = 0.05,
                   scale_x: float = 0.01, scale_y: float = 0.01, scale_z: float = 0.001,
                   clip_x: float = 0.05, clip_y: float = 0.05, clip_z: float = 0.005,
                   num_points: int = 1024, partial_p_keep: List = None):
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

    Returns:
        train_transforms, test_transforms: Both contain list of transformations to be applied
    """

    partial_p_keep = partial_p_keep if partial_p_keep is not None else [0.7, 0.7]

    if noise_type == "clean":
        # 1-1 correspondence for each point (resample first before splitting), no noise
        train_transforms = [Transforms.Resampler(num_points),
                            Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler_MBES(rot_mag_z=rot_mag_z,
                                                                     trans_mag_x=trans_mag_x,
                                                                     trans_mag_y=trans_mag_y,
                                                                     trans_mag_z=trans_mag_z),
                            Transforms.ShufflePoints()]

        #TODO: double check FixedResampler implementation!
        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.FixedResampler(num_points),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomTransformSE3_euler_MBES(rot_mag_z=rot_mag_z,
                                                                     trans_mag_x=trans_mag_x,
                                                                     trans_mag_y=trans_mag_y,
                                                                     trans_mag_z=trans_mag_z),
                           Transforms.ShufflePoints()]

    elif noise_type == "jitter":
        # Points randomly sampled (might not have perfect correspondence), gaussian noise to position
        train_transforms = [Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler_MBES(rot_mag_z=rot_mag_z,
                                                                     trans_mag_x=trans_mag_x,
                                                                     trans_mag_y=trans_mag_y,
                                                                     trans_mag_z=trans_mag_z),
                            Transforms.Resampler(num_points),
                            Transforms.RandomJitterMBES(scale_x=scale_x, scale_y=scale_y, scale_z=scale_z, clip_x=clip_x, clip_y=clip_y, clip_z=clip_z),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomTransformSE3_euler_MBES(rot_mag_z=rot_mag_z,
                                                                    trans_mag_x=trans_mag_x,
                                                                    trans_mag_y=trans_mag_y,
                                                                    trans_mag_z=trans_mag_z),
                           Transforms.Resampler(num_points),
                           Transforms.RandomJitterMBES(scale_x=scale_x, scale_y=scale_y, scale_z=scale_z, clip_x=clip_x, clip_y=clip_y, clip_z=clip_z),
                           Transforms.ShufflePoints()]

    elif noise_type == "crop":
        # Both source and reference point clouds cropped, plus same noise in "jitter"
        train_transforms = [Transforms.SplitSourceRef(),
                            Transforms.RandomCropMBES(partial_p_keep),
                            Transforms.RandomTransformSE3_euler_MBES(rot_mag_z=rot_mag_z,
                                                                    trans_mag_x=trans_mag_x,
                                                                    trans_mag_y=trans_mag_y,
                                                                    trans_mag_z=trans_mag_z),
                            Transforms.Resampler(num_points),
                            Transforms.RandomJitterMBES(scale_x=scale_x, scale_y=scale_y, scale_z=scale_z, clip_x=clip_x, clip_y=clip_y, clip_z=clip_z),
                            Transforms.ShufflePoints()]
 
        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomCropMBES(partial_p_keep),
                           Transforms.RandomTransformSE3_euler_MBES(rot_mag_z=rot_mag_z,
                                                                    trans_mag_x=trans_mag_x,
                                                                    trans_mag_y=trans_mag_y,
                                                                    trans_mag_z=trans_mag_z),
                           Transforms.Resampler(num_points),
                           Transforms.RandomJitterMBES(scale_x=scale_x, scale_y=scale_y, scale_z=scale_z, clip_x=clip_x, clip_y=clip_y, clip_z=clip_z),
                           Transforms.ShufflePoints()]
    elif noise_type == "jitter_in_meters":
        # Points randomly sampled (might not have perfect correspondence), gaussian noise to position in meters
        train_transforms = [Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler_MBES(rot_mag_z=rot_mag_z,
                                                                     trans_mag_x=trans_mag_x,
                                                                     trans_mag_y=trans_mag_y,
                                                                     trans_mag_z=trans_mag_z),
                            Transforms.Resampler(num_points),
                            Transforms.RandomJitterMBESInMeterScale(scale_x=scale_x, scale_y=scale_y, scale_z=scale_z, clip_x=clip_x, clip_y=clip_y, clip_z=clip_z),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomTransformSE3_euler_MBES(rot_mag_z=rot_mag_z,
                                                                    trans_mag_x=trans_mag_x,
                                                                    trans_mag_y=trans_mag_y,
                                                                    trans_mag_z=trans_mag_z),
                           Transforms.Resampler(num_points),
                           Transforms.RandomJitterMBESInMeterScale(scale_x=scale_x, scale_y=scale_y, scale_z=scale_z, clip_x=clip_x, clip_y=clip_y, clip_z=clip_z),
                           Transforms.ShufflePoints()]
    elif noise_type == "crop_then_jitter_in_meters":
        # Both source and reference point clouds cropped, plus same noise in "jitter_in_meters"
        train_transforms = [Transforms.SplitSourceRef(),
                            Transforms.RandomCropMBES(partial_p_keep),
                            Transforms.RandomTransformSE3_euler_MBES(rot_mag_z=rot_mag_z,
                                                                    trans_mag_x=trans_mag_x,
                                                                    trans_mag_y=trans_mag_y,
                                                                    trans_mag_z=trans_mag_z),
                            Transforms.Resampler(num_points),
                            Transforms.RandomJitterMBESInMeterScale(scale_x=scale_x, scale_y=scale_y, scale_z=scale_z, clip_x=clip_x, clip_y=clip_y, clip_z=clip_z),
                            Transforms.ShufflePoints()]
 
        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomCropMBES(partial_p_keep),
                           Transforms.RandomTransformSE3_euler_MBES(rot_mag_z=rot_mag_z,
                                                                    trans_mag_x=trans_mag_x,
                                                                    trans_mag_y=trans_mag_y,
                                                                    trans_mag_z=trans_mag_z),
                           Transforms.Resampler(num_points),
                           Transforms.RandomJitterMBESInMeterScale(scale_x=scale_x, scale_y=scale_y, scale_z=scale_z, clip_x=clip_x, clip_y=clip_y, clip_z=clip_z),
                           Transforms.ShufflePoints()]
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
        self._root = root
        self._subset = subset
        self._transform = transform
        self.n_in_feats = args.in_feats_dim
        self.overlap_radius = args.overlap_radius
        self.nbr_pings_per_patch = args.nbr_pings_per_patch
        self.nbr_beams_per_patch = args.nbr_beams_per_patch
        self.voxel_size = args.voxel_size
        self.draw_items = args.draw_items
        
        with open(os.path.join(self._root, f'{self._subset}_files.txt')) as fid:
            npy_filelist = [line.strip() for line in fid]
            npy_filelist = [os.path.join(self._root, f) for f in npy_filelist]
        self._data, self._labels = self._read_npy_files(npy_filelist)

        
    def __getitem__(self, item):
        sample = {'points': self._data[item], 'label': self._labels[item], 'idx': np.array(item,
                                                                                           dtype=np.int32)}
        
        if self._transform:
            sample = self._transform(sample)
        # transform to our format
        src_pcd = sample['points_src'][:,:3]
        tgt_pcd = sample['points_ref'][:,:3]
        rot = sample['transform_gt'][:,:3]
        trans = sample['transform_gt'][:,3][:,None]
        matching_inds = get_correspondences(to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd),to_tsfm(rot,trans),self.overlap_radius)
        
        if(self.n_in_feats == 1):
            src_feats=np.ones_like(src_pcd[:,:1]).astype(np.float32)
            tgt_feats=np.ones_like(tgt_pcd[:,:1]).astype(np.float32)
        elif(self.n_in_feats == 3):
            src_feats = src_pcd.astype(np.float32)
            tgt_feats = tgt_pcd.astype(np.float32)

        for k,v in sample.items():
            if k not in ['deterministic','label', 'idx']:
                sample[k] = torch.from_numpy(v).unsqueeze(0)

        if self.draw_items:
            # draw src and tgt point clouds with matching inds
            src_pcd_o3d = to_o3d_pcd(src_pcd)
            tgt_pcd_o3d = to_o3d_pcd(tgt_pcd)
            #src_pcd_o3d.paint_uniform_color([1, 0.706, 0])
            #tgt_pcd_o3d.paint_uniform_color([0, 0.651, 0.929])
            o3d.visualization.draw_geometries([src_pcd_o3d, tgt_pcd_o3d])

        return src_pcd,tgt_pcd,src_feats,tgt_feats,rot,trans, matching_inds, src_pcd, tgt_pcd, sample

    def __len__(self):
        return len(self._data)

    def _read_npy_files(self, fnames):

        all_data = []
        all_labels = []
        
        for fname in fnames:
            f = np.load(fname)
            nbr_pings_in_f, nbr_beams_in_f, _ = f.shape

            # Split the npy into patches of (nbr_pings_per_patch, nbr_beams_per_patch, 3)
            for ping_id_start in range(0, nbr_pings_in_f, self.nbr_pings_per_patch):
                for beam_id_start in range(0, nbr_beams_in_f, self.nbr_beams_per_patch):
                    ping_id_end = min(ping_id_start + self.nbr_pings_per_patch, nbr_pings_in_f)
                    beam_id_end = min(beam_id_start + self.nbr_beams_per_patch, nbr_beams_in_f)

                    patch_raw = f[ping_id_start:ping_id_end,
                              beam_id_start:beam_id_end]
                    
                    # Filter data points that are (0, 0, 0), i.e. no data
                    patch_raw = patch_raw[~np.all(patch_raw==0, axis=2)].astype(np.float32)

                    # patch shape could be 0 if the Ping No restarts itself...
                    if patch_raw.shape[0] == 0:
                        continue

                    # Voxel down sampling
                    patch = self._voxel_down_sample(patch_raw)
                    print(f'Patch shape before voxel down sampling: {patch_raw.shape}\n'
                          f'Patch shape after voxel down sampling: {patch.shape}\n'
                          f'Retained {patch.shape[0] / patch_raw.shape[0]*100:.2f}% points\n')

                    # Normalize points
                    patch = self._normalize_points(patch)
                    patch_points = np.random.permutation(patch['patch_normalized'])
                    patch_label = PatchLabel(fname, ping_id_start, ping_id_end,
                                                 beam_id_start, beam_id_end,
                                                 centroid=patch['centroid'],
                                                 max_dist=patch['max_dist'])

                    # Store permuted points
                    all_data.append(patch_points.astype(np.float32))
                    all_labels.append(patch_label)
        
        np.save(os.path.join(self._root, f'{self._subset}_data.npy'), all_data)
        np.save(os.path.join(self._root, f'{self._subset}_labels.npy'), all_labels)
        
        return all_data, all_labels

    def _voxel_down_sample(self, patch):
        pcd = to_o3d_pcd(patch.reshape(-1, 3))
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        patch = np.asarray(pcd.points)
        return patch

    def _normalize_points(self, patch):
        centroid = np.mean(patch, axis=0)
        patch_centered = patch - centroid
        max_dist = np.max(np.linalg.norm(patch_centered, axis=1))
        patch_normalized = patch_centered / max_dist
        return {'patch_normalized': patch_normalized, 'centroid': centroid, 'max_dist': max_dist}


@dataclass
class PatchLabel:
    fname: str
    ping_id_start: int
    ping_id_end: int
    beam_id_start: int
    beam_id_end: int
    centroid: float
    max_dist: float
