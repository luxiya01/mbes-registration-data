import math
from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group
import torch
import torch.utils.data

import mbes_data.common.math.se3 as se3
import mbes_data.common.math.so3 as so3

def uniform_2_sphere(num: int = None):
    """Uniform sampling on a 2-sphere

    Source: https://gist.github.com/andrewbolster/10274979

    Args:
        num: Number of vectors to sample (or None if single)

    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)

    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)

def uniform_on_xy_plane(num: int = None, forbidden_direction: np.ndarray = None):
    """Uniform sampling on the xy plane.
    Args:
        num: Number of phi to sample (or None if single)
        forbidden_direction: If not None, the sampled vector will not be closed
                             to parallel to this vector.
    
    Returns:
        Random vector (np.nparray) of size (num, 3) with norm 1.
        The z-component is always 0.
    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
    
    theta = np.pi / 2
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    if forbidden_direction is not None and np.greater_equal(np.abs(
        np.dot(forbidden_direction, np.array([x, y]))), 0.9):
        print('Rerunning uniform sampling on xy plane due to forbidden direction')
        return uniform_on_xy_plane(num, forbidden_direction)

    return np.stack((x, y, z), axis=-1)

class SplitSourceRef:
    """Clones the point cloud into separate source and reference point clouds"""
    def __call__(self, sample: Dict):
        sample['points_raw'] = sample.pop('points')
        if isinstance(sample['points_raw'], torch.Tensor):
            sample['points_src'] = sample['points_raw'].detach()
            sample['points_ref'] = sample['points_raw'].detach()
        else:  # is numpy
            sample['points_src'] = sample['points_raw'].copy()
            sample['points_ref'] = sample['points_raw'].copy()

        return sample


class Resampler:
    def __init__(self, num: int):
        """Resamples a point cloud containing N points to one containing M

        Guaranteed to have no repeated points if M <= N.
        Otherwise, it is guaranteed that all points appear at least once.

        Args:
            num (int): Number of points to resample to, i.e. M

        """
        self.num = num

    def __call__(self, sample):

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if 'points' in sample:
            sample['points'] = self._resample(sample['points'], self.num)
        else:
            if 'crop_proportion' not in sample:
                src_size, ref_size = self.num, self.num
            elif len(sample['crop_proportion']) == 1:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = self.num
            elif len(sample['crop_proportion']) == 2:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = math.ceil(sample['crop_proportion'][1] * self.num)
                src_size = int(self.num * sample['crop_proportion'][0])
                ref_size = int(self.num * sample['crop_proportion'][1])
            else:
                raise ValueError('Crop proportion must have 1 or 2 elements')

            sample['points_src'] = self._resample(sample['points_src'], src_size)
            sample['points_ref'] = self._resample(sample['points_ref'], ref_size)

        return sample

    @staticmethod
    def _resample(points, k):
        """Resamples the points such that there is exactly k points.

        If the input point cloud has <= k points, it is guaranteed the
        resampled point cloud contains every point in the input.
        If the input point cloud has > k points, it is guaranteed the
        resampled point cloud does not contain repeated point.
        """

        if k <= points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs, :]
        elif points.shape[0] == k:
            return points
        else:
            rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                        np.random.choice(points.shape[0], k - points.shape[0], replace=True)])
            return points[rand_idxs, :]


class FixedResampler(Resampler):
    """Fixed resampling to always choose the first N points.
    Always deterministic regardless of whether the deterministic flag has been set
    """
    @staticmethod
    def _resample(points, k):
        multiple = k // points.shape[0]
        remainder = k % points.shape[0]

        resampled = np.concatenate((np.tile(points, (multiple, 1)), points[:remainder, :]), axis=0)
        return resampled


class RandomJitter:
    """ generate perturbations """
    def __init__(self, scale=0.01, clip=0.05):
        self.scale = scale
        self.clip = clip

    def jitter(self, pts):

        noise = np.clip(np.random.normal(0.0, scale=self.scale, size=(pts.shape[0], 3)),
                        a_min=-self.clip, a_max=self.clip)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def __call__(self, sample):

        if 'points' in sample:
            sample['points'] = self.jitter(sample['points'])
        else:
            sample['points_src'] = self.jitter(sample['points_src'])
            sample['points_ref'] = self.jitter(sample['points_ref'])

        return sample

class RandomJitterMBES:
    """Generate random perturbations to MBES point cloud with different magnitude along xyz axis.
    All args are given in the normalized unit sphere.

    Args:
        scale_x (float): Standard deviation of perturbation along x axis
        scale_y (float): Standard deviation of perturbation along y axis
        scale_z (float): Standard deviation of perturbation along z axis
        clip_x (float): Maximum magnitude of perturbation along x axis
        clip_y (float): Maximum magnitude of perturbation along y axis
        clip_z (float): Maximum magnitude of perturbation along z axis
    """

    def __init__(self, scale_x=0.01, scale_y=0.01, scale_z=0.001, clip_x=0.05, clip_y=0.05, clip_z=0.005):
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_z = scale_z
        self.clip_x = clip_x
        self.clip_y = clip_y
        self.clip_z = clip_z

    def jitter(self, pts):

        # Generate noise for each axis
        noise_x = np.clip(np.random.normal(0.0, scale=self.scale_x, size=(pts.shape[0], 1)),
                            a_min=-self.clip_x, a_max=self.clip_x)
        noise_y = np.clip(np.random.normal(0.0, scale=self.scale_y, size=(pts.shape[0], 1)),
                            a_min=-self.clip_y, a_max=self.clip_y)
        noise_z = np.clip(np.random.normal(0.0, scale=self.scale_z, size=(pts.shape[0], 1)),
                            a_min=-self.clip_z, a_max=self.clip_z)

        # Add noise to xyz
        pts[:, 0] += noise_x[:, 0]
        pts[:, 1] += noise_y[:, 0]
        pts[:, 2] += noise_z[:, 0]

        return pts

    def __call__(self, sample):

        if 'points' in sample:
            sample['points'] = self.jitter(sample['points'])
        else:
            sample['points_src'] = self.jitter(sample['points_src'])
            sample['points_ref'] = self.jitter(sample['points_ref'])

        return sample

class RandomJitterMBESInMeterScale(RandomJitterMBES):
    """Generate random perturbations to MBES point cloud with different magnitude along xyz axis.
    All arguments are given in meters. This method transforms the normalized points to meters by
    multiplying the scale factor, applies perturbation, and then transforms back to normalized points.

    Args:
        scale_x (float): Standard deviation of perturbation along x axis, in meters
        scale_y (float): Standard deviation of perturbation along y axis, in meters
        scale_z (float): Standard deviation of perturbation along z axis, in meters
        clip_x (float): Maximum magnitude of perturbation along x axis, in meters
        clip_y (float): Maximum magnitude of perturbation along y axis, in meters
        clip_z (float): Maximum magnitude of perturbation along z axis, in meters
    """

    def __init__(self, scale_x=0.01, scale_y=0.01, scale_z=0.001, clip_x=0.05, clip_y=0.05, clip_z=0.005):
        super().__init__(scale_x, scale_y, scale_z, clip_x, clip_y, clip_z)

    def _jitter_in_meter_scale(self, points, max_dist):
        points_in_meter = points * max_dist
        points_in_meter = self.jitter(points_in_meter)
        points = points_in_meter / max_dist
        return points

    def __call__(self, sample):
        if 'points' in sample:
            sample['points'] = self._jitter_in_meter_scale(sample['points'], sample['label'].max_dist)
        else:
            sample['points_src'] = self._jitter_in_meter_scale(sample['points_src'], sample['label'].max_dist)
            sample['points_ref'] = self._jitter_in_meter_scale(sample['points_ref'], sample['label'].max_dist)

        return sample


class RandomCrop:
    """Randomly crops the *source* point cloud, approximately retaining half the points

    A direction is randomly sampled from S2, and we retain points which lie within the
    half-space oriented in this direction.
    If p_keep != 0.5, we shift the plane until approximately p_keep points are retained
    """
    def __init__(self, p_keep: List = None):
        if p_keep is None:
            p_keep = [0.7, 0.7]  # Crop both clouds to 70%
        self.p_keep = np.array(p_keep, dtype=np.float32)

    @staticmethod
    def crop(points, p_keep):
        rand_xyz = uniform_2_sphere()
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid

        dist_from_plane = np.dot(points_centered, rand_xyz)
        if p_keep == 0.5:
            mask = dist_from_plane > 0
        else:
            mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

        return points[mask, :]

    def __call__(self, sample):

        sample['crop_proportion'] = self.p_keep
        if np.all(self.p_keep == 1.0):
            return sample  # No need crop

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if len(self.p_keep) == 1:
            sample['points_src'] = self.crop(sample['points_src'], self.p_keep[0])
        else:
            sample['points_src'] = self.crop(sample['points_src'], self.p_keep[0])
            sample['points_ref'] = self.crop(sample['points_ref'], self.p_keep[1])
        return sample

class RandomCropMBES:
    """Randomly crops the point clouds, approximately retaining p_keep% points

    A direction is randomly sampled from XY plane, and we retain points which lie within the
    half-space oriented in this direction.
    If p_keep != 0.5, we shift the plane until approximately p_keep points are retained
    """
    def __init__(self, p_keep: List = None, enforce_forbidden_direction: bool = True):
        if p_keep is None:
            p_keep = [0.7, 0.7]  # Crop both clouds to 70%
        self.p_keep = np.array(p_keep, dtype=np.float32)
        self.enforce_forbidden_direction = enforce_forbidden_direction

    @staticmethod
    def crop(points, p_keep, enforce_forbidden_direction):
        rand_xyz = uniform_on_xy_plane()

        if enforce_forbidden_direction:
            eigenvalues, eigenvectors = np.linalg.eig(np.cov(points[:, :2], rowvar=False))
            min_eigenvalue_idx = np.argmin(eigenvalues)
            forbidden_direction = eigenvectors[:, min_eigenvalue_idx]
            rand_xyz = uniform_on_xy_plane(forbidden_direction=forbidden_direction)

        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid

        dist_from_plane = np.dot(points_centered, rand_xyz)
        if p_keep == 0.5:
            mask = dist_from_plane > 0
        else:
            mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

        return points[mask, :]

    def __call__(self, sample):

        sample['crop_proportion'] = self.p_keep
        if np.all(self.p_keep == 1.0):
            return sample  # No need crop

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if len(self.p_keep) == 1:
            sample['points_src'] = self.crop(sample['points_src'], self.p_keep[0], self.enforce_forbidden_direction)
        else:
            sample['points_src'] = self.crop(sample['points_src'], self.p_keep[0], self.enforce_forbidden_direction)
            sample['points_ref'] = self.crop(sample['points_ref'], self.p_keep[1], self.enforce_forbidden_direction)
        return sample

class RandomTransformSE3:
    def __init__(self, rot_mag: float = 180.0, trans_mag: float = 1.0, random_mag: bool = False):
        """Applies a random rigid transformation to the source point cloud

        Args:
            rot_mag (float): Maximum rotation in degrees
            trans_mag (float): Maximum translation T. Random translation will
              be in the range [-X,X] in each axis
            random_mag (bool): If true, will randomize the maximum rotation, i.e. will bias towards small
                               perturbations
        """
        self._rot_mag = rot_mag
        self._trans_mag = trans_mag
        self._random_mag = random_mag

    def generate_transform(self):
        """Generate a random SE3 transformation (3, 4) """

        if self._random_mag:
            attentuation = np.random.random()
            rot_mag, trans_mag = attentuation * self._rot_mag, attentuation * self._trans_mag
        else:
            rot_mag, trans_mag = self._rot_mag, self._trans_mag

        # Generate rotation
        rand_rot = special_ortho_group.rvs(3)
        axis_angle = Rotation.as_rotvec(Rotation.from_matrix(rand_rot))
        axis_angle *= rot_mag / 180.0
        rand_rot = Rotation.from_rotvec(axis_angle).as_matrix()

        # Generate translation
        rand_trans = np.random.uniform(-trans_mag, trans_mag, 3)
        rand_SE3 = np.concatenate((rand_rot, rand_trans[:, None]), axis=1).astype(np.float32)

        return rand_SE3

    def apply_transform(self, p0, transform_mat):
        p1 = se3.transform(transform_mat, p0[:, :3])
        if p0.shape[1] == 6:  # Need to rotate normals also
            n1 = so3.transform(transform_mat[:3, :3], p0[:, 3:6])
            p1 = np.concatenate((p1, n1), axis=-1)

        igt = transform_mat
        gt = se3.inverse(igt)

        return p1, gt, igt

    def transform(self, tensor):
        transform_mat = self.generate_transform()
        return self.apply_transform(tensor, transform_mat)

    def __call__(self, sample):

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if 'points' in sample:
            sample['points'], _, _ = self.transform(sample['points'])
        else:
            src_transformed, transform_r_s, transform_s_r = self.transform(sample['points_src'])
            sample['transform_gt'] = transform_r_s  # Apply to source to get reference
            sample['points_src'] = src_transformed

        return sample


# noinspection PyPep8Naming
class RandomTransformSE3_euler(RandomTransformSE3):
    """Same as RandomTransformSE3, but rotates using euler angle rotations

    This transformation is consistent to Deep Closest Point but does not
    generate uniform rotations

    """
    def generate_transform(self):

        if self._random_mag:
            attentuation = np.random.random()
            rot_mag, trans_mag = attentuation * self._rot_mag, attentuation * self._trans_mag
        else:
            rot_mag, trans_mag = self._rot_mag, self._trans_mag

        # Generate rotation
        anglex = np.random.uniform() * np.pi * rot_mag / 180.0
        angley = np.random.uniform() * np.pi * rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * rot_mag / 180.0

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        R_ab = Rx @ Ry @ Rz
        t_ab = np.random.uniform(-trans_mag, trans_mag, 3)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]), axis=1).astype(np.float32)
        return rand_SE3


class RandomRotatorZ(RandomTransformSE3):
    """Applies a random z-rotation to the source point cloud"""

    def __init__(self):
        super().__init__(rot_mag=360)

    def generate_transform(self):
        """Generate a random SE3 transformation (3, 4) """

        rand_rot_deg = np.random.random() * self._rot_mag
        rand_rot = Rotation.from_euler('z', rand_rot_deg, degrees=True).as_matrix()
        rand_SE3 = np.pad(rand_rot, ((0, 0), (0, 1)), mode='constant').astype(np.float32)

        return rand_SE3


class RandomTransformSE3_euler_MBES(RandomTransformSE3):
    """Applies a random transformation to the source MBES point cloud.
    Only rotate around z-axis. Allows different translation magnitudes for x,y,z.
    
    Args:
        rot_mag_z (float): Maximum rotation around z-axis in degrees
        trans_mag_x (float): Maximum translation in x-direction
        trans_mag_y (float): Maximum translation in y-direction
        trans_mag_z (float): Maximum translation in z-direction
        random_mag (bool): If true, will randomize the maximum rotation, i.e. will bias towards small
                            perturbations

    """

    def __init__(self, rot_mag_z: float = 360, trans_mag_x: float = 0.5, trans_mag_y: float = 0.5,
                 trans_mag_z: float = 0.05, random_mag: bool = False):
        super().__init__(random_mag=random_mag)
        self._rot_mag_z = rot_mag_z
        self._trans_mag_x = trans_mag_x
        self._trans_mag_y = trans_mag_y
        self._trans_mag_z = trans_mag_z

    def generate_transform(self):
        """Generate a random SE3 transformation (3,4)"""

        if self._random_mag:
            attentuation = np.random.random()
            rot_mag_z = attentuation * self._rot_mag_z
            trans_mag_x = attentuation * self._trans_mag_x
            trans_mag_y = attentuation * self._trans_mag_y
            trans_mag_z = attentuation * self._trans_mag_z
        else:
            rot_mag_z = self._rot_mag_z
            trans_mag_x = self._trans_mag_x
            trans_mag_y = self._trans_mag_y
            trans_mag_z = self._trans_mag_z

        # Generate rotation in z-axis
        anglez = np.random.uniform() * np.pi * rot_mag_z / 180.0

        cosz = np.cos(anglez)
        sinz = np.sin(anglez)
        Rz = np.array([[cosz, -sinz, 0],
                          [sinz, cosz, 0],
                          [0, 0, 1]])
        R_ab = Rz

        # Generate translation
        t_ab = np.zeros(3)
        t_ab[0] = np.random.uniform(-trans_mag_x, trans_mag_x)
        t_ab[1] = np.random.uniform(-trans_mag_y, trans_mag_y)
        t_ab[2] = np.random.uniform(-trans_mag_z, trans_mag_z)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]), axis=1).astype(np.float32)
        return rand_SE3


class ShufflePoints:
    """Shuffles the order of the points"""
    def __call__(self, sample):
        if 'points' in sample:
            sample['points'] = np.random.permutation(sample['points'])
        else:
            sample['points_ref'] = np.random.permutation(sample['points_ref'])
            sample['points_src'] = np.random.permutation(sample['points_src'])
        return sample


class SetDeterministic:
    """Adds a deterministic flag to the sample such that subsequent transforms
    use a fixed random seed where applicable. Used for test"""
    def __call__(self, sample):
        sample['deterministic'] = True
        return sample


class Dict2DcpList:
    """Converts dictionary of tensors into a list of tensors compatible with Deep Closest Point"""
    def __call__(self, sample):

        target = sample['points_src'][:, :3].transpose().copy()
        src = sample['points_ref'][:, :3].transpose().copy()

        rotation_ab = sample['transform_gt'][:3, :3].transpose().copy()
        translation_ab = -rotation_ab @ sample['transform_gt'][:3, 3].copy()

        rotation_ba = sample['transform_gt'][:3, :3].copy()
        translation_ba = sample['transform_gt'][:3, 3].copy()

        euler_ab = Rotation.from_matrix(rotation_ab).as_euler('zyx').copy()
        euler_ba = Rotation.from_matrix(rotation_ba).as_euler('xyz').copy()

        return src, target, \
               rotation_ab, translation_ab, rotation_ba, translation_ba, \
               euler_ab, euler_ba


class Dict2PointnetLKList:
    """Converts dictionary of tensors into a list of tensors compatible with PointNet LK"""
    def __call__(self, sample):

        if 'points' in sample:
            # Train Classifier (pretraining)
            return sample['points'][:, :3], sample['label']
        else:
            # Train PointNetLK
            transform_gt_4x4 = np.concatenate([sample['transform_gt'],
                                               np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)], axis=0)
            return sample['points_src'][:, :3], sample['points_ref'][:, :3], transform_gt_4x4
