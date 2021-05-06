import logging
import os
import pickle

import h5py
import multiprocessing
import numpy as np
# import pcl
import torch.utils.data as data
from tqdm import tqdm
from multiprocessing import Pool

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

EX_SIZE = 2000


def filter_cloud_by_threshold(cloud,
                              x_min=-np.inf,
                              x_max=np.inf,
                              y_min=-np.inf,
                              y_max=np.inf,
                              z_min=-np.inf,
                              z_max=np.inf):
    cloud = cloud[cloud[:, 0] >= x_min]
    cloud = cloud[cloud[:, 0] < x_max]
    cloud = cloud[cloud[:, 1] >= y_min]
    cloud = cloud[cloud[:, 1] < y_max]
    cloud = cloud[cloud[:, 2] >= z_min]
    cloud = cloud[cloud[:, 2] < z_max]
    return cloud


def extract_person(cloud):
    cloud = pcl.PointCloud(cloud.astype(np.float32))

    seg = cloud.make_segmenter()
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_MaxIterations(100)
    seg.set_distance_threshold(0.5)

    tree = cloud.make_kdtree()

    ec = cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.1)
    ec.set_MinClusterSize(1000)
    ec.set_MaxClusterSize(10000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    points = np.zeros((len(cluster_indices[0]), 3), dtype=np.float32)
    for i, indice in enumerate(cluster_indices[0]):
        points[i][0] = cloud[indice][0]
        points[i][1] = cloud[indice][1]
        points[i][2] = cloud[indice][2]

    return points


def preprocess(cloud):
    cloud = cloud[~(cloud == np.zeros(3)).all(axis=1)]
    filtered = filter_cloud_by_threshold(cloud, x_min=-1, x_max=1, y_min=-1.41, z_max=3.5)
    person = extract_person(filtered)
    return person


class ITOPDataset(data.Dataset):

    def __init__(self, root_path: str, train: bool = True):
        self.root_path = root_path
        self.train = train

        if self.is_cache_available():
            logger.info('Loading dataset from cache')
            self.__get_dataset_from_cache()
        else:
            logger.info('Generating dataset')
            self.__load_dataset()
            logger.info('Storing dataset to cache')
            self.__save_dataset_to_cache()

    def __getitem__(self, index: int):
        # downsampling
        indexies = np.random.choice(self.dataset[index].shape[0], size=EX_SIZE, replace=False)
        points = self.dataset[index][indexies]
        coords = self.real_world_coordinates[index].astype(np.float32)
        normalized_points, normalized_coords, mean, maxv = self.normalize(points, coords)

        return normalized_points, normalized_coords, mean, maxv

    @staticmethod
    def normalize(points, gt):
        mean = points.mean(axis=0)
        maxv = (points - mean).max(axis=0)
        points_normalized = (points - mean) / maxv
        coord_normalized = (gt - mean) / maxv

        return points_normalized, coord_normalized, mean, maxv

    @staticmethod
    def denormalize(points, mean, maxv):
        # batch
        if len(points.shape) == 3:
            return np.asarray([p * ma + me for p, ma, me in zip(points, maxv, mean)])
            pass
        elif len(points.shape) == 2:
            return points * maxv + mean
        else:
            print('Unknown shape for denormalization', len(points.shape))

    def __len__(self):
        return len(self.dataset)

    def is_cache_available(self):
        point_cloud_filename = 'train-dataset.pkl' if self.train else 'test-dataset.pkl'
        labels_filename = 'train-coords.pkl' if self.train else 'test-coords.pkl'

        return (os.path.exists(os.path.join(self.root_path, point_cloud_filename)) and
                os.path.exists(os.path.join(self.root_path, labels_filename)))

    def __get_dataset_from_cache(self):
        ids_filename = 'train-ids.pkl' if self.train else 'ids.pkl'
        point_cloud_filename = 'train-dataset.pkl' if self.train else 'test-dataset.pkl'
        coords_filename = 'train-coords.pkl' if self.train else 'test-coords.pkl'

        with open(os.path.join(self.root_path, point_cloud_filename), 'rb') as f:
            self.dataset = pickle.load(f)
        with open(os.path.join(self.root_path, coords_filename), 'rb') as f:
            self.real_world_coordinates = pickle.load(f)
        with open(os.path.join(self.root_path, ids_filename), 'rb') as f:
            self.ids = pickle.load(f)

    def __save_dataset_to_cache(self):
        ids_filename = 'train-ids.pkl' if self.train else 'ids.pkl'
        point_cloud_filename = 'train-dataset.pkl' if self.train else 'test-dataset.pkl'
        coords_filename = 'train-coords.pkl' if self.train else 'test-coords.pkl'

        with open(os.path.join(self.root_path, point_cloud_filename), 'wb') as f:
            pickle.dump(self.dataset, f)
        with open(os.path.join(self.root_path, coords_filename), 'wb') as f:
            pickle.dump(self.real_world_coordinates, f)
        with open(os.path.join(self.root_path, ids_filename), 'wb') as f:
            pickle.dump(self.ids, f)

    def __load_dataset(self):
        point_cloud_filename = 'ITOP_side_train_point_cloud.h5' if self.train else 'ITOP_side_test_point_cloud.h5'
        labels_filename = 'ITOP_side_train_labels.h5' if self.train else 'ITOP_side_test_labels.h5'

        logger.info(f"Loading dataset {point_cloud_filename}")
        point_cloud = h5py.File(os.path.join(self.root_path, point_cloud_filename), 'r')
        labels = h5py.File(os.path.join(self.root_path, labels_filename), 'r')

        is_valid, real_world_coordinates, ids = (
            np.asarray(labels.get('is_valid')).astype(bool),
            np.asarray(labels.get('real_world_coordinates')),
            np.asarray(labels.get('id'))
        )
        point_cloud = np.asarray(point_cloud.get('data'))[is_valid]
        logger.info(f"Dataset loaded")

        after_preprocessing = []
        pool = Pool(processes=multiprocessing.cpu_count())
        for result in tqdm(pool.imap(func=preprocess, iterable=point_cloud), total=len(point_cloud)):
            after_preprocessing.append(result)

        self.real_world_coordinates = real_world_coordinates[is_valid]
        self.ids = ids[is_valid]
        self.dataset = after_preprocessing


class ITOPDatasetOnline(data.Dataset):

    def __init__(self, root_path: str, train: bool = True):
        self.root_path = root_path
        self.train = train
        self.__load_dataset()

    def __getitem__(self, index: int):
        # downsampling
        point_cloud = self._point_cloud.get('data')
        is_valid, real_world_coordinates = (
            self._labels.get('is_valid'),
            self._labels.get('real_world_coordinates'),
        )
        if not is_valid:
            return self.__getitem__(index + 1)

        points, coords = self._preprocess(point_cloud[index], real_world_coordinates[index])
        return points, coords.flatten().astype(np.float32)

    def _preprocess(self, point_cloud, real_world_coordinates):
        filtered = filter_cloud_by_threshold(np.asarray(point_cloud), x_min=-1, x_max=1, y_min=-1.41, z_max=3.5)
        person_cloud = extract_person(filtered)
        coords = (real_world_coordinates / np.linalg.norm(real_world_coordinates)).flatten().astype(np.float32)

        indexies = np.random.choice(person_cloud.shape[0], size=EX_SIZE, replace=False)
        person_cloud = person_cloud[indexies]

        return person_cloud, coords

    def __len__(self):
        return self._point_cloud['data'].shape[0]

    def __load_dataset(self):
        point_cloud_filename = 'ITOP_side_train_point_cloud.h5' if self.train else 'ITOP_side_test_point_cloud.h5'
        labels_filename = 'ITOP_side_train_labels.h5' if self.train else 'ITOP_side_test_labels.h5'

        self._point_cloud = h5py.File(os.path.join(self.root_path, point_cloud_filename), 'r')
        self._labels = h5py.File(os.path.join(self.root_path, labels_filename), 'r')
