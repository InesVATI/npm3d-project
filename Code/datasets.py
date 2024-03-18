from ply import read_ply
from multiscale_features import extract_multiscale_features
import numpy as np
import pickle
import os
import configparser


class PointDataset:
    def __init__(self, num_per_class:int=100000,
                 data_folder : str = '__data') -> None:
        self.data_folder = data_folder
        self.num_per_class = num_per_class

    def _get_random_inds(self, labels: np.ndarray):
        pass

    def get_training_val_data(self):
        pass

    def _extract_features(self, cloud_path:str):
        pass

class CassetteDataset(PointDataset):
    def __init__(self, num_per_class:int=1000,
                 data_folder : str = '__data',
                 method:str='DEFAULT') -> None:
        super().__init__(num_per_class, data_folder)

        config = configparser.ConfigParser()
        config.read(f'{os.path.dirname(data_folder)}/Code/cassette_config.ini')
        self.method_config = config[method]

        self.label_names = {0: 'Unclassified',
                            1: 'Ground',
                            2: 'Building', # Facade
                            3: 'Traffic Signs',
                            4: 'Pedestrians',
                            5: 'Cars',
                            6: 'Vegetation',
                            7: 'Motorcycles',}
        
    def _get_random_inds(self, labels: np.ndarray):
        training_inds = np.empty(0, dtype=np.int32)
        val_inds = np.empty(0, dtype=np.int32)

        for label in self.label_names.keys():
            if label == 0:
                continue

            label_inds = np.where(labels == label)[0]
            # if not enough points choose 95% for training and 5% for validation
            if len(label_inds) <= self.num_per_class:
                random_choice = np.random.choice(len(label_inds), int(len(label_inds) * .95), replace=False)
            else:
                random_choice = np.random.choice(len(label_inds), self.num_per_class, replace=False)

            training_inds = np.hstack((training_inds, label_inds[random_choice]))
            leftover_ind = np.delete(label_inds, random_choice) # remaining points are used for validation
            if len(leftover_ind) > 200:
                val_inds = np.hstack((val_inds, leftover_ind[:200]))
            else:
                val_inds = np.hstack((val_inds, leftover_ind))

        return training_inds, val_inds

    def _extract_features(self, cloud_path:str='Cassette_Cloud_forClassification_subsampled.ply'):
        """ Choose random points in point cloud from each class and extract features """
        
        cloud_path = f'{self.data_folder}/{cloud_path}'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        labels = cloud_ply['class']

        training_inds, val_inds = self._get_random_inds(labels)

        # extract features
        all_query_inds = np.hstack((training_inds, val_inds))
        all_features = extract_multiscale_features(cloud[all_query_inds, :], 
                                                    cloud,
                                                    r0=self.method_config.getfloat('r0'),
                                                    nb_scales=self.method_config.getint('nb_scales'),
                                                    ratio_radius=self.method_config.getfloat('ratio_radius'),
                                                    rho=self.method_config.getfloat('rho'),
                                                    k=self.method_config.getint('k'),
                                                    neighborhood_def=self.method_config['neighborhood_def'],
                                                    add_height_feats=self.method_config.getboolean('add_height_feats')
                                                    )
        
        train_features = all_features[:len(training_inds), :]
        val_features = all_features[len(training_inds):, :]
        
        train_labels = labels[training_inds]
        val_labels =  labels[val_inds]

        return train_features, train_labels, val_features, val_labels

    def get_training_val_data(self):
        """ Choose random vector of features from each class """

        if self.method_config['features_saved_file'].endswith('.pkl') :
            with open(self.data_folder/self.method_config['features_saved_file'], 'rb') as f:
                all_features = pickle.load(f)
            with open(self.data_folder/'Cassette_labels.pkl', 'rb') as f:
                labels = pickle.load(f)

            training_inds, val_inds = self._get_random_inds(labels)

            train_features = all_features[training_inds]
            train_labels = labels[training_inds]
            val_features = all_features[val_inds]
            val_labels = labels[val_inds]
            
        else :
            train_features, train_labels, val_features, val_labels = self._extract_features()

        return train_features, train_labels, val_features, val_labels


class MiniParisLilleDataset(PointDataset):

    def __init__(self, num_per_class:int=10000,
                 data_folder : str = '__data',
                 method: str = 'DEFAULT',
                 num_test_samples_per_class: int = 10000) -> None:
        super().__init__(num_per_class, data_folder)
        config = configparser.ConfigParser()
        config.read(f'{os.path.dirname(data_folder)}/Code/Mini_config.ini')
        self.method_config = config[method]

        self.label_names = {0: 'Unclassified',
                            1: 'Ground',
                            2: 'Building',
                            3: 'Poles',
                            4: 'Pedestrians',
                            5: 'Cars',
                            6: 'Vegetation'}
        self.num_test_samples = num_test_samples_per_class
        
    def _get_random_inds(self, labels: np.ndarray):
        training_inds = np.empty(0, dtype=np.int32)

        for label in self.label_names.keys():
            if label == 0:
                continue

            label_inds = np.where(labels == label)[0]
            # if not enough points take every one for training
            if len(label_inds) <= self.num_per_class:
                random_choice = np.arange(len(label_inds))
            else:
                random_choice = np.random.choice(len(label_inds), self.num_per_class, replace=False)

            training_inds = np.hstack((training_inds, label_inds[random_choice]))

        return training_inds
    

    def get_training_val_data(self):
        """ Choose random vector of features from each class """
        if self.method_config['features_saved_file'].endswith('.pkl') :
            with open(self.data_folder/'training'/self.method_config['features_saved_file'], 'rb') as f:
                all_features = pickle.load(f)
            with open(self.data_folder/'training'/'trainLille_labels.npy', 'rb') as f:
                labels = np.load(f)
            training_inds = self._get_random_inds(labels)

            train_features, train_labels = all_features[training_inds], labels[training_inds]
        
        else :
            train_features, train_labels = self._extract_features(cloud_folder='training', train=True)

        test_saved_files = [f for f in sorted(os.listdir(self.data_folder/ 'test')) if f.endswith('.npy')]
        if len(test_saved_files) == 2:
            with open(self.data_folder/'test'/test_saved_files[0], 'rb') as f:
                val_features = pickle.load(f)
            with open(self.data_folder/'test'/test_saved_files[1], 'rb') as f:
                val_labels = pickle.load(f)

        else:
            val_features, val_labels = self._extract_features(cloud_folder='test', train=False)

            # saved features
            test_feature_files = self.data_folder/'test'/'MiniParis1_features.npy'
            test_label_files = self.data_folder/'test'/'MiniParis1_labels.npy'
            with open(test_feature_files, 'wb') as f:
                np.save(f, val_features)
            with open(test_label_files, 'wb') as f:
                np.save(f, val_labels)

        return train_features, train_labels, val_features, val_labels


    def _extract_features(self, cloud_folder:str = 'training', train: bool =True):
        ply__files = [f for f in os.listdir(self.data_folder/cloud_folder) if f.endswith('.ply')]
        n_feats = 21 if self.method_config.getboolean('add_height_feats') else 18
        n_samples = self.num_per_class if train else self.num_test_samples
        query_features = np.empty((0, self.method_config.getint('nb_scales') * n_feats))
        query_labels = np.empty((0,))

        # loop over training clouds
        for file in ply__files:
            # read the cloud
            cloud_ply = read_ply(os.path.join(self.data_folder, cloud_folder, file))
            cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
            labels = cloud_ply['class']
            query_inds = np.empty(0, dtype=np.int32)

            for label in self.label_names.keys():
                if label == 0:
                    continue

                label_inds = np.where(labels == label)[0]
                # if not enough points select every one 
                if len(label_inds) <= n_samples:
                    random_choice = np.arange(len(label_inds))
                else:
                    random_choice = np.random.choice(len(label_inds), n_samples, replace=False)

                query_inds = np.hstack((query_inds, label_inds[random_choice]))

            # extract features
            features = extract_multiscale_features(cloud[query_inds, :],
                                                   cloud,
                                                   r0=self.method_config.getfloat('r0'),
                                                   nb_scales=self.method_config.getint('nb_scales'),
                                                   ratio_radius=self.method_config.getfloat('ratio_radius'),
                                                   rho=self.method_config.getfloat('rho'),
                                                   k=self.method_config.getint('k'),
                                                   neighborhood_def=self.method_config['neighborhood_def'],
                                                   add_height_feats=self.method_config.getboolean('add_height_feats')
                                                   )
        
            # Concatenate labels 
            query_features = np.vstack((query_features, features))
            query_labels = np.hstack((query_labels, labels[query_inds]))
    
        return query_features, query_labels
    




            




