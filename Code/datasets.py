from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, jaccard_score
from Code.ply import read_ply
from Code.multiscale_features import extract_multiscale_features
import numpy as np
import pickle
import time
from os import listdir
from os.path import join, exists
import configparser

class CassetteDataset:
    def __init__(self, num_per_class:int=1000,
                 data_folder = '__data',
                 method:str='DEFAULT') -> None:
        self.data_folder = data_folder
        self.num_per_class = num_per_class
        config = configparser.ConfigParser()
        config.read('Code/cassette_config.ini')
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

    def _extract_features(self):
        """ Choose random points in point cloud from each class and extract features """
        
        cloud_path = self.data_folder / 'Cassette_Cloud_forClassification_subsampled.ply'
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

    def get_training_data(self):
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


class MiniParisLilleDataset:
    def __init__(self, num_per_class=500, nb_scales:int=2) -> None:
        self.num_per_class = num_per_class
        self.r0 = .8
        self.num_scales = nb_scales
        self.ratio_radius = 2
        self.num_geom_feats = 18
        self.rho = 5

        label_names = {0: 'Unclassified',
                       1: 'Ground',
                       2: 'Building',
                       3: 'Poles',
                       4: 'Pedestrians',
                       5: 'Cars',
                       6: 'Vegetation'}
        self.labels_id = label_names.keys()


    def extract_train_val_features(self, path):
        ply__files = [f for f in listdir(path) if f.endswith('.ply')]
        train_features = np.empty((0, self.num_scales * self.num_geom_feats))
        train_labels = np.empty((0,))
        val_features = np.empty((0, self.num_scales * self.num_geom_feats))
        val_labels = np.empty((0,))

        # loop over training clouds
        for i, file in enumerate(ply__files):
            # read the cloud
            cloud_ply = read_ply(join(path, file))
            cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
            labels = cloud_ply['class']
            training_inds = np.empty(0, dtype=np.int32)
            val_inds = np.empty(0, dtype=np.int32)

            for label in self.labels_id:
                if label == 0:
                    continue

                label_inds = np.where(labels == label)[0]
                # if not enough points choose 95% for training and 5% for validation
                if len(label_inds) <= self.num_per_class:
                    random_choice = np.random.choice(len(label_inds), int(len(label_inds) * .95), replace=False)
                else:
                    random_choice = np.random.choice(len(label_inds), self.num_per_class, replace=False)

                training_inds = np.hstack((training_inds, label_inds[random_choice]))
                leftover_ind = np.delete(label_inds, random_choice)
                if len(leftover_ind) > 100:
                    val_inds = np.hstack((val_inds, leftover_ind[:100]))
                else:
                    val_inds = np.hstack((val_inds, leftover_ind))


            # extract features
            all_query_inds = np.hstack((training_inds, val_inds))
            all_features = extract_multiscale_features(cloud[all_query_inds, :], 
                                                       cloud,
                                                       r0=self.r0,
                                                       nb_scales=self.num_scales,
                                                       ratio_radius=self.ratio_radius,
                                                       rho=self.rho,
                                                       )
            
            # Concatenate labels 
            train_features = np.vstack((train_features, all_features[:len(training_inds), :]))
            val_features = np.vstack((val_features, all_features[len(training_inds):, :]))
            
            train_labels = np.hstack((train_labels, labels[training_inds]))
            val_labels = np.hstack((val_labels, labels[val_inds]))
            if i ==2:
                break

        # save features and labels
        train_features_file = join(path, 'train_features.pkl')
        val_features_file = join(path, 'val_features.pkl')
        with open(train_features_file, 'wb') as f:
            pickle.dump(train_features, f)
        with open(val_features_file, 'wb') as f:
            pickle.dump(val_features, f)
        train_labels_file = join(path, 'train_labels.npy')
        np.save(train_labels_file, train_labels)
        val_labels_file = join(path, 'val_labels.npy')
        np.save(val_labels_file, val_labels)
        

        return train_features, train_labels, val_features, val_labels
    
    def extract_test_features(self, path):
        ply__files = [f for f in listdir(path) if f.endswith('.ply')]
        
        test_features = np.empty((0, self.num_scales * self.num_geom_feats))

        for file in ply__files:

            cloud_ply = read_ply(join(path, file))
            cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

            feature_file = file[:-4] + '_features.npy'
            feature_file = join(path, feature_file)

            if exists(feature_file):
                features = np.load(feature_file)

            else:
                features = extract_multiscale_features(cloud, 
                                                       cloud,
                                                       r0=self.r0,
                                                       nb_scales=self.num_scales,
                                                       ratio_radius=self.ratio_radius,
                                                       rho=self.rho,
                                                       )
                np.save(feature_file, features) 
            test_features = np.vstack((test_features, features))

        return test_features
    

if __name__ == "__main__":
    data_path = '__data/training'
    dataset = MiniParisLilleDataset(num_per_class=100000, nb_scales=3)
    
    print('Extracting features...')
    t0 = time.time()
    train_features, train_labels, val_features, val_labels = dataset.extract_train_val_features(data_path)
    t1 = time.time()
    print('Done in {:.1f}s'.format(t1 - t0))

    print(f'train feat {train_features.shape} train labels {train_labels.shape} val feat {val_features.shape} val labels {val_labels.shape}')

    print('Training the classifier...')
    t0 = time.time()
    clf = RandomForestClassifier(n_estimators=100,
                                 criterion="gini", max_depth=30)

    # scale the features ?
    clf.fit(train_features, train_labels)
    t1 = time.time()
    print('Done in {:.1f}s'.format(t1 - t0))


    # print('Check the accuracy on train set...')
    # train_pred = clf.predict(train_features[:20])
    # print(f'train pred {train_pred.shape}')
    # print('Accuracy: {:.1f}%'.format(100 * accuracy_score(train_labels[:20], train_pred)))
    # print('Jaccard index: {:.1f}%'.format(100 * jaccard_score(train_labels[:20], train_pred, average='micro'))) # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html


    print('Validation...')
    t0 = time.time()
    val_pred = clf.predict(val_features)
    t1 = time.time()
    class_score = jaccard_score(val_labels, val_pred, average=None) # 7
    print(f'Jaccard score per class {class_score}')
    print('Done in {:.1f}s'.format(t1 - t0))
    print('Accuracy: {:.1f}%'.format(100 * accuracy_score(val_labels, val_pred)))
    print('Jaccard index: {:.1f}%'.format(100 * jaccard_score(val_labels, val_pred, average='micro'))) 

    print('Testing...')
    t0 = time.time()
    test_features = dataset.extract_test_features('__data/test')
    t1 = time.time()
    print('Features extraction Done in {:.1f}s'.format(t1 - t0))
    print('Save predictions...')
    t0 = time.time()
    test_pred = clf.predict(test_features)
    np.savetxt('__data/MiniDijon9.txt', test_pred, fmt='%d')
    t1 = time.time()
    print('Done in {:.1f}s'.format(t1 - t0))
    print(f'test pred {test_pred.shape}')





            




