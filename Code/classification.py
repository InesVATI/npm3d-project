from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, jaccard_score
from Code.ply import read_ply
from Code.multiscale_features import extract_multiscale_features
import numpy as np
import time
from os import listdir
from os.path import join

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

        return train_features, train_labels, val_features, val_labels
    



if __name__ == "__main__":
    data_path = '__data/training'
    dataset = MiniParisLilleDataset(num_per_class=500, nb_scales=2)
    
    print('Extracting features...')
    t0 = time.time()
    train_features, train_labels, val_features, val_labels = dataset.extract_train_val_features(data_path)
    t1 = time.time()
    print('Done in {:.1f}s'.format(t1 - t0))

    print(f'train feat {train_features.shape} train labels {train_labels.shape} val feat {val_features.shape} val labels {val_labels.shape}')

    print('Training the classifier...')
    t0 = time.time()
    clf = RandomForestClassifier()

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
    print('Done in {:.1f}s'.format(t1 - t0))
    print('Accuracy: {:.1f}%'.format(100 * accuracy_score(val_labels, val_pred)))
    print('Jaccard index: {:.1f}%'.format(100 * jaccard_score(val_labels, val_pred, average='micro'))) 



            




