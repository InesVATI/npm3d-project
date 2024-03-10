import numpy as np
from ply import write_ply
import time
from pathlib import Path
from utils import grid_subsampling
from process_data import get_CassetteDataset
from multiscale_features import extract_multiscale_features
from classification import repeat_method
from datasets import CassetteDataset
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import pickle

if __name__ == '__main__':
    root_folder = Path(__file__).parent.parent
    if False: # save features on all cloud

        cloud, label = get_CassetteDataset(filename=root_folder / '__data' / 'Cassette_Cloud_forClassification_subsampled.ply')
        N = cloud.shape[0]
        print('cloud', cloud.shape)
        nonlabel_ind = label == 0
        
        # select random training set and testing set among point with label different from 0
        # rd_ind = np.random.choice(np.arange(N)[~nonlabel_ind], 40, replace=False)
        training_points = cloud[~nonlabel_ind]
        labels_training = label[~nonlabel_ind]

        t0 = time.time()
        features = extract_multiscale_features(training_points,
                                            cloud, 
                                            r0=.5,
                                            nb_scales=1, 
                                            ratio_radius=2., 
                                            rho=5, 
                                            neighborhood_def="spherical")
        t1 = time.time()
        print('compute multiscale features took', t1-t0, 'seconds')
        # print('features', features.shape)
        # print('feature', features.nonzero())

        # save features and corresponding labels
        with open(root_folder / '__data' / 'Cassette_features_scale0.pkl', 'wb') as f:
            pickle.dump(features, f)
        
        with open(root_folder / '__data' / 'Cassette_labels.pkl', 'wb') as f:
            pickle.dump(labels_training, f)
    
    if True :
        method = 'KNN_NEIGH_DEF'
        dataset = CassetteDataset(num_per_class=1000,
                                  data_folder=root_folder / '__data',
                                  method=method)
        classifier = RandomForestClassifier(n_estimators=150,
                                            criterion="gini",
                                            max_depth=30) # class_weight="balanced"
        # classifier = HistGradientBoostingClassifier(max_iter=100, 
        #                                             max_depth=30,
        #                                             class_weight="balanced") 
        repeat_method(dataset=dataset,
                      classifier=classifier,
                      method=method, 
                      nb_repeats=10,
                      save_results_file=root_folder / '__results' / 'Cassette_benchmark_results.pkl'
                      )

    