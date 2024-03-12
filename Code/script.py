import numpy as np
from ply import write_ply
import time
from pathlib import Path
from utils import grid_subsampling
from process_data import get_CassetteDataset
from multiscale_features import extract_multiscale_features
from classification import repeat_method, perform_classification
from datasets import CassetteDataset, MiniParisLilleDataset
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import jaccard_score
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
    
    if False :
        method = 'DEFAULT'
        dataset = CassetteDataset(num_per_class=1000,
                                  data_folder=root_folder / '__data',
                                  method=method)
        # classifier = RandomForestClassifier(n_estimators=150,
        #                                     criterion="gini",
        #                                     class_weight="balanced"
        #                                     ) 
        classifier = HistGradientBoostingClassifier(max_iter=150, 
                                                    class_weight="balanced") 
        val_labels, val_pred = perform_classification(dataset, classifier)
        class_score = jaccard_score(val_labels, val_pred, average=None)
        print("method", method)
        print('class_score', class_score)
        print('jaccard_score', jaccard_score(val_labels, val_pred, average='weighted'))

    if False :
        for method in ['W_HEIGHT_FEAT'] :
            dataset = CassetteDataset(num_per_class=1000,
                                    data_folder=root_folder / '__data',
                                    method=method)
            classifier = RandomForestClassifier(n_estimators=150,
                                                criterion="gini",
                                                class_weight="balanced"
                                                ) 
            # classifier = HistGradientBoostingClassifier(max_iter=150, 
            #                                             class_weight="balanced") 
            repeat_method(dataset=dataset,
                        classifier=classifier,
                        method=method, 
                        nb_repeats=10,
                        save_results_file=root_folder / '__results' / 'Cassette_benchmarkRF_results.pkl'
                        )
            
    if False:
        with open(root_folder / '__results' / 'Cassette_benchmarkBoosting_results.pkl', 'rb') as f:
            results = pickle.load(f)
            print('results', results)
    if True : # train on MiniLille and predict on MiniParis
        data_path = root_folder/'__data'
        # dataset = MiniParisLilleDataset(num_per_class=5000,
        #                                 data_folder=data_path,
        #                                 method='DEFAULT')
        
        # t0 = time.time()
        # training_features, training_labels = dataset.get_training_data()
        # test_features

        # # perform classification
        # classifier = RandomForestClassifier(n_estimators=150,
        #                                         criterion="gini",
        #                                         class_weight="balanced"
        #                                         ) 
        # classifier.fit(training_features, training_labels)

        # val_pred = classifier.predict(val_features)

        # t0 = time.time()
        # features = dataset.extract_test_features(f'{root_folder}/__data/test')
        # t1 = time.time()
        # print('compute multiscale features on MiniParis took', t1-t0, 'seconds')




    