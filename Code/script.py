import time
from pathlib import Path
from classification import repeat_method, perform_classification
from datasets import CassetteDataset, MiniParisLilleDataset
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import jaccard_score

if __name__ == '__main__':
    root_folder = Path(__file__).parent.parent
    
    if True : # Comparison RandomForest vs HistGradientBoosting
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

    if True : # Benchmarking
        for method in ['DEFAULT', 'W_HEIGHT_FEAT', 'NO_MULTI_SCALE', 'KNN_NEIGH_DEF'] :
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
            

    if True : # train on MiniLille and predict on MiniParis
        data_path = root_folder/'__data'
        dataset = MiniParisLilleDataset(num_per_class=100000,
                                        data_folder=data_path,
                                        method='DEFAULT',
                                        num_test_samples_per_class=80000)
        
        t0 = time.time()
        training_features, training_labels, val_features, val_labels = dataset.get_training_val_data()
        t1 = time.time()
        print(f"Time to get random train and val points: {t1-t0} seconds")


        # perform classification
        t0 = time.time()
        classifier = RandomForestClassifier(n_estimators=150,
                                            criterion="gini",
                                            class_weight="balanced"
                                            ) 
        classifier.fit(training_features, training_labels)
        t1 = time.time()
        print(f"Time to train classifier: {t1-t0} seconds")
        t0 = time.time()
        val_pred = classifier.predict(val_features)
        t1 = time.time()
        print(f"Time to predict labels: {t1-t0} seconds")
        class_score = jaccard_score(val_labels, val_pred, average=None)
        print('class_score', class_score)
        print('weighted jaccard_score', jaccard_score(val_labels, val_pred, average='weighted'))





    