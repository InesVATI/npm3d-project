from sklearn.metrics import jaccard_score, accuracy_score
from pathlib import Path
import pickle
import numpy as np
import time

benchmark_dict0 = {
    'DEFAULT' : {'Ground' : [],
                 'Building' : [],
                 'Traffic Signs' : [],
                 'Pedestrians' : [],
                 'Cars' : [],
                 'Vegetation' : [],
                 'Motorcycles' : [],
                 "Weighted IoU" : []},
    'W_HEIGHT_FEAT' : {'Ground' : [],
                       'Building' : [],
                       'Traffic Signs' : [],
                       'Pedestrians' : [],
                       'Cars' : [],
                       'Vegetation' : [],
                       'Motorcycles' : [],
                       'Weighted IoU' : []
                    },
    'NO_MULTI_SCALE' : {'Ground' : [],
                        'Building' : [],
                        'Traffic Signs' : [],
                        'Pedestrians' : [],
                        'Cars' : [],
                        'Vegetation' : [],
                        'Motorcycles' : [],
                        'Weighted IoU' : []
                        },  
    'KNN_NEIGH_DEF' : {'Ground' : [],
                       'Building' : [],
                       'Traffic Signs' : [],
                       'Pedestrians' : [],
                        'Cars' : [],
                        'Vegetation' : [],
                        'Motorcycles' : [],
                        'Weighted IoU' : []
                        }, 
}

def perform_classification(dataset, classifier):
    print('----- Randomly choose training points -----')
    t0 = time.time()
    training_features, training_labels, val_features, val_labels = dataset.get_training_val_data()
    t1 = time.time()
    print(f"Time to get random training points: {t1-t0} seconds")

    # perform classification
    classifier.fit(training_features, training_labels)

    val_pred = classifier.predict(val_features)
    
    return val_labels, val_pred
    
def repeat_method(dataset,
                  classifier,
                  method: str,
                  save_results_file: str,
                  nb_repeats: int = 10):
    root_folder = Path(__file__).parent.parent
    save_results_path = root_folder / '__results' / save_results_file
    if save_results_file.exists():
        
        with open(save_results_path, 'rb') as f:
            benchmark_dict = pickle.load(f)
    else:
        benchmark_dict = benchmark_dict0

    metrics_stats = np.zeros((nb_repeats, len(benchmark_dict['DEFAULT'])))
    for i in range(nb_repeats):
        val_labels, val_pred = perform_classification(dataset, classifier)
        class_score = jaccard_score(val_labels, val_pred, average=None)
        metrics_stats[i] = np.hstack( (class_score,
                                       jaccard_score(val_labels, val_pred, average='weighted')))
        
    for i, key in enumerate(benchmark_dict[method].keys()):
        benchmark_dict[method][key] = [metrics_stats[:, i].mean(), metrics_stats[:, i].std()]

    # save new results
    with open(save_results_path, 'wb') as f:
        pickle.dump(benchmark_dict, f)

    print(benchmark_dict)


