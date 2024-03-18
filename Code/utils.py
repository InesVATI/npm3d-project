
import numpy as np
import configparser
from texttable import Texttable
import latextable
from typing import Literal

def create_config_file(data : Literal['Cassette', 'Mini'] = 'Cassette'):


    config = configparser.ConfigParser()

    if data.lower() == 'cassette':
        config['DEFAULT'] = {
            'r0' : '0.5',
            'nb_scales': '4',
            'ratio_radius': '2',
            'rho': '5',
            'neighborhood_def':'spherical',
            'features_saved_file' : 'None',
            'k' : '0',
            'add_height_feats' : 'False' 
        }
    elif data.lower() == 'mini':
        config['DEFAULT'] = {
            'r0' : '0.8',
            'nb_scales': '3',
            'ratio_radius': '2',
            'rho': '5',
            'neighborhood_def':'spherical',
            'features_saved_file' : 'trainLille_features.pkl',
            'k' : '0',
            'add_height_feats' : 'False' 
        }

    config['W_HEIGHT_FEAT'] = config['DEFAULT']
    config['W_HEIGHT_FEAT']['add_height_feats'] = 'True'

    config['NO_MULTI_SCALE'] = config['DEFAULT']
    config['NO_MULTI_SCALE']['nb_scales'] = '1' 

    config['KNN_NEIGH_DEF'] = config['DEFAULT']
    config['KNN_NEIGH_DEF']['neighborhood_def'] = 'knn'
    config['KNN_NEIGH_DEF']['k'] = '20'


    with open(f'Code/{data}_config.ini', 'w') as configfile:
        config.write(configfile)



def grid_subsampling(cloud : np.ndarray, 
                     label: np.ndarray,
                     voxel_size : float=.1):
    """
    Subsample point cloud by a grid method
    :param cloud: Nx3 array of the point cloud
    :param voxel_size: size of the grid (m)
    :return: subsampled cloud
    """
    vox_id = (cloud - np.min(cloud, axis=0)) // voxel_size
    non_empty_voxel_id, inverse, nb_points_in_voxels = np.unique(vox_id,
                                                                return_inverse=True,
                                                                return_counts=True,
                                                                axis=0)

    idx_in_voxels = np.argsort(inverse)
    subsampled_cloud = np.zeros((len(non_empty_voxel_id), 3), cloud.dtype)
    subsampled_label = np.zeros((len(non_empty_voxel_id), 1), label.dtype)
    label = label.astype(np.int32)
    last_seen = 0
    for i in range(len(non_empty_voxel_id)):
        ind_pts = idx_in_voxels[last_seen : last_seen + nb_points_in_voxels[i]]
        voxels = cloud[ ind_pts ]
        label_votes = np.bincount(label[ind_pts])
        subsampled_cloud[i] = voxels.mean(axis=0)
        subsampled_label[i] = np.argmax(label_votes)
        last_seen += nb_points_in_voxels[i]

    return subsampled_cloud, subsampled_label

def table_label_size(labels : np.ndarray, label_names : dict,
                     caption_table : str = 'Class size'):
    """
    Create a table with the size of each label
    :param labels: list of labels
    :param labels_names: list of labels names
    :return: table
    """
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.add_rows([['Label', 'Size']])
    for key in list(label_names.keys())[1:]:
        table.add_row([label_names[key], np.sum(labels == key)])
    
    print(latextable.draw_latex(
        table,
        caption=caption_table,
        label='tab:label_size'
    ))

def get_table_benchmark(benchmark_dict : dict, 
                        caption_table : str = 'Benchmark results',
                        label_table : str = 'tab:benchmark'):
    """
    Create a table with the benchmark results
    :param benchmark_dict: dictionary with the benchmark results
    :return: table
    """
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.add_rows([['Method', 'Ground', 'Building', 'Traffic Signs', 'Pedestrians', 'Cars', 'Vegetation', 'Motorcycles', 'Weighted IoU']])
    for key in benchmark_dict.keys():
        values = benchmark_dict[key]
        table.add_row([key.replace('_', ' '), f"${values['Ground'][0]*100:.2f} \pm {values['Ground'][1]*100:.2f}$",
                       f"${values['Building'][0]*100:.2f} \pm {values['Building'][1]*100:.2f}$",
                       f"${values['Traffic Signs'][0]*100:.2f} \pm {values['Traffic Signs'][1]*100:.2f}$",
                       f"${values['Pedestrians'][0]*100:.2f} \pm {values['Pedestrians'][1]*100:.2f}$",
                       f"${values['Cars'][0]*100:.2f} \pm {values['Cars'][1]*100:.2f}$",
                       f"${values['Vegetation'][0]*100:.2f} \pm {values['Vegetation'][1]*100:.2f}$",
                       f"${values['Motorcycles'][0]*100:.2f} \pm {values['Motorcycles'][1]*100:.2f}$",
                       f"${values['Weighted IoU'][0]*100:.2f} \pm {values['Weighted IoU'][1]*100:.2f}$"
                       ])
    
    print(latextable.draw_latex(
        table,
        caption=caption_table,
        label=label_table
    ))
