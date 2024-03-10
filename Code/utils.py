
import numpy as np
import configparser

def create_config_file(data = 'Cassette'):


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
            'nb_scales': '2',
            'ratio_radius': '2',
            'rho': '5',
            'neighborhood_def':'spherical',
            'features_saved_file' : 'None',
            'k' : '0',
            'add_height_feats' : 'False' 
        }

    config['W_HEIGHT_FEAT'] = config['DEFAULT']
    config['W_HEIGHT_FEAT']['add_height_feats'] = 'True'

    config['NO_MULTI_SCALE'] = config['DEFAULT']
    config['NO_MULTI_SCALE']['nb_scales'] = '1'
    config['NO_MULTI_SCALE']['features_saved_file'] = "Cassette_features_scale0.pkl" 

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


if __name__ == '__main__':
    create_config_file()

    # config = configparser.ConfigParser()
    # config.read('Code/cassette_config.ini')
    # method = config['DEFAULT']
   

    