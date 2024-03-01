# import jax
# import jax.numpy as jnp
import numpy as np
    

def grid_subsampling(cloud : np.ndarray, voxel_size : float=.1):
    """
    Subsample point cloud by a grid method
    :param cloud: Nx3 array of the point cloud
    :param voxel_size: size of the grid (m)
    :return: subsampled cloud
    """
    vox_id = (cloud - np.min(cloud, axis=0)) // voxel_size
    non_empty_voxel_id, inverse, nb_points_in_voxels = np.unique( vox_id,
                                                                return_inverse=True,
                                                                return_counts=True,
                                                                axis=0)

    idx_in_voxels = np.argsort(inverse)
    subsampled_cloud = np.zeros((len(non_empty_voxel_id), 3), cloud.dtype)
    last_seen = 0
    for i in range(len(non_empty_voxel_id)):

        voxels = cloud[ idx_in_voxels[last_seen : last_seen + nb_points_in_voxels[i]]]
        subsampled_cloud[i] = voxels.mean(axis=0)

        last_seen += nb_points_in_voxels[i]

    return subsampled_cloud