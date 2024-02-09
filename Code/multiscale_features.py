import numpy as np
import matplotlib.pyplot as plt 
from sklearn.neighbors import KDTree
from inspect_data import get_dataset
from typing import Optional, Tuple, Literal
import time 

def PCA(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the PCA of a set of points
    :param points: (N, 3)-array
    Returns: 
        eigenvalues: (3,)-array
        eigenvectors: (3, 3)-array
    """
    centered_points = points - points.mean(axis=0)
    covariance_matrix = centered_points.T @ centered_points / points.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    return eigenvalues, eigenvectors

def compute_local_PCA(query_points : np.ndarray,
                      cloud_points: np.ndarray,
                      neighborhood_def: Literal["spherical", "knn"] = "spherical",
                      radius : Optional[float]=None,
                      k: Optional[int]=None,
                      plot_hist: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the local PCA of a set of points in a point cloud
    :param query_points: points to compute PCA on
    :param cloud_points: point cloud
    :param radius: radius of the spherical neighborhood to consider if neighborhood_def is "spherical"
    :param k: number of neighbors to consider if neighborhood_def is "knn"
    
    Returns: 
        all_eigenvalues: (N, 3)-array
        all_eigenvectors: (N, 3, 3)-array
    """

    tree = KDTree(cloud_points)
    query_neighborhoods = (
        tree.query_radius(query_points, r=radius)
        if neighborhood_def == "spherical"
        else tree.query(query_points, k=k, return_distance=False)
        if neighborhood_def == "knn"
        else None
    )
    if query_neighborhoods is None:
        raise ValueError("Invalid neighborhood definition")

    # plot histogram of the number of neighbors
    if plot_hist:
        neighborhood_sizes = [neighborhood.shape[0] for neighborhood in query_neighborhoods] 
        print(
            f"Average size of neighborhood: {neighborhood_sizes.mean()}\n",
            f"Minimal number of neighbors: {np.min(neighborhood_sizes)}, Maximum number of neighbors: {np.max(neighborhood_sizes)}"
        )
        bins_values, _, _ = plt.hist(neighborhood_sizes, bins="auto")
        plt.title(f"Histogram of the neighborhood sizes for {bins_values.shape[0]} bins")
        plt.xlabel('Number of neighbors')
        plt.ylabel('Number of neighborhoods')

    # compute PCA for each neighborhood
    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))
    for i in range(len(query_neighborhoods)):
        all_eigenvalues[i], all_eigenvectors[i] = PCA(cloud_points[query_neighborhoods[i]])

    return all_eigenvalues, all_eigenvectors

def compute_features(all_eigenvalues: np.ndarray,
                     all_eigenvectors: np.ndarray) -> np.ndarray:
    """
    Compute the features from the eigenvalues and eigenvectors
    :param all_eigenvalues: (N, 3)-array
    :param all_eigenvectors: (N, 3, 3)-array
    Returns:
        features: (N, 11)-array
    """
    eps = 1e-8
    sum_eigenvalues = all_eigenvalues.sum(axis=1)
    omnivariance = np.prod(all_eigenvalues, axis=1)**(1/3)
    eigenentropy = -np.sum(all_eigenvalues * np.log(all_eigenvalues + eps), axis=1)
    linearity = 1 - all_eigenvalues[:, 1] / (all_eigenvalues[:, 2] + eps)
    planarity = (all_eigenvalues[:, 1] - all_eigenvalues[:, 0]) / (all_eigenvalues[:, 2] + eps)
    sphericity = all_eigenvalues[:, 0] / (all_eigenvalues[:, 2] + eps)
    change_curvature = all_eigenvalues[:, 2] / (sum_eigenvalues + eps)

    verticality_smallest = np.abs(np.arcsin(all_eigenvectors[:, 2, 0]))
    verticlaity_largest = np.abs(np.arcsin(all_eigenvectors[:, 2, 2]))
    


    features = np.concatenate((all_eigenvalues, all_eigenvectors.reshape(-1, 9)), axis=1)
    return features

def extract_multiscale_features(query_points: np.ndarray,
                                cloud: np.ndarray,
                                r0: float = .1,
                                nb_scales: int = 8,
                                ratio_radius: float = 2.,
                                rho: float = 5,
                                neighborhood_def: Literal["spherical", "knn"] = "spherical"):
    """
    Extract multiscale features from a subset of points (training or testing set) in point cloud
    :param r0: smallest radius for spherical neighborhood
    :param nb_scales: number of scales
    Returns:
        features: (N, 9*nb_scales)-array
    """
    pass

def grid_subsampling(cloud, voxel_size=.1):
    """
    Subsample point cloud by a grid method
    :param cloud: Nx3 array of the point cloud
    :param cell_size: size of the grid
    :return: subsampled cloud
    """
    
    non_empty_voxel_id, inverse, nb_points_in_voxels = np.unique( (cloud - np.min(cloud, axis=0)) // voxel_size,
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
    


if __name__ == '__main__':
    check("Other")
    # cloud, label = get_dataset()

    # t0 = time.time()
    # subsampled_cloud = grid_subsampling(cloud, voxel_size=1)
    # t1 = time.time()
    # print(f'Subsampling cloud of shape {cloud.shape} took {t1-t0:.3f} seconds.')
    # print(subsampled_cloud.shape)

    # arr = np.array([[1, 2, 3, 2, 1]])
    # unique, inverse, counts = np.unique(arr, return_inverse=True, return_counts=True, axis=1)
    # print('unique')
    # print(unique)
    # print('inverse', inverse)
    # print('argsort inverse', np.argsort(inverse))
    # print('reconstruction')
    # print(unique[:, inverse])
    # print('counts\n', counts)

    # arr = np.array([6,4,22, 6, 22, 6])
    # print(np.bincount(arr))
