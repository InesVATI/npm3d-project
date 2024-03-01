import numpy as np
import matplotlib.pyplot as plt 
from sklearn.neighbors import KDTree
from Code.process_data import get_dataset
from Code.utils import grid_subsampling
from typing import Optional, Tuple, Literal
import time 
import pickle

from pathlib import Path

root_folder = Path(__file__).parent.parent
figure_folder = root_folder / 'Figures'

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
                      plot_hist: bool = False,
                      figure_file : str = 'neigh_hist.png') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the local PCA of a set of points in a point cloud
    :param query_points: points to compute PCA on
    :param cloud_points: point cloud
    :param radius: radius of the spherical neighborhood to consider if neighborhood_def is "spherical"
    :param k: number of neighbors to consider if neighborhood_def is "knn"
    :param plot_hist: whether to plot the histogram of the number of neighbors
    :param figure_file: file path to save the histogram if plot_hist is True

    Returns: 
        query_neighborhoods: array of indices of the neighbors of each query point
        all_eigenvalues: (N, 3)-array
        all_eigenvectors: (N, 3, 3)-array
    """

    tree = KDTree(cloud_points)
    query_neighborhoods = (
        tree.query_radius(query_points, r=radius)
        if neighborhood_def.lower() == "spherical"
        else tree.query(query_points, k=k, return_distance=False)
        if neighborhood_def.lower() == "knn"
        else None
    )
    if query_neighborhoods is None:
        raise ValueError("Invalid neighborhood definition")

    # plot histogram of the number of neighbors
    if plot_hist:
        fig = plt.figure()
        neighborhood_sizes = [neighborhood.shape[0] for neighborhood in query_neighborhoods] 
        print(
            f"Average size of neighborhood: {np.mean(neighborhood_sizes)}\n",
            f"Minimal number of neighbors: {np.min(neighborhood_sizes)}, Maximum number of neighbors: {np.max(neighborhood_sizes)}"
        )
        bins_values, _, _ = plt.hist(neighborhood_sizes, bins="auto", rwidth=.8)
        plt.title(f"Histogram of the neighborhood sizes for {bins_values.shape[0]} bins")
        plt.xlabel('Number of neighbors')
        plt.ylabel('Number of neighborhoods')
        fig.savefig(figure_file)

    # compute PCA for each neighborhood
    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))
    for i in range(len(query_neighborhoods)):
        all_eigenvalues[i], all_eigenvectors[i] = PCA(cloud_points[query_neighborhoods[i]])

    return query_neighborhoods, all_eigenvalues, all_eigenvectors

def compute_moment_features(cloud: np.ndarray,
                            query_points: np.ndarray,
                            query_neighbor_indices: np.ndarray,
                            all_eigenvectors: np.ndarray) -> np.ndarray:

    # Absolute moments and vertical moments
    absolute_moments = np.zeros((query_points.shape[0], 6))
    vertical_moments = np.zeros((query_points.shape[0], 2))

    for i, point in enumerate(query_points):
        D = cloud[query_neighbor_indices[i]] - point

        absolute_moments[i] = np.array([np.abs(np.mean(D @ all_eigenvectors[i, :, 0]) ), # first order moment around smallest eigenvector
                                        np.abs(np.mean(D @ all_eigenvectors[i, :, 1]) ), # first order moment around middle eigenvector
                                        np.mean(D @ all_eigenvectors[i, :, 2] ), # first order moment around largest eigenvector
                                        np.mean( (D @ all_eigenvectors[i, :, 0])**2 ), # second order moment around smallest eigenvector
                                        np.mean( (D @ all_eigenvectors[i, :, 1])**2 ), # second order moment around middle eigenvector
                                        np.mean( (D @ all_eigenvectors[i, :, 2])**2 )  # second order moment around largest eigenvector
                                        ])
        vertical_moments[i] = np.array([np.mean(D[:, 2]), # first order vertical moment (around e_z)
                                        np.mean(D[:, 2]**2) # second order vertical moment (around e_z)
                                        ])
    return np.concatenate((absolute_moments, vertical_moments), axis=1)


def compute_Thomas_features(cloud: np.ndarray, query_points: np.ndarray,
                            query_neighbor_indices: np.ndarray,
                            all_eigenvalues: np.ndarray,
                            all_eigenvectors: np.ndarray) -> np.ndarray:
    """
    Compute the features from the eigenvalues and eigenvectors
    :param all_eigenvalues: (N, 3)-array
    :param all_eigenvectors: (N, 3, 3)-array
    Returns:
        features: (N, n)-array where N is the number of points to compute the features on and n is the number of features
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
    verticality_largest = np.abs(np.arcsin(all_eigenvectors[:, 2, 2]))

    neighborhood_sizes = np.array([[neighborhood.shape[0]] for neighborhood in query_neighbor_indices])

    # Absolute moments
    moment_features = compute_moment_features(cloud, query_points, query_neighbor_indices, all_eigenvectors)

    features = np.hstack((np.stack((sum_eigenvalues,
                                        omnivariance,
                                        eigenentropy,
                                        linearity,
                                        planarity,
                                        sphericity,
                                        change_curvature,
                                        verticality_smallest,
                                        verticality_largest), axis=1),
                                moment_features,
                                neighborhood_sizes))
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
    :param rho: ratio of radius and grid size
    Returns:
        features: (N, n*nb_scales)-array, where n is the number of features at each scale (here, 18)
    """
    features = np.zeros((query_points.shape[0], 18*nb_scales))
    pos_cloud = cloud - np.min(cloud, axis=0)

    for i in range(nb_scales):
        radius = r0*(ratio_radius**i) 

        if i == 0: # compute features on the original cloud
            query_neighbor_indices, all_eigenvalues, all_eigenvectors = compute_local_PCA(query_points, cloud, 
                                                                               neighborhood_def="spherical",
                                                                               radius=radius, 
                                                                               plot_hist=False, 
                                                                               figure_file=figure_folder / f'neigh_hist_scale{i}.png')
    
            features[:, i*18:(i+1)*18] = compute_Thomas_features(cloud,
                                                                 query_points,
                                                                 query_neighbor_indices,
                                                                 all_eigenvalues,
                                                                 all_eigenvectors)

        # compute features on subsampled clouds
        else:
            voxel_size = radius/rho
            non_empty_voxel_id, inverse, nb_pt_in_voxel_id = np.unique( pos_cloud // voxel_size,
                                                                        return_inverse=True,
                                                                        return_counts=True,
                                                                        axis=0)
            indices_in_voxels = np.argsort(inverse)
            subsampled_cloud = np.zeros((len(non_empty_voxel_id), 3), cloud.dtype)
            indices_original_query_pts = [] # list containing the indices in the original query points associated to each centroid
            # original_query_pts_not_visited = np.repeat(True, query_points.shape[0])
            new_query_points = np.empty((0, 3), dtype=query_points.dtype)
            last_seen = 0
            for j in range(len(non_empty_voxel_id)):
                voxels = cloud[indices_in_voxels[last_seen : last_seen + nb_pt_in_voxel_id[j]]]
                subsampled_cloud[j] = voxels.mean(axis=0)
                last_seen += nb_pt_in_voxel_id[j]
                # [original_query_pts_not_visited]
                idx = np.where( np.isin( query_points, voxels, assume_unique=True).all(axis=1))[0]
                if len(idx) > 0:
                    indices_original_query_pts.append( idx )
                    new_query_points = np.vstack((new_query_points, subsampled_cloud[j]))
                    # original_query_pts_not_visited[idx] = False

            query_neighbor_indices, all_eigenvalues, all_eigenvectors = compute_local_PCA(new_query_points, 
                                                                                          subsampled_cloud,
                                                                                          neighborhood_def="spherical",
                                                                                          radius=radius,
                                                                                          plot_hist=False,
                                                                                          figure_file=figure_folder / f'neigh_hist_scale{i}.png')
            feats = compute_Thomas_features(subsampled_cloud,
                                            new_query_points,
                                            query_neighbor_indices,
                                            all_eigenvalues,
                                            all_eigenvectors) # nb_new_query_points x 18
            
            for j, idx in enumerate(indices_original_query_pts):
                features[idx, i*18:(i+1)*18] = feats[j]
                
    return features
    

if __name__ == '__main__':

    cloud, label = get_dataset()

    # # select random training set and testing set
    rd_ind = np.random.choice(cloud.shape[0], 10, replace=False)
    training_points = cloud[rd_ind]
    # time0 = time.time()
    # query_neighbor_indices, all_eigenvalues, all_eigenvectors = compute_local_PCA(training_points, cloud, 
    #                                                                            neighborhood_def="spherical",
    #                                                                            radius=.1, plot_hist=False, 
    #                                                                            figure_file=figure_folder / 'neigh_hist_scale0.png')
    # t1 = time.time()
    # print('compute PCA took', t1-time0, 'seconds')
    
    # t0 = time.time()
    # features = compute_Thomas_features(cloud, training_points, query_neighbor_indices, all_eigenvalues, all_eigenvectors)
    # t1 = time.time()
    # print('compute Thomas features took', t1-t0, 'seconds')
    # print(' feature', features.shape)
    # print(np.min(features), np.max(features[:-1]))
    # print('cloud shape', cloud.shape)
    # t0 = time.time()
    # subsampled_cloud = grid_subsampling(cloud, voxel_size=1)
    # t1 = time.time()
    # print(f'Subsampling cloud of shape {cloud.shape} took {t1-t0:.3f} seconds.')
    # print(subsampled_cloud.shape)

    t0 = time.time()
    features = extract_multiscale_features(training_points,
                                           cloud, 
                                           r0=.1,
                                           nb_scales=2, 
                                           ratio_radius=2., 
                                           rho=5, 
                                           neighborhood_def="spherical")
    t1 = time.time()
    print('compute multiscale features took', t1-t0, 'seconds')
    print('features', features.shape)
    print('feature', features.nonzero())

    # save features
    with open(root_folder / '__data' / 'Cassette_features.pkl', 'wb') as f:
        pickle.dump(features, f)
        
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
