from sklearn.cluster import KMeans
from sklearn.cluster._kmeans import _check_sample_weight, _labels_inertia, _tolerance

def _init_centroids(X, n_clusters, init, random_state, x_squared_norms):
    """Compute the initial centroids."""
    inst = KMeans(n_clusters)
    return inst._init_centroids(X, x_squared_norms, init, random_state)


def _validate_center_shape(X, n_clusters, centers) -> None:
    """Check if centers is compatible with X and n_clusters."""
    inst = KMeans(n_clusters)
    return inst._validate_center_shape(X, centers)
