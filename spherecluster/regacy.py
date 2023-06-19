import numpy
from mlinsights.mlmodel._kmeans_022 import (
    _labels_inertia_precompute_dense,
    _labels_inertia_skl,
)
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.cluster._kmeans import _check_sample_weight


def _labels_inertia(
    X,
    sample_weight,
    x_squared_norms,
    centers,
    precompute_distances=True,
    distances=None,
):
    """E step of the K-means EM algorithm.

    Compute the labels and the inertia of the given samples and centers.
    This will compute the distances in-place.
    """

    if precompute_distances:
        return _labels_inertia_skl(
            X,
            sample_weight=sample_weight,
            centers=centers,
            x_squared_norms=x_squared_norms,
        )

    sample_weight = _check_sample_weight(sample_weight, X)
    # set the default value of centers to -1 to be able to detect any anomaly
    # easily
    if distances is None:
        distances = numpy.zeros(shape=(0,), dtype=X.dtype)
    # distances will be changed in-place
    if issparse(X):
        raise NotImplementedError(  # pragma no cover
            "Sparse matrix is not implemented for norm 'L1'."
        )
    return _labels_inertia_precompute_dense(
        norm="L2",
        X=X,
        sample_weight=sample_weight,
        centers=centers,
        distances=distances,
    )


def _init_centroids(X, n_clusters, init, random_state, x_squared_norms):
    """Compute the initial centroids."""
    inst = KMeans(n_clusters)
    return inst._init_centroids(X, x_squared_norms, init, random_state)


def _validate_center_shape(X, n_clusters, centers) -> None:
    """Check if centers is compatible with X and n_clusters."""
    inst = KMeans(n_clusters)
    return inst._validate_center_shape(X, centers)
