"""Feature visualization."""

import numpy as np


def pca_colors(features, fit_on=None, percentile=1.0):
    """Map ``[N,C]`` features to ``[N,3]`` uint8 RGB via a 3-component PCA.

    Args:
        features: features to colour.
        fit_on: fit the PCA on these instead, e.g. a pooled set shared across
            several shapes to get comparable colours.
        percentile: components are clipped to the ``[p, 100-p]`` range before
            rescaling to ``[0,1]`` so outliers do not wash out the result.
    """
    from sklearn.decomposition import PCA

    features = np.asarray(features, dtype=np.float32)
    basis = features if fit_on is None else np.asarray(fit_on, dtype=np.float32)

    pca = PCA(n_components=3)
    pca.fit(basis)
    proj = pca.transform(features)

    lo = np.percentile(proj, percentile, axis=0)
    hi = np.percentile(proj, 100.0 - percentile, axis=0)
    span = np.where(hi - lo > 1e-8, hi - lo, 1.0)
    proj = np.clip((proj - lo) / span, 0.0, 1.0)
    return (proj * 255).astype(np.uint8)
