# 06 Clustering and visualization

Stores the digits images (unit-normalised, 64 dimensions) in SrvDB and
clusters the same vectors with scikit-learn's `KMeans` (k = 10). For each
cluster, `db.search(centroid, k=1)` returns the stored image closest to the
centroid: a real record that represents the cluster, with its label read
back through `db.get`.

```bash
python examples/gallery/06_clustering/clustering.py [--fast]
```

Output: the adjusted Rand index of the clusters against the true digit
labels (1.0 is a perfect match, 0 is chance), each cluster's size and
representative, and `output/06_clustering.png`: the vectors projected on
the first two PCA components, one grey shade and marker per cluster, with
the representatives circled.

SrvDB returns ids and scores, not raw vectors; keep the array you embedded
(or embed again) when you need it for clustering.
