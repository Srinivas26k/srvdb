# 07 Classification

k-nearest-neighbour classification of the digits (75 % train, 25 % test,
stratified, k = 5). The training images go into SrvDB, flat and HNSW; a
test image gets the majority label of its five nearest stored images. The
reference is `KNeighborsClassifier(metric="cosine", algorithm="brute")`,
which computes the same vote.

```bash
python examples/gallery/07_classification/classification.py [--fast]
```

Output: accuracy, the share of predictions that agree with scikit-learn,
and latency per test image measured **both** ways for every classifier:

- **per call:** one image per call (`db.search(row)` or
  `clf.predict(row)`), how an application classifies items as they
  arrive. It includes each library's fixed Python overhead per call, which
  is large for scikit-learn's `predict`; do not read it as a difference in
  search speed.
- **batch:** the whole test set in one call (`db.search_batch(rows)` or
  `clf.predict(rows)`), divided by the number of images. This is the fair
  comparison of the search itself.

`output/07_classification.png` has two panels: accuracy (bars from zero,
values labelled) and latency per image with a per-call and a batch bar
for each classifier. Small disagreements in labels come from ties between
equally distant neighbours.
