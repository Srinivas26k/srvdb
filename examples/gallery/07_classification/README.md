# 07 Classification

k-nearest-neighbour classification of the digits (75 % train, 25 % test,
stratified, k = 5). The training images go into SrvDB, flat and HNSW; a
test image gets the majority label of its five nearest stored images. The
reference is `KNeighborsClassifier(metric="cosine", algorithm="brute")`,
which computes the same vote.

```bash
python examples/gallery/07_classification/classification.py [--fast]
```

Output: accuracy, mean latency per test image and the share of
predictions that agree with scikit-learn, and `output/07_classification.png`
with accuracy and latency bars side by side.

Latency is per single query (one image at a time) for all three, which is
how an application classifies incoming items; scikit-learn is faster when
it predicts a whole batch at once. Small disagreements come from ties
between equally distant neighbours.
