"""KNN/PCA reference implementation. No sklearn estimators in this module."""
import numpy as np


class PCA:
    def __init__(self, components=100):
        self.components = components

    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.mean_ = x.mean(axis=0)
        _, s, vt = np.linalg.svd(x - self.mean_, full_matrices=False)
        self.basis_ = vt[:self.components].T.copy()
        self.ratio_ = (s[:self.components] ** 2) / np.sum(s ** 2)
        return self

    def transform(self, x, components=None):
        d = self.components if components is None else components
        return np.ascontiguousarray((x - self.mean_) @ self.basis_[:, :d], dtype=np.float32)


def distances(a, b, metric="l2"):
    """Rows of a are queries; L2 return value is SQUARED Euclidean distance."""
    if metric == "l2":
        return np.maximum((a*a).sum(1)[:, None] + (b*b).sum(1)[None, :] - 2*a@b.T, 0)
    if metric == "l1":
        return np.abs(a[:, None, :] - b[None, :, :]).sum(2)
    raise ValueError(metric)


def select_neighbors(d, k):
    """argpartition plus explicit distance ties: smaller training row wins."""
    if not 1 <= k <= d.shape[1]:
        raise ValueError("k must be between 1 and training size")
    ix = np.argpartition(d, k - 1, axis=1)[:, :k]
    boundary = np.take_along_axis(d, ix, 1).max(1)
    # Resolve ties across the kth boundary, not just inside the selected subset.
    for row in range(len(d)):
        below = np.flatnonzero(d[row] < boundary[row])
        equal = np.flatnonzero(d[row] == boundary[row])
        ix[row] = np.concatenate((below, equal[:k-len(below)]))
    vals = np.take_along_axis(d, ix, 1)
    order = np.lexsort((ix, vals), axis=1)
    ix = np.take_along_axis(ix, order, 1)
    return ix, np.take_along_axis(d, ix, 1)


def neighbors(train, query, k=9, metric="l2", engine="numpy", batch=64):
    train = np.ascontiguousarray(train, dtype=np.float32)
    query = np.ascontiguousarray(query, dtype=np.float32)
    indices, values = [], []
    if engine == "torch":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        a = torch.as_tensor(train, device=device)
        norm = (a*a).sum(1)
    for start in range(0, len(query), batch):
        q = query[start:start+batch]
        if engine == "numpy":
            d = distances(q, train, metric)
        else:
            import torch
            b = torch.as_tensor(q, device=device)
            if metric == "l2":
                d = ((b*b).sum(1)[:, None]+norm[None, :]-2*b@a.T).clamp_min(0)
            elif metric == "l1":
                d = torch.cdist(b, a, p=1)
            else:
                raise ValueError(metric)
            # Only small batches reach CPU; tie rules match the NumPy path.
            d = d.cpu().numpy()
        ix, ds = select_neighbors(d, k)
        indices.append(ix)
        values.append(ds)
    return np.concatenate(indices), np.concatenate(values)


def vote(labels, ds, k, metric="l2", weights="uniform"):
    labels, ds = labels[:, :k], ds[:, :k]
    if weights == "uniform":
        w = np.ones_like(ds, dtype=np.float64)
    elif weights == "distance":
        # sklearn distance weights are 1/d, NOT 1/d^2.
        actual = np.sqrt(ds) if metric == "l2" else ds
        zero = actual == 0
        w = np.where(zero.any(1)[:, None], zero.astype(float), 1/np.maximum(actual, 1e-30))
    else:
        raise ValueError(weights)
    scores = np.zeros((len(labels), 10), dtype=np.float64)
    for c in range(10):
        scores[:, c] = ((labels == c)*w).sum(1)
    # Course requires smaller class ID for equal majority counts.
    rank = np.argsort(-scores, axis=1, kind="stable")
    return rank[:, 0], rank, scores


def metrics(truth, pred, rank):
    cm = np.zeros((10, 10), dtype=np.int64)
    np.add.at(cm, (truth, pred), 1)
    tp = np.diag(cm)
    precision = tp/np.maximum(cm.sum(0), 1)
    recall = tp/np.maximum(cm.sum(1), 1)
    f1 = 2*precision*recall/np.maximum(precision+recall, 1e-30)
    return {"top1": float(np.mean(pred == truth)),
            "top3": float(np.mean((rank[:, :3] == truth[:, None]).any(1))),
            "top5": float(np.mean((rank[:, :5] == truth[:, None]).any(1))),
            "macro_f1": float(f1.mean()), "cm": cm.tolist(),
            "precision": precision.tolist(), "recall": recall.tolist(), "f1": f1.tolist()}


class MyKNN:
    """Teacher-style fit/predict interface around the from-scratch primitives."""
    def __init__(self, k=5, metric="l2", weights="uniform", engine="numpy", batch=64):
        self.k, self.metric, self.weights = k, metric, weights
        self.engine, self.batch = engine, batch

    def fit(self, x, y):
        self.x_ = np.ascontiguousarray(x, dtype=np.float32)
        self.y_ = np.asarray(y, dtype=int).copy()
        if len(self.x_) != len(self.y_) or not 1 <= self.k <= len(self.x_):
            raise ValueError("Invalid training lengths or k")
        if np.any((self.y_ < 0) | (self.y_ > 9)):
            raise ValueError("This MNIST implementation uses classes 0 through 9")
        return self

    def predict_proba(self, x):
        ix, ds = neighbors(self.x_, x, self.k, self.metric, self.engine, self.batch)
        _, _, scores = vote(self.y_[ix], ds, self.k, self.metric, self.weights)
        return scores/scores.sum(1)[:, None]

    def predict(self, x):
        return self.predict_proba(x).argmax(1)
