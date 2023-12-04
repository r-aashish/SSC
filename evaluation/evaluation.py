
import copy
import random
import numpy as np
import torch
from sklearn.metrics import ndcg_score


def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def evaluate_ranking(model, test, train=None, k=10):
    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    ndcg_scores = []

    for user_id in range(test.shape[0]):
        start, stop = test.indptr[user_id], test.indptr[user_id + 1]
        test_items = test.indices[start:stop]

        if not len(test_items):
            continue

        predictions = model.predict(user_id)
        predictions = np.argsort(-predictions)

        if train is not None:
            train_items = train.indices[train.indptr[user_id]:train.indptr[user_id + 1]]
            predictions = np.setdiff1d(predictions, train_items, assume_unique=True)

        top_k_predictions = predictions[:k]
        relevance = np.in1d(top_k_predictions, test_items, assume_unique=True).astype(int)
        user_ndcg = ndcg_at_k(relevance, k)
        ndcg_scores.append(user_ndcg)

    mean_ndcg = np.mean(ndcg_scores)
    return mean_ndcg


def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]  # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)

    return NDCG / valid_user

def evaluate_ndcg(model, data_loader, device, all_relevant_movies, top_k=10):
    model.eval()
    true_scores = []
    pred_scores = []

    with torch.no_grad():
        for batch in data_loader:
            user_ids = batch['user_id'].to(device)
            predictions = model(user_ids)  # Model's predictions
            _, top_predictions = torch.topk(predictions, k=top_k, dim=1)

            for i, user_id in enumerate(user_ids):
                user_true_scores = []
                relevant_movies = all_relevant_movies.get(user_id.item(), set())

                # Check if the top predictions are within the valid range
                if top_predictions.size(1) > predictions.size(1):
                    continue  # Skip if top_predictions are out of bounds

                user_predicted_ids = top_predictions[i]
                scores = predictions[i][user_predicted_ids].tolist()

                for movie_id in user_predicted_ids.tolist():
                    score = 1 if movie_id in relevant_movies else 0
                    user_true_scores.append(score)

                true_scores.append(user_true_scores)
                pred_scores.append(scores)

    ndcg = ndcg_score(true_scores, pred_scores)
    return ndcg