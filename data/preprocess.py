"""
Classes describing datasets of user-item interactions. Instances of these
are returned by dataset-fetching and dataset-processing functions.
"""

import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class Interactions(object):
    """
    Interactions object. Contains (at a minimum) pair of user-item
    interactions. This is designed only for implicit feedback scenarios.

    Parameters
    ----------

    file_path: file contains (user,item,rating) triplets
    user_map: dict of user mapping
    item_map: dict of item mapping
    """

    
    
    def __init__(self, data=None, file_path=None):
        if data is not None:
            # If data is provided directly, use it to set user_ids and item_ids
            self.user_ids, self.item_ids = zip(*data) if data else ([], [])
            self.ratings = np.ones(len(self.user_ids))
        elif file_path is not None:
            # If a file path is provided, read the data from the file
            with open(file_path, 'r') as f:
                self.user_ids, self.item_ids = zip(*[
                    (int(line[0]), int(line[1])) for line in 
                    (line.strip().split() for line in f)
                ])
            self.ratings = np.ones(len(self.user_ids))
        else:
            raise ValueError("Either data or file_path must be provided.")
          
        # Create mappings for users and items
        self.user_map = {user_id: index for index, user_id in enumerate(sorted(set(self.user_ids)))}
        self.item_map = {item_id: index for index, item_id in enumerate(sorted(set(self.item_ids)))}

        # Map the user_ids and item_ids to the internal indices
        self.user_ids = np.array([self.user_map[user_id] for user_id in self.user_ids], dtype=np.int32)
        self.item_ids = np.array([self.item_map[item_id] for item_id in self.item_ids], dtype=np.int32)

        self.num_users = len(set(self.user_ids)) if len(self.user_ids) > 0 else 0
        self.num_items = len(set(self.item_ids)) if len(self.item_ids) > 0 else 0


        # Further processing here as needed, such as generating sequences or matrices for the interaction data
    def __len__(self):

            return len(self.user_ids)

    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix.
        """

        row = self.user_ids
        col = self.item_ids
        data = np.ones(len(self))

        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_users, self.num_items))

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """

        return self.tocoo().tocsr()

    def to_sequence(self, sequence_length=5, target_length=1):
        """
        Transform to sequence form.

        Valid subsequences of users' interactions are returned. For
        example, if a user interacted with items [1, 2, 3, 4, 5, 6, 7, 8, 9], the
        returned interactions matrix at sequence length 5 and target length 3
        will be be given by:

        sequences:

           [[1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7]]

        targets:

           [[6, 7],
            [7, 8],
            [8, 9]]

        sequence for test (the last 'sequence_length' items of each user's sequence):

        [[5, 6, 7, 8, 9]]

        Parameters
        ----------

        sequence_length: int
            Sequence length. Subsequences shorter than this
            will be left-padded with zeros.
        target_length: int
            Sequence target length.
        """

        # change the item index start from 1 as 0 is used for padding in sequences
        for k, v in self.item_map.items():
            self.item_map[k] = v + 1
        self.item_ids = self.item_ids + 1
        self.num_items += 1

        max_sequence_length = sequence_length + target_length

        # Sort first by user id
        sort_indices = np.lexsort((self.user_ids,))

        user_ids = self.user_ids[sort_indices]
        item_ids = self.item_ids[sort_indices]

        user_ids, indices, counts = np.unique(user_ids,
                                              return_index=True,
                                              return_counts=True)

        num_subsequences = sum([c - max_sequence_length + 1 if c >= max_sequence_length else 1 for c in counts])

        sequences = np.zeros((num_subsequences, sequence_length),
                             dtype=np.int64)
        sequences_targets = np.zeros((num_subsequences, target_length),
                                     dtype=np.int64)
        sequence_users = np.empty(num_subsequences,
                                  dtype=np.int64)

        test_sequences = np.zeros((self.num_users, sequence_length),
                                  dtype=np.int64)
        test_users = np.empty(self.num_users,
                              dtype=np.int64)

        _uid = None
        for i, (uid,
                item_seq) in enumerate(_generate_sequences(user_ids,
                                                           item_ids,
                                                           indices,
                                                           max_sequence_length)):
            if uid != _uid:
                test_sequences[uid][:] = item_seq[-sequence_length:]
                test_users[uid] = uid
                _uid = uid
            sequences_targets[i][:] = item_seq[-target_length:]
            sequences[i][:] = item_seq[:sequence_length]
            sequence_users[i] = uid

        self.sequences = SequenceInteractions(sequence_users, sequences, sequences_targets)
        self.test_sequences = SequenceInteractions(test_users, test_sequences)


class SequenceInteractions(object):
    """
    Interactions encoded as a sequence matrix.

    Parameters
    ----------
    user_ids: np.array
        sequence users
    sequences: np.array
        The interactions sequence matrix, as produced by
        :func:`~Interactions.to_sequence`
    targets: np.array
        sequence targets
    """

    def __init__(self,
                 user_ids,
                 sequences,
                 targets=None):
        self.user_ids = user_ids
        self.sequences = sequences
        self.targets = targets

        self.L = sequences.shape[1]
        self.T = None
        if np.any(targets):
            self.T = targets.shape[1]


def _sliding_window(tensor, window_size, step_size=1):
    if len(tensor) - window_size >= 0:
        for i in range(len(tensor), 0, -step_size):
            if i - window_size >= 0:
                yield tensor[i - window_size:i]
            else:
                break
    else:
        num_paddings = window_size - len(tensor)
        # Pad sequence with 0s if it is shorter than windows size.
        yield np.pad(tensor, (num_paddings, 0), 'constant')


def _generate_sequences(user_ids, item_ids,
                        indices,
                        max_sequence_length):
    for i in range(len(indices)):

        start_idx = indices[i]

        if i >= len(indices) - 1:
            stop_idx = None
        else:
            stop_idx = indices[i + 1]

        for seq in _sliding_window(item_ids[start_idx:stop_idx],
                                   max_sequence_length):
            yield (user_ids[i], seq)

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

class MovieLensDataset(Dataset):
    def __init__(self, user_ids, movie_ids):
        self.user_ids = user_ids
        self.movie_ids = movie_ids

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'movie_id': self.movie_ids[idx]
        }

def load_movielens_data(filepath, test_size=0.2):
    user_ids, movie_ids = [], []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                user_id, movie_id = parts
                user_ids.append(int(user_id))
                movie_ids.append(int(movie_id))
    
    user_ids = torch.tensor(user_ids)
    movie_ids = torch.tensor(movie_ids)

    # Split data into training and test sets
    train_user_ids, test_user_ids, train_movie_ids, test_movie_ids = train_test_split(
        user_ids, movie_ids, test_size=test_size)

    train_dataset = MovieLensDataset(train_user_ids, train_movie_ids)
    test_dataset = MovieLensDataset(test_user_ids, test_movie_ids)

    return train_dataset, test_dataset
 

# Example usage:
# dataset = load_movielens_data('path/to/movielens_1m.txt')
# data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
