# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import random
import math
import os

torch.backends.cudnn.benchmark = False
# Dataset and evaluation protocols reused from
# https://github.com/hexiangnan/neural_collaborative_filtering
from Dataset import Dataset
from data import sequence2mat
from evaluate import evaluate_model


# %%
def generate_instances(train_mat, positive_size=2, negative_time=8, is_sparse=False):
    data = []
    users_num, items_num = train_mat.shape

    if is_sparse:
        indptr = train_mat.indptr
        indices = train_mat.indices
    for u in range(users_num):
        if is_sparse:
            rated_items = indices[indptr[u]:indptr[u + 1]]  # 用户u中有评分项的id
        else:
            rated_items = np.where(train_mat[u, :] > 0)[0]

        for item0 in rated_items:
            for item1 in np.random.choice(rated_items, size=positive_size):
                data.append([u, item0, item1, 1.])
            for _ in range(positive_size * negative_time):
                item1 = np.random.randint(items_num)  # no matter item1 is positive or negtive
                item2 = np.random.randint(items_num)
                while item2 in rated_items:
                    item2 = np.random.randint(items_num)
                data.append([u, item2, item1, 0.])
    return data


# %%
def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # CPU seed
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)  # GPU
    #     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True


def evaluate(model, test_ratings, test_negatives, K=10):
    """Helper that calls evaluate from the NCF libraries."""
    (hits, ndcgs) = evaluate_model(model, test_ratings, test_negatives, K=K, num_thread=1)
    return np.array(hits).mean(), np.array(ndcgs).mean()


def get_similar_items(item_mat, idx, topk=5):
    m, k = item_mat.shape
    target_item = item_mat[idx, :]
    target_mat = np.reshape(np.tile(target_item, m), (-1, k))
    sim = [np.dot(target_mat[i], item_mat[i]) / (np.linalg.norm(target_mat[i]) * np.linalg.norm(item_mat[i]))
           for i in range(m)]
    sorted_items = np.argsort(-np.array(sim))
    return sorted_items[:topk + 1]  # the most similar is itself


def get_key(item_dict, value):
    key = -1
    for (k, v) in item_dict.items():
        if v == value:
            key = k
    return key


# read original records
def get_item_dict(file_dir):
    # output:
    # N: the number of user;
    # M: the number of item
    # data: the list of rating information
    user_ids_dict, rated_item_ids_dict = {}, {}
    N, M, u_idx, i_idx = 0, 0, 0, 0
    data_rating = []
    data_time = []
    f = open(file_dir)
    for line in f.readlines():
        if '::' in line:
            u, i, r = line.split('::')[:3]
        elif ',' in line:
            u, i, r = line.split(',')[:3]
        else:
            u, i, r = line.split()[:3]

        if u not in user_ids_dict:
            user_ids_dict[u] = u_idx
            u_idx += 1
        if i not in rated_item_ids_dict:
            rated_item_ids_dict[i] = i_idx
            i_idx += 1
        data_rating.append([user_ids_dict[u], rated_item_ids_dict[i], float(r)])

    f.close()
    N = u_idx
    M = i_idx

    return rated_item_ids_dict


def id_name(file_dir):
    id_name_dict = {}
    f = open(file_dir, 'r', encoding='latin-1')
    for line in f.readlines():
        movie_id, movie_name = line.split('|')[:2]
        id_name_dict[int(movie_id)] = movie_name

    return id_name_dict


class BPRLoss(nn.Module):

    def __init__(self):
        super(BPRLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pos_preds, neg_preds):
        distance = pos_preds - neg_preds
        loss = torch.sum(torch.log((1 + torch.exp(-distance))))

        #         print('loss:', loss)
        return loss


# %%
class UInet(nn.Module):
    def __init__(self, embedding_user, embedding_item, embedding_size=16, out_channels=64, kernel_size=2, stride=1,
                 padding=0, n_class=1):
        super(UInet, self).__init__()
        self.embedding_size = embedding_size
        self.embedding_user = embedding_user
        self.embedding_item =  embedding_item
        self.cnn = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                             padding=padding)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(int(((self.embedding_size - kernel_size + 2 * padding) / stride + 1) * out_channels),
                                n_class)

    def forward(self, x):
        x = x.cuda()
        print("shape", self.embedding_item, x[:, 0], x[:, 1], x[:, 2])
        embed_users = self.embedding_user(x[:, 0])
        embed_items0 = self.embedding_item(x[:, 1])
        embed_items1 = self.embedding_item(x[:, 2])
        out = torch.cat([embed_users, embed_items0], 1).reshape(-1, 1, 2, self.embedding_size)
        out = self.cnn(out)
        out = self.relu(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

    def predict(self, pairs, batch_size, verbose):
        """Computes predictions for a given set of user-item pairs.
        Args:
          pairs: A pair of lists (users, items) of the same length.
          batch_size: unused.
          verbose: unused.
        Returns:
          predictions: A list of the same length as users and items, such that
          predictions[i] is the models prediction for (users[i], items[i]).
        """
        del batch_size, verbose
        num_examples = len(pairs[0])
        assert num_examples == len(pairs[1])
        predictions = np.empty(num_examples)
        pairs = np.array(pairs, dtype=np.int16)
        for i in range(num_examples):
            x = np.c_[pairs[0][i], pairs[1][i], pairs[1][i]]
            x = torch.from_numpy(x).long()
            out = self.forward(x)
            predictions[i] = out.reshape(-1).data.numpy()
        return predictions


# %%
class UIInet(nn.Module):
    def __init__(self, embedding_user, embedding_item, embedding_size=16, out_channels=64, kernel_size=2, stride=1,
                 padding=0, n_class=1):
        super(UIInet, self).__init__()
        self.embedding_size, self.kernel_size = embedding_size, kernel_size
        self.embedding_user, self.embedding_item = embedding_user, embedding_item
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding)
        self.cnn2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.relu = nn.ReLU()
        if self.kernel_size == 2:
            input_size = (self.embedding_size - self.kernel_size + 2 * padding) / stride + 1
            channel_num = out_channels
        else:
            input_size = self.embedding_size
            channel_num = out_channels
        self.linear = nn.Linear(int(((input_size - self.kernel_size + 2 * padding) / stride + 1) * channel_num),
                                n_class)

    def forward(self, x):
        # x = x.cuda()
        embed_users = self.embedding_user(x[:, 0])
        embed_items0 = self.embedding_item(x[:, 1])
        embed_items1 = self.embedding_item(x[:, 2])
        out = torch.cat([embed_items0, embed_users, embed_items1], 1).reshape(-1, 1, 3, self.embedding_size)
        out = self.cnn1(out)
        out = self.relu(out)
        if self.kernel_size == 2:
            out = self.cnn2(out)
            out = self.relu(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

    def predict(self, pairs, batch_size, verbose):
        """Computes predictions for a given set of user-item pairs.
        Args:
          pairs: A pair of lists (users, items) of the same length.
          batch_size: unused.
          verbose: unused.
        Returns:
          predictions: A list of the same length as users and items, such that
          predictions[i] is the models prediction for (users[i], items[i]).
        """
        del batch_size, verbose
        num_examples = len(pairs[0])
        assert num_examples == len(pairs[1])
        predictions = np.empty(num_examples)
        pairs = np.array(pairs, dtype=np.int16)
        for i in range(num_examples):
            x = np.c_[pairs[0][i], pairs[1][i], pairs[1][i]]
            x = torch.from_numpy(x).long()
            out = self.forward(x)
            predictions[i] = out.reshape(-1).data.numpy()
        return predictions


# %%
class Net(nn.Module):
    def __init__(self, users_num, items_num, embedding_size=16, out_channels=64, kernel_size=2, stride=1, padding=0,
                 n_class=1):
        super(Net, self).__init__()
        self.embedding_size, self.kernel_size, self.items_num, self.users_num = embedding_size, kernel_size, items_num, users_num
        self.embedding_user = nn.Embedding(self.users_num, self.embedding_size)
        self.embedding_item = nn.Embedding(self.items_num, self.embedding_size)
        # self.embedding_user  = nn.Embedding.from_pretrained(torch.nn.init.normal(tensor=torch.Tensor(self.users_num, self.embedding_size), mean=0, std=0.1))
        # self.embedding_item = nn.Embedding.from_pretrained(torch.nn.init.normal(tensor=torch.Tensor(self.items_num, self.embedding_size), mean=0, std=0.1))

        self.net_ui = UInet(embedding_user=self.embedding_user,
                            embedding_item=self.embedding_item,
                            embedding_size=self.embedding_size,
                            out_channels=out_channels,
                            kernel_size=2,
                            stride=stride,
                            padding=padding,
                            n_class=n_class)
        self.net_uii = UIInet(embedding_user=self.embedding_user,
                              embedding_item=self.embedding_item,
                              embedding_size=self.embedding_size,
                              out_channels=out_channels,
                              kernel_size=self.kernel_size,
                              stride=stride,
                              padding=padding,
                              n_class=n_class)

    def forward(self, x):
        # x = x.cuda()
        out1 = self.net_ui(x)
        out2 = self.net_uii(x)
        return out1, out2

    def predict(self, pairs, batch_size, verbose):
        """Computes predictions for a given set of user-item pairs.
        Args:
          pairs: A pair of lists (users, items) of the same length.
          batch_size: unused.
          verbose: unused.
        Returns:
          predictions: A list of the same length as users and items, such that
          predictions[i] is the models prediction for (users[i], items[i]).
        """
        del batch_size, verbose
        num_examples = len(pairs[0])
        assert num_examples == len(pairs[1])
        predictions = np.empty(num_examples)
        pairs = np.array(pairs, dtype=np.int16)
        for i in range(num_examples):
            x = np.c_[pairs[0][i], pairs[1][i], pairs[1][i]]
            x = torch.from_numpy(x).long()
            out, _ = self.forward(x)
            predictions[i] = out.detach().cpu().reshape(-1).data.numpy()
        return predictions

    def get_embeddings(self):
        idx = torch.LongTensor([i for i in range(self.items_num)]).cpu()
        embeddings = self.embedding_item(idx)
        return embeddings


# %%
def train(model, train_mat, test_ratings, test_negatives, users_num, items_num, train_list=None, test_list=None,
          learning_rate=1e-2, weight_decay=1e-6, alpha=1., positive_size=1, negative_time=4, epochs=2,
          batch_size=1024, topK=10, mode='hr'):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = BPRLoss()  # nn.MSELoss().cuda()

    if train_list != None:
        train_mat = sequence2mat(sequence=train_list, N=users_num, M=items_num)  # train data : user-item matrix
        is_sparse = False

    hr_list = []
    loss_his = []
    ndcg_list = []
    hr, ndcg = evaluate(model, test_ratings, test_negatives, K=topK)
    embeddings = model.get_embeddings()
    hr_list.append(hr)
    ndcg_list.append(ndcg)
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg = hr, ndcg
    for epoch in range(epochs):
        data_sequence = generate_instances(train_mat, positive_size=positive_size, negative_time=negative_time,
                                           is_sparse=True)
        # data_sequence = read_list("output/" + str(epoch) + ".txt")

        train_size = len(data_sequence)
        np.random.shuffle(data_sequence)
        batch_size = batch_size
        total_batch = math.ceil(train_size / batch_size)
        best_loss = 0
        for batch in range(total_batch):
            start = (batch * batch_size) % train_size
            end = min(start + batch_size, train_size)
            data_array = np.array(data_sequence[start:end])
            x = torch.from_numpy(data_array[:, :3]).long()
            y = torch.from_numpy(data_array[:, -1]).reshape(-1, 1)
            x = x
            y = y
            y1, y2 = model(x)
            loss = criterion(y2.float(), y.float()) + (alpha + criterion(y1.float(), y.float()))
            best_loss += loss

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

        # Evaluation
        hr, ndcg = evaluate(model, test_ratings, test_negatives, K=topK)
        hr_list.append(hr)
        loss_his.append(best_loss)
        ndcg_list.append(ndcg)
        print('epoch=%d, loss=%.8f, HR=%.4f, NDCG=%.4f' % (epoch, best_loss, hr, ndcg))
        model_path = "/home/janneh/Test_Dev/babou/cocnn-main/model"

        if hr > best_hr:
            best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            torch.save(model.state_dict(), os.path.join(model_path, 'model.pth'))
            # print("The current best: ", hr, ndcg, epoch, total_loss)

        mlist = hr_list
        if mode == 'ndcg':
            mlist = ndcg_list

        if (len(mlist) > 5) and (mlist[-2] < mlist[-3] > mlist[-1]):
            best_hr, best_ndcg, best_loss = hr_list[-3], ndcg_list[-3], loss_his[-3]
            torch.save(model.state_dict(), os.path.join(model_path, 'model.pth'))

            embeddings = model.get_embeddings()
            # break
        best_hr, best_ndcg, best_loss = hr, ndcg, best_loss
        embeddings = model.get_embeddings()

    print("End. Best HR = %.4f, NDCG = %.4f, loss = %.8f " % (best_hr, best_ndcg, best_loss))
    return embeddings


#
# # %%
# dataset_path = 'data/pinterest-20'
#
# # Load the dataset
# dataset = Dataset(dataset_path)
# train_mat, test_ratings, test_negatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
# print('Dataset: #user=%d, #item=%d, #train_pairs=%d, #test_pairs=%d'
#       % (dataset.num_users, dataset.num_items, train_mat.nnz, len(test_ratings)))
#
# embedding_size = 64  # e=32,o=32 or 64
# out_channels = 8
# learning_rate = 1e-2
# weight_decay = 1e-6
# alpha = 1
# kernel_size = 2
#
# positive_size = 1
# negative_time = 4
# epochs = 20
# batch_size = 256  # batch大小
# topK = 10
# mode = 'hr'
#
# setup_seed(3)
# # Initialize the model
# model = Net(users_num=dataset.num_users, items_num=dataset.num_items,
#             embedding_size=embedding_size, out_channels=out_channels, kernel_size=kernel_size)
# print("Janneh--->A", torch.cuda.is_available())
# if torch.cuda.is_available():
#     model = model.cpu()
#
# # Train and evaluate model
# embeddings = train(model=model,
#                    train_mat=train_mat.tocsr(),
#                    test_ratings=test_ratings,
#                    test_negatives=test_negatives,
#                    users_num=dataset.num_users,
#                    items_num=dataset.num_items,
#                    learning_rate=learning_rate,
#                    weight_decay=weight_decay,
#                    alpha=alpha,
#                    positive_size=positive_size,
#                    negative_time=negative_time,
#                    epochs=epochs,
#                    batch_size=batch_size,
#                    topK=topK,
#                    mode=mode)
# print('----------------------------------------------------------')

# # %%
# dataset_path = 'data/100k'
#
# # Load the dataset
# dataset = Dataset(dataset_path)
# train_mat, test_ratings, test_negatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
# print('Dataset: #user=%d, #item=%d, #train_pairs=%d, #test_pairs=%d'
#       % (dataset.num_users, dataset.num_items, train_mat.nnz, len(test_ratings)))
#
# embedding_size = 32  # e=32,o=32 or 64 when kernel_size==3
# out_channels = 64
# learning_rate = 1e-2
# weight_decay = 1e-6
# alpha = 1.0
# kernel_size = 3
#
# positive_size = 1
# negative_time = 4
# epochs = 64
# batch_size = 1024  # batch大小
# topK = 10
# mode = 'hr'
# seed = 10
#
# setup_seed(seed)
# # Initialize the model
# model = Net(users_num=dataset.num_users, items_num=dataset.num_items,
#             embedding_size=embedding_size, out_channels=out_channels, kernel_size=kernel_size).cuda()
# print("Janneh--->B", torch.cuda.is_available())
# if torch.cuda.is_available():
#     model = model.cuda()
#
# # Train and evaluate model
# embeddings = train(model=model,
#                    train_mat=train_mat.tocsr(),
#                    test_ratings=test_ratings,
#                    test_negatives=test_negatives,
#                    users_num=dataset.num_users,
#                    items_num=dataset.num_items,
#                    learning_rate=learning_rate,
#                    weight_decay=weight_decay,
#                    alpha=alpha,
#                    positive_size=positive_size,
#                    negative_time=negative_time,
#                    epochs=epochs,
#                    batch_size=batch_size,
#                    topK=topK,
#                    mode=mode)
# print('----------------------------------------------------------')
#
# %%
dataset_path = 'data/pinterest-20'

# Load the dataset
dataset = Dataset(dataset_path)
train_mat, test_ratings, test_negatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
print('Dataset: #user=%d, #item=%d, #train_pairs=%d, #test_pairs=%d'
      % (dataset.num_users, dataset.num_items, train_mat.nnz, len(test_ratings)))

embedding_size = 64  # e=32,o=8 when kernel_size==2
out_channels = 32
learning_rate = 1e-2
weight_decay = 1e-6
alpha = 1.0
kernel_size = 2

positive_size = 1
negative_time = 4
epochs = 64
batch_size = 256  # batch大小
topK = 10
mode = 'hr'
seed = 17

setup_seed(seed)
# Initialize the model
model = Net(users_num=dataset.num_users, items_num=dataset.num_items,
            embedding_size=embedding_size, out_channels=out_channels, kernel_size=kernel_size)

if torch.cuda.is_available():
    model = model.cuda()

# Train and evaluate model
embeddings = train(model=model,
                   train_mat=train_mat.tocsr(),
                   test_ratings=test_ratings,
                   test_negatives=test_negatives,
                   users_num=dataset.num_users,
                   items_num=dataset.num_items,
                   learning_rate=learning_rate,
                   weight_decay=weight_decay,
                   alpha=alpha,
                   positive_size=positive_size,
                   negative_time=negative_time,
                   epochs=epochs,
                   batch_size=batch_size,
                   topK=topK,
                   mode=mode)
print('----------------------------------------------------------')
#
# # %% [markdown]
# # #### build the dict: original id <-> new id, original id <-> name
#
# # %%
# file_dir = 'datasets/ml-100k/u.item'
# id_name_dict = id_name(file_dir)  # original id : movie name
#
# file_dir = 'datasets/ml-100k/u.data'
# item_dict = get_item_dict(file_dir)  # original id : new id
#
# # %%
# movieid_list = [174, 195, 449]
#
# for movieid in movieid_list:
#     print('MovieID:', movieid, '; MovieName:', id_name_dict[movieid])
#     original_id = str(movieid)
#     target_item = item_dict[original_id]
#
#     top5 = get_similar_items(embeddings.data.numpy(), idx=target_item)
#     movie_list = [get_key(item_dict=item_dict, value=i) for i in top5]
#     rec_list = [id_name_dict[int(movie_id)] for movie_id in movie_list[1:]]
#     for i in range(len(rec_list)):
#         print('\n{0}: {1}'.format(i + 1, rec_list[i]))
#     print('------------------------------------------------------------------')
#
# # %%
#
#
