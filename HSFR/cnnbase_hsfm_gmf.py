import torch
import torch.nn as nn


# Generalized matrix factorization is  from the paper  Neural collaborative filtering
#  https://arxiv.org/abs/1708.05031

class GMF(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim, dropout):
        super(GMF, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        embedding_dim: number of embedding dimensions;
        hidden_layer: dimension of each hidden layer (list type);
        dropout: dropout rate between fully connected layers.
        """
        self.dropout = dropout

        self.embed_user = nn.Embedding(user_num, embedding_dim)
        self.embed_item = nn.Embedding(item_num, embedding_dim)

        self.predict_layer = nn.Linear(embedding_dim, 1)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

        # Kaiming/Xavier initialization can not deal with non-zero bias terms
        if self.predict_layer.bias is not None:
            self.predict_layer.bias.data.zero_()

    def forward(self, user, item):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)
        output = embed_user * embed_item
        prediction = self.predict_layer(output)
        return prediction.view(-1)


class HSFM(nn.Module):
    def __init__(self, channels=64, r=4):
        super(HSFM, self).__init__()
        # inter_channels = int(channels // r)   this not not being used ?
        # Local attention
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

        )

        # Global attention
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        xa = x + y
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xi = x * wei + y * (1 - wei)
        return xi


class Model(nn.Module):

    def __init__(self, user_count, item_count):
        super(Model, self).__init__()

        self.user_count = user_count
        self.item_count = item_count

        # embedding setting
        self.embedding_size = 64
        self.spatial_shape = [64, 64]
        self.flat_shape = [10, 10]
        self.k_size = [3, 3]
        self.stride = 1

        self.filtered = [(self.flat_shape[0] - self.k_size[0]) // self.stride + 1,
                         (self.flat_shape[1] - self.k_size[1]) // self.stride + 1]

        fc_length = self.filtered[0] * self.filtered[1]
        self.fc1 = torch.nn.Linear(4, fc_length)
        # init target matrix of matrix factorization
        #         self.P = nn.Parameter(torch.randn((self.user_count, self.embedding_size)).cuda(), requires_grad=True)
        #         self.Q = nn.Parameter(torch.randn((self.item_count, self.embedding_size)).cuda(), requires_grad=True)
        self.P = nn.Embedding(self.user_count, self.embedding_size).cuda()
        self.Q = nn.Embedding(self.item_count, self.embedding_size).cuda()

        # GMF
        self.GMF_model = GMF(self.user_count, self.item_count, self.embedding_size, 0.0)
        # cnn setting
        self.channel_size = 32
        self.kernel_size = 2
        self.strides = 2
        self.in_layer = nn.Conv2d(3, self.channel_size * 2, self.kernel_size, stride=self.strides)
        self.sec_layer = nn.Conv2d(self.channel_size * 2, self.channel_size * 2, self.kernel_size,
                                   stride=self.strides)
        self.trd_layer = nn.Conv2d(self.channel_size * 2, self.channel_size, self.kernel_size,
                                   stride=self.strides)

        self.in_drop = nn.Dropout(0.2)
        self.in_relu = nn.ReLU()

        self.hsfm = HSFM(channels=self.channel_size)
        self.out_drp = nn.Dropout(0.2)
        self.out_layer = nn.Linear(self.user_count, 1)  # nn.Conv2d(128, 32, kernel_size=1) #nn.Linear(1,64)

    def forward(self, user_ids, item_ids):
        # GMF
        gmf = self.GMF_model(user_ids, item_ids)
        # convert float to int
        user_ids = list(map(int, user_ids))
        item_ids = list(map(int, item_ids))

        user_embeddings = self.P(torch.tensor(user_ids).cuda())
        item_embeddings = self.Q(torch.tensor(item_ids).cuda())

        user_2d = user_embeddings.unsqueeze(2)  # .view(-1, *self.spatial_shape)
        item_2d = item_embeddings.unsqueeze(1)  # .view(-1, *self.k_size)

        # 2D convolution for non-linear interaction map
        m = torch.bmm(user_2d, item_2d).view(-1, 1, *self.spatial_shape)
        p = torch.cat([m, m, m], dim=1)
        # print(m.shape, p.shape)
        # cnn
        x = self.in_layer(p)  # output: batch_size  32  1 * 1
        x = self.sec_layer(x)
        x = self.trd_layer(x)
        #        x = self.f_drop(x)
        y = x
        x = self.in_drop(x)
        x = self.in_relu(x)

        x = self.hsfm(x, y)

        x = x.view(256, -1, *[2, 2])
        x = x.sum(dim=1)

        x = x.view(256, -1)
        # print(x.shape)
        x = self.fc1(x)
        # m = y
        # matrix multiplication
        x = torch.mm(x, self.P.weight.transpose(1, 0))
        # print(x.shape, m.shape)
        x = self.out_drp(x)
        x = self.out_layer(x)

        pred = torch.sigmoid(x)
        return pred.view(-1), gmf

