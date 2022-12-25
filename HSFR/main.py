import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import data_util
import evaluate

from HSFR import Model

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.00001, help="learning rate")  # 0.00001
parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
parser.add_argument("--epochs", type=int, default=20, help="training epoches")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
# parser.add_argument("--embedding_dim", type=int, default=64, help="dimension of embedding")
# parser.add_argument("--hidden_layer", type=list, default=[128, 64, 32, 16], help="dimension of each hidden layer")
parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
parser.add_argument("--test_num_ng", type=int, default=256, help="sample part of negative items for testing")
parser.add_argument("--data_set", type=str, default="ml-1m", help="data set. 'ml-1m' or 'pinterest-20'")
parser.add_argument("--data_path", type=str, default="data")
parser.add_argument("--model_path", type=str, default="model")
parser.add_argument("--out", default=True, help="save model or not")
parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
args = parser.parse_args()

args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

cudnn.benchmark = True


class BPRLoss(nn.Module):

    def __init__(self):
        super(BPRLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pos_preds, neg_preds):
        distance = pos_preds - neg_preds
        loss = torch.sum(torch.log((1 + torch.exp(-distance))))

        #         print('loss:', loss)
        return loss


if __name__ == "__main__":

    origin = str(os.path.abspath(__file__))[:-7]
    data_path = os.path.join(origin, args.data_path)
    model_path = os.path.join(origin, args.model_path)

    # print(data_path)
    # print(model_path)
    data_file = os.path.join(data_path, args.data_set)
    print(data_file)
    train_data, test_data, user_num, item_num, train_mat = data_util.load_all(data_file)

    train_dataset = data_util.NCFData(train_data, item_num, train_mat, args.num_ng, True)
    test_dataset = data_util.NCFData(test_data, item_num, train_mat, 0, False)
    train_loader = data.DataLoader(train_dataset, drop_last=True, batch_size=args.batch_size, shuffle=True,
                                   num_workers=4)
    test_loader = data.DataLoader(test_dataset, drop_last=True, batch_size=args.test_num_ng, shuffle=False,
                                  num_workers=0)

    model = Model(user_num, item_num)
    model.to(device=args.device)
    loss_cnn = BPRLoss()
    loss_gmc = nn.BCEWithLogitsLoss()

    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)  # check

    writer = SummaryWriter()  # for visualization
    # ########################## TRAINING #####################################
    count, best_hr = 0, 0
    for epoch in range(args.epochs):
        model.train()  # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.ng_sample()

        length = len(train_loader)

        for user, item, label in train_loader:
            user = user.to(device=args.device)
            item = item.to(device=args.device)
            label = label.to(device=args.device)
            model.zero_grad()
            pred = model(user, item)

            pos = model(user, item)
            neg = model(user, label)

            loss = loss_cnn(pos, neg)  # - 0.01*l2
            #uncomment the following 3 lines to use the weights from GMF
            # loss1 = loss_cnn(pos, neg)
            # loss2 = loss_gmc(pred, label.float())
            # loss = loss1 + loss2

            loss.backward()
            optimizer.step()
            # writer.add_scalar('data/loss', loss.item(), count)
            count += 1

        model.eval()
        HR, NDCG = evaluate.metrics(model, test_loader, args.top_k, args.device)
        writer.add_scalar("ml-1mchanneh64HR@hsmf+stack", HR, epoch)
        writer.add_scalar("ml-1mchannel64NDCCG@hsfm+stack", NDCG, epoch)
        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
              time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("HR: {:.4f}\tNDCG: {:.4f}".format(np.mean(HR), np.mean(NDCG)))

        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
            if args.out:
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                torch.save(model.state_dict(), os.path.join(model_path, 'model.pth'))

    print("End. Best epoch {:03d}: HR = {:.4f}, NDCG = {:.4f}".format(best_epoch, best_hr, best_ndcg))
