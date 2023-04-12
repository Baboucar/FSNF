import os
import time
import argparse
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
##from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import writer, SummaryWriter

import data_util
import evaluate

from cnnbase_hsfm import Model

# from cnnbase import Model
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.00001, help="learning rate")
parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
parser.add_argument("--epochs", type=int, default=20, help="training epochs")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
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
        loss = - torch.log(self.sigmoid(distance)).mean()
        return loss


def load_model(model, checkname):
    try:
        if not osp.isfile(checkname):
            raise RuntimeError("=> no checkpoint found at '{}'".format(checkname))
        checkpoint = torch.load(checkname, map_location='cuda:0')

        if torch.cuda.is_available:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch: {}, best_pred: {})"
              .format(checkname, checkpoint['epoch'], checkpoint['best_pred']))
        model.eval()
        print("Success")
    except Exception as e:
        print("fail")
        print(e)


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
    loss_function =  BPRLoss().cuda()

    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)  # check

    writer = SummaryWriter()  # for visualization
    # ########################## TRAINING #####################################
    count, best_hr, best_tloss, best_vloss = 0, 0, 0, 0
    for epoch in range(args.epochs):
        model.train()  # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.ng_sample()

        length = len(train_loader)
        totalloss = 0
        for user, item, label in train_loader:
            user = user.to(device=args.device)
            item = item.to(device=args.device)
            label = label.to(device=args.device)

            model.zero_grad()

            pos = model(user, item)
            neg= model(user, label)

            loss = loss_function(pos, neg)  # - 0.01*l2

            # loss = (loss1 + loss2) // 2.0
            loss = 0.5 * loss

            loss.backward()
            optimizer.step()
            # writer.add_scalar('data/loss', loss.item(), count)
            count += 1
        print("view----->", totalloss, length, totalloss / length)
        best_tloss = (totalloss / length)
        model.eval()
        HR, NDCG = evaluate.metrics(model, test_loader, args.top_k, args.device)
        vloss = evaluate.validation_BPRloss(model, test_loader, args.device)
        best_vloss = vloss
        writer.add_scalar("ml-1m@hsmf", HR, epoch)
        writer.add_scalar("ml1mNDCCG@hsfm", NDCG, epoch)
        writer.add_scalar("ml1-best_vloss@hsfm", best_vloss, epoch)
        writer.add_scalar("ml-1best_tloss@hsfm", best_tloss, epoch)
        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
              time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("HR: {:.4f}\tNDCG: {:.4f}, tloss: {:.4f}\tvloss: {:.4f}".format(np.mean(HR), np.mean(NDCG), best_tloss,
                                                                              best_vloss))

        if HR > best_hr:
            best_epoch, best_hr, best_ndcg, best_tloss, best_vloss = epoch, HR, NDCG, best_tloss, best_vloss
            if args.out:
                if not os.path.exists(args.model_path):
                    os.mkdir(args.model_path)
                torch.save(model.state_dict(), os.path.join(args.model_path, 'model.pth'))

    print("End. Best epoch {:03d}: HR = {:.4f}, NDCG = {:.4f}, best_tloss = {:.4f}, best_vloss = {:.4f}".format(
        best_epoch, best_hr, best_ndcg, best_tloss, best_vloss))
