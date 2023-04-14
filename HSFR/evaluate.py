import numpy as np
import torch
from torch import nn


class BPRLoss(nn.Module):

    def __init__(self):
        super(BPRLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pos_preds, neg_preds):
        distance = pos_preds - neg_preds
        loss = torch.sum(torch.log((1 + torch.exp(-distance))))

        #         print('loss:', loss)
        return loss
def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def metrics(model, test_loader, top_k, device):
    HR, NDCG = [], []

    for user, item, label in test_loader:
        user = user.to(device=device)
        item = item.to(device=device)

        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)

        recommends = torch.take(item, indices).cpu().numpy().tolist()

        gt_item = item[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)

def validation_BPRloss(trained_model, dataloader, device):
    brp_loss = BPRLoss() #nn.MSELoss()

    with torch.no_grad():
        total_loss = 0
        for batch in dataloader:
            user_id, item_id, ratings = [i.to(device) for i in batch]

            pos = trained_model(user_id, item_id)
            neg = trained_model(user_id, ratings)

            vloss = brp_loss(pos, neg)
            total_loss = total_loss + vloss.item()
            # mse += torch.sqrt(mse_loss(predict[0], ratings))
            # sample_count += len(ratings)

    return (total_loss/len(dataloader))
