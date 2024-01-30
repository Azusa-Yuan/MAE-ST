import torch


def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    mae_loss = torch.abs(true - pred)
    # print(mae_loss[mae_loss>3].shape, mae_loss[mae_loss<1].shape, mae_loss.shape)
    return torch.mean(mae_loss), mae_loss


def mask_MAE(preds, labels, mask=None, mask_value=0.1):
    preds = preds * mask
    labels = labels * mask
    mae, mae_loss = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
    return mae, mae_loss


def MAE(pred, true, null_val=None):
    mae_loss = torch.abs(true - pred)
    # print(mae_loss[mae_loss>3].shape, mae_loss[mae_loss<1].shape, mae_loss.shape)
    return torch.mean(mae_loss)


def MAE_torch2(pred, true, mask):
    mask = torch.gt(mask, 0.2)
    pred = torch.masked_select(pred, mask)
    true = torch.masked_select(true, mask)
    mae_loss = torch.abs(true - pred)
    # print(mae_loss[mae_loss>3].shape, mae_loss[mae_loss<1].shape, mae_loss.shape)
    return torch.mean(mae_loss), mae_loss