import torch
from typing import Tuple, Union
from easytorch.utils.dist import master_only
from basicts.runners import BaseTimeSeriesForecastingRunner
from basicts.metrics import masked_mae, masked_rmse, masked_mape
from typing import Tuple, Union, Optional
import torch.nn as nn
from basicts.data.registry import SCALER_REGISTRY


class MAESTRunner(BaseTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.metrics = cfg.get("METRICS", {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape})
        self.forward_features = cfg["MODEL"].get("FROWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)

        self.register_epoch_meter("train_loss", "train", "{:.4f}")
        self.register_epoch_meter("val_loss", "val", "{:.4f}")
        self.register_epoch_meter("test_loss", "test", "{:.4f}")
        self.change_epoch = cfg["MODEL"].get("change_epoch", None)

    # override
    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Training details.

        Args:
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader
            epoch (int): current epoch.
            iter_index (int): current iter.

        Returns:
            loss (torch.Tensor)
        """

        iter_num = (epoch-1) * self.iter_per_epoch + iter_index

        forward_return = list(self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True))
        mask = forward_return[2]

        # re-scale data
        real_value_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[1], **self.scaler["args"])
        # 当remask的数量超过1时 要加上多个loss
        if isinstance(forward_return[0], list):
            loss_rec_all = 0
            for i in range(len(forward_return[0])):
                # re-scale data
                prediction_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[0][i], **self.scaler["args"])
                loss, loss_basic = self.loss(prediction_rescaled, real_value_rescaled, mask)
                loss_rec_all += loss
            loss_rec = loss_rec_all
        else:
            prediction_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[0], **self.scaler["args"])
            loss, loss_basic = self.loss(prediction_rescaled, real_value_rescaled, mask)
            loss_rec = loss

        # # loss
        # loss = self.loss(prediction_rescaled, real_value_rescaled)
        # if epoch > self.change_epoch:
        #     loss_kl = nn.KLDivLoss(reduction='sum')(forward_return[3].log(), forward_return[4]) * 0.1
        #     loss += loss_kl

        # metrics
        # for metric_name, metric_func in self.metrics.items():j
        #     metric_item = self.metric_forward(metric_func, forward_return[:2])
        #     self.update_epoch_meter("train_"+metric_name, metric_item.item())
        self.update_epoch_meter("train_loss", loss_rec.item())
        return loss_rec

    # override
    @master_only
    def on_validating_end(self, train_epoch: Optional[int]):
        """Callback at the end of validating.

        Args:
            train_epoch (Optional[int]): current epoch if in training process.
        """

        if train_epoch is not None:
            try:
                self.val_best = self.save_best_model(train_epoch, "val_loss", greater_best=False)
            except:
                self.test_best = False
                self.save_best_model(train_epoch, "val_loss", greater_best=False)

    # override
    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple], train_epoch=None):
        """Validation details.

        Args:
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader
            train_epoch (int): current epoch if in training process. Else None.
            iter_index (int): current iter.
        """
        forward_return = self.forward(data=data, epoch=train_epoch, iter_num=None, train=False)

        # re-scale data
        real_value_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[1], **self.scaler["args"])
        mask = forward_return[2]
        loss_rec_all = 0
        if isinstance(forward_return[0], list):
            loss_rec_all = 0
            for i in range(len(forward_return[0])):
                # re-scale data
                prediction_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[0][i], **self.scaler["args"])
                loss, loss_basic = self.loss(prediction_rescaled, real_value_rescaled, mask)
                loss_rec_all += loss
            loss_rec = loss_rec_all
        else:
            prediction_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[0], **self.scaler["args"])
            loss, loss_basic = self.loss(prediction_rescaled, real_value_rescaled, mask)
            loss_rec = loss

        # loss = self.loss(prediction_rescaled, real_value_rescaled)
        self.update_epoch_meter("val_loss", loss_rec.item())

    # override
    @torch.no_grad()
    @master_only
    def test(self):
        """Evaluate the model.

        Args:
            train_epoch (int, optional): current epoch if in training process.
        """

        # test loop
        prediction = []
        real_value = []
        for _, data in enumerate(self.test_data_loader):
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)
            prediction.append(forward_return[0])        # preds = forward_return[0]
            real_value.append(forward_return[1])        # testy = forward_return[1]
        prediction = torch.cat(prediction, dim=0)
        real_value = torch.cat(real_value, dim=0)
        # metrics
        loss = self.metric_forward(self.loss, [prediction, real_value])
        # summarize the results.
        # test performance of different horizon
        self.update_epoch_meter("test_loss", loss.item())

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select input features and reshape data to fit the target model.

        Args:
            data (torch.Tensor): input history data, shape [B, L, N, C].

        Returns:
            torch.Tensor: reshaped data
        """

        # select feature using self.forward_features
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select target features and reshape data back to the BasicTS framework

        Args:
            data (torch.Tensor): prediction of the model with arbitrary shape.

        Returns:
            torch.Tensor: reshaped data with shape [B, L, N, C]
        """

        # select feature using self.target_features
        data = data[:, :, :, self.target_features]
        return data

    def forward(self, data: tuple, epoch:int = None, iter_num: int = None, train:bool = True, **kwargs) -> tuple:
        """feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history data). [B, L, N, C] for each of them
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            tuple: (prediction, real_value)
        """

        # preprocess
        future_data, history_data = data
        history_data        = self.to_running_device(history_data)      # B, L, N, C
        future_data         = self.to_running_device(future_data)       # B, L, N, C

        out, mask, probability = self.model(history_data, None, epoch=epoch)

        return out, self.select_target_features(history_data), mask, probability
