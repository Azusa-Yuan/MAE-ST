import torch
from torch import nn
from .MAEST import MAEST


class enhance(nn.Module):
    """Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting"""

    def __init__(self, pre_trained_path, pre_trained_args, backend_args, model='GWN'):
        super().__init__()
        self.pre_trained_path = pre_trained_path

        # initiative the pre_trained_model and backend models
        self.pre_trained_model = MAEST(**pre_trained_args)
        # load pre_trained_model
        self.load_pre_trained_model()

        self.backend_args = backend_args
        self.model = model
        self.predictor = None
        self.load_backend_model()

    def load_backend_model(self):
        """Load pre-trained model"""

        if self.model == 'MTGNN':
            from model.model_arch.MTGNN.MTGNN import MTGNN
            self.predictor = MTGNN(**self.backend_args)
        elif self.model == 'STGCN':
            from model.model_arch.STGCN.stgcn import STGCN
            self.predictor = STGCN(**self.backend_args)
        elif self.model == 'ASTGCN':
            from model.model_arch.MAEST.ASTGCN import ASTGCN
            self.predictor = ASTGCN(**self.backend_args)
        elif self.model == 'GWN':
            from model.model_arch.MAEST.gwnet_arch import GraphWaveNet
            self.predictor = GraphWaveNet(**self.backend_args)
        elif self.model == 'Linear':
            from model.model_arch.Liner_arch.stid_arch import STID
            self.predictor = STID(**self.backend_args)
        elif self.model == 'TGCN':
            from model.model_arch.TGCN.TGCN import TGCN
            self.predictor = TGCN(**self.backend_args)
        elif self.model == "Fourier":
            from model.model_arch.MAEST.FourierGNN import FGN
            self.predictor = FGN(**self.backend_args)
        elif self.model == 'msdr':
            from model.model_arch.MSDR.gmsdr_model import GMSDRModel
            self.predictor = GMSDRModel(**self.backend_args)
        else:
            raise ValueError

    def load_pre_trained_model(self):
        # load parameters
        checkpoint_dict = torch.load(self.pre_trained_path)
        self.pre_trained_model.load_state_dict(checkpoint_dict["model_state_dict"])
        # freeze parameters
        for param in self.pre_trained_model.parameters():
            param.requires_grad = False

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        # pre_trained
        pre_trained_feature = self.pre_trained_model(history_data, None)
        # enhancing downstream STGNNs
        output = self.predictor(history_data, pre_trained_feature)
        return output
