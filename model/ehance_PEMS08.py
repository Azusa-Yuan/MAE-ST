import os
import sys
import torch

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from basicts.utils.serialization import load_adj
from .model_arch.MAEST.enhance import enhance
from .model_runner import EnhanceRunner
from basicts.utils.serialization import load_pkl
from basicts.data import TimeSeriesForecastingDataset
from basicts.losses import masked_mae

CFG = EasyDict()

# ================= general ================ #
CFG.DESCRIPTION = "MAEST_GWN"
CFG.RUNNER = EnhanceRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "PEMS08"
CFG.DATASET_TYPE = "Traffic Flow"
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 0
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True
CFG.GPU_NUM = 1

# ================= parameters ================= #
node_feats_full = load_pkl("datasets/{0}/data_in{1}_out{2}.pkl".format(CFG.DATASET_NAME, CFG.DATASET_INPUT_LEN, CFG.DATASET_OUTPUT_LEN))["processed_data"]
train_index_list = load_pkl("datasets/{0}/index_in{1}_out{2}.pkl".format(CFG.DATASET_NAME, CFG.DATASET_INPUT_LEN, CFG.DATASET_OUTPUT_LEN))["train"]
node_feats = node_feats_full[:train_index_list[-1][-1], ...]
adj_mx, _ = load_adj("datasets/" + CFG.DATASET_NAME +
                     "/adj_mx.pkl", "doubletransition")
node, _ = adj_mx[0].shape

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "MAE_enhance_GWN"
CFG.MODEL.ARCH = enhance

CFG.MODEL.PARAM = {
    "pre_trained_path": "MAEST_ckpt/MAEST_PEMS08.pt",

    "pre_trained_args": {
        "num_nodes": node,
        "input_base_dim": 1,
        "input_extra_dim": 2,
        "hidden_dim": 64,
        "output_dim": 1,
        "horizon": 12,
        "embed_dim": 16,
        "embed_dim_spa": 16,
        "HS": 10,
        "HT": 16,
        "HT_Tem": 8,
        "num_route": 2,
        "mode": "enhance",
        "mask_ratio": 0.25,
        "num_remasking": 2,
        "remask_method": "random",
        "remask_ration": 0.1,
        "use_mixed_proj": False,
        "mask_method": "all"
    },

    "backend_args": {
        "num_nodes": node,
        "supports": [torch.tensor(i) for i in adj_mx],
        "dropout": 0.3,
        "gcn_bool": True,
        "addaptadj": True,
        "aptinit": None,
        "in_dim": 64,
        "out_dim": 12,
        "residual_channels": 32,
        "dilation_channels": 32,
        "skip_channels": 256,
        "end_channels": 512,
        "kernel_size": 2,
        "blocks": 4,
        "layers": 2
    },
    "model" : 'GWN',

}
CFG.MODEL.FROWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = True


# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr": 0.003,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM= {
    "milestones": [25, 50, 75],
    "gamma": 0.3
}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 10
}
CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.NULL_VAL = 0.0
# read data
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = True

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 64
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = True

# ================= test ================ #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# evluation
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 64
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = True
