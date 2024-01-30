import os
import sys


# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
from basicts.utils.serialization import load_adj
from .model_arch.MAEST.MAEST import MAEST
from .model_runner.MAESTRunner import MAESTRunner
from basicts.utils.serialization import load_pkl
from .model_loss.mask_MAE import mask_MAE
from basicts.data import TimeSeriesForecastingDataset

CFG = EasyDict()

# ================= general ================ #
CFG.DESCRIPTION = "MAEST configuration"
CFG.RUNNER = MAESTRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "Solar"
CFG.DATASET_TYPE = "Solar"
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
node = node_feats.shape[1]

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "MAEST"
CFG.MODEL.ARCH = MAEST

CFG.MODEL.PARAM = {
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
    "mode": "pretrain",
    "mask_ratio": 0.25,
    "num_remasking": 2,
    "remask_method": "random",
    "remask_ration": 0.25,
    "use_mixed_proj": False,
    "mask_method": "all",
    "steps_per_day": 24
}
CFG.MODEL.change_epoch = 300
CFG.MODEL.FROWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = True
CFG.MODEL.INIT = True

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = mask_MAE
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.003,
    "weight_decay": 1.0e-5,
    "eps": 1.0e-8,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM= {
    "milestones": [80, 150, 250],
    "gamma": 0.3
}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 11
}
CFG.TRAIN.NUM_EPOCHS = 300
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
CFG.VAL.DATA.NUM_WORKERS = 3
CFG.VAL.DATA.PIN_MEMORY = True

# ================= test ================ #
# CFG.TEST = EasyDict()
# CFG.TEST.INTERVAL = 1
# # evluation
# # test data
# CFG.TEST.DATA = EasyDict()
# # read data
# CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# # dataloader args, optional
# CFG.TEST.DATA.BATCH_SIZE = 16
# CFG.TEST.DATA.PREFETCH = False
# CFG.TEST.DATA.SHUFFLE = False
# CFG.TEST.DATA.NUM_WORKERS = 2
# CFG.TEST.DATA.PIN_MEMORY = False
