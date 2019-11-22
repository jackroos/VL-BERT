from easydict import EasyDict as edict
import yaml

_C = edict()
config = _C

# ------------------------------------------------------------------------------------- #
# Common options
# ------------------------------------------------------------------------------------- #
_C.RNG_SEED = -1
_C.OUTPUT_PATH = ''
_C.MODULE = ''
_C.GPUS = ''
_C.LOG_FREQUENT = 50
_C.VAL_FREQUENT = 1
_C.CHECKPOINT_FREQUENT = 1
_C.MODEL_PREFIX = ''
_C.NUM_WORKERS_PER_GPU = 4
_C.SCALES = ()

# ------------------------------------------------------------------------------------- #
# Common dataset options
# ------------------------------------------------------------------------------------- #
_C.DATASET = edict()
_C.DATASET.DATASET = ''
_C.DATASET.LABEL_INDEX_IN_BATCH = 7
_C.DATASET.APPEND_INDEX = False
_C.DATASET.TASK = 'Q2AR'
_C.DATASET.BASIC_ALIGN = False
_C.DATASET.DATASET_PATH = ''
_C.DATASET.ROOT_PATH = ''
_C.DATASET.TRAIN_IMAGE_SET = ''
_C.DATASET.VAL_IMAGE_SET = ''
_C.DATASET.TEST_IMAGE_SET = ''
_C.DATASET.TRAIN_ANNOTATION_FILE = ''
_C.DATASET.VAL_ANNOTATION_FILE = ''
_C.DATASET.TEST_ANNOTATION_FILE = ''
_C.DATASET.ONLY_USE_RELEVANT_DETS = True
_C.DATASET.ADD_IMAGE_AS_A_BOX = True
_C.DATASET.ZIP_MODE = False
_C.DATASET.CACHE_MODE = False
_C.DATASET.IGNORE_DB_CACHE = True
_C.DATASET.MASK_SIZE = 14
_C.DATASET.QA2R_NOQ = False
_C.DATASET.QA2R_AUG = False

# ------------------------------------------------------------------------------------- #
# Common network options
# ------------------------------------------------------------------------------------- #
_C.NETWORK = edict()
_C.NETWORK.BLIND = False
_C.NETWORK.NO_GROUNDING = False
_C.NETWORK.PARTIAL_PRETRAIN = ""
_C.NETWORK.PARTIAL_PRETRAIN_PREFIX_CHANGES = []
_C.NETWORK.PARTIAL_PRETRAIN_SEGMB_INIT = False
_C.NETWORK.FOR_MASK_VL_MODELING_PRETRAIN = False
_C.NETWORK.NO_OBJ_ATTENTION = False
_C.NETWORK.IMAGE_NUM_LAYERS = 50
_C.NETWORK.IMAGE_C5_DILATED = False
_C.NETWORK.IMAGE_STRIDE_IN_1x1 = False
_C.NETWORK.PIXEL_MEANS = ()
_C.NETWORK.PIXEL_STDS = ()
_C.NETWORK.IMAGE_FEAT_PRECOMPUTED = False
_C.NETWORK.IMAGE_PRETRAINED = ''
_C.NETWORK.IMAGE_PRETRAINED_EPOCH = 0
_C.NETWORK.IMAGE_FROZEN_BACKBONE_STAGES = [1, 2]
_C.NETWORK.IMAGE_FROZEN_BN = True
_C.NETWORK.IMAGE_FINAL_DIM = 512
_C.NETWORK.IMAGE_SEMANTIC = True
_C.NETWORK.OUTPUT_CONV5 = False
_C.NETWORK.QA_ONE_SENT = False
_C.NETWORK.BERT_MODEL_NAME = 'bert-base-uncased'
_C.NETWORK.BERT_PRETRAINED = ''
_C.NETWORK.BERT_PRETRAINED_EPOCH = 0
_C.NETWORK.BERT_FROZEN = True
_C.NETWORK.BERT_ALIGN_QUESTION = True
_C.NETWORK.BERT_ALIGN_ANSWER = True
_C.NETWORK.BERT_USE_LAYER = -2
_C.NETWORK.BERT_WITH_NSP_LOSS = False
_C.NETWORK.BERT_WITH_MLM_LOSS = False
_C.NETWORK.ENABLE_CNN_REG_LOSS = True
_C.NETWORK.CNN_REG_DROPOUT = 0.0
_C.NETWORK.CNN_LOSS_TOP = False
_C.NETWORK.CNN_LOSS_WEIGHT = 1.0
_C.NETWORK.ANS_LOSS_WEIGHT = 1.0
_C.NETWORK.ANSWER_FIRST = False
_C.NETWORK.LOAD_REL_HEAD = True

_C.NETWORK.VLBERT = edict()
# _C.NETWORK.VLBERT.vocab_size = None
_C.NETWORK.VLBERT.input_size = 1280
# 1: LN + [1x1 conv] 2: LN + [1x1 conv] + dropout 3: LN + [1x1 conv] + dropout + BertLayer
_C.NETWORK.VLBERT.input_transform_type = 1
_C.NETWORK.VLBERT.word_embedding_frozen = False
_C.NETWORK.VLBERT.obj_pos_id_relative = True
_C.NETWORK.VLBERT.hidden_size = 512
_C.NETWORK.VLBERT.visual_size = 512
_C.NETWORK.VLBERT.num_hidden_layers = 4
_C.NETWORK.VLBERT.num_attention_heads = 8
_C.NETWORK.VLBERT.intermediate_size = 2048
_C.NETWORK.VLBERT.hidden_act = "gelu"
_C.NETWORK.VLBERT.hidden_dropout_prob = 0.1
_C.NETWORK.VLBERT.attention_probs_dropout_prob = 0.1
_C.NETWORK.VLBERT.max_position_embeddings = 512
_C.NETWORK.VLBERT.type_vocab_size = 3
_C.NETWORK.VLBERT.vocab_size = 30522
_C.NETWORK.VLBERT.initializer_range = 0.02
_C.NETWORK.VLBERT.visual_scale_text_init = 0.0
_C.NETWORK.VLBERT.visual_scale_object_init = 0.0
_C.NETWORK.VLBERT.visual_ln = False
# 1: class embedding 2: class agnostic embedding 3: average of word embedding of text
_C.NETWORK.VLBERT.object_word_embed_mode = 1
_C.NETWORK.VLBERT.with_pooler = True
_C.NETWORK.VLBERT.position_padding_idx = -1

_C.NETWORK.CLASSIFIER_TYPE = "2fc"    # 2fc or 1fc
_C.NETWORK.CLASSIFIER_HIDDEN_SIZE = 1024
_C.NETWORK.CLASSIFIER_DROPOUT = 0.1
_C.NETWORK.CLASSIFIER_SIGMOID = False
_C.NETWORK.CLASSIFIER_SIGMOID_LOSS_POSITIVE_WEIGHT = 1.0

# ------------------------------------------------------------------------------------- #
# Common training related options
# ------------------------------------------------------------------------------------- #
_C.TRAIN = edict()
_C.TRAIN.LR_MULT = []
_C.TRAIN.VISUAL_SCALE_TEXT_LR_MULT = 1.0
_C.TRAIN.VISUAL_SCALE_OBJECT_LR_MULT = 1.0
_C.TRAIN.VISUAL_SCALE_CLIP_GRAD_NORM = -1
_C.TRAIN.SHUFFLE = True
_C.TRAIN.FLIP_PROB = 0.5
_C.TRAIN.BATCH_IMAGES = 1
_C.TRAIN.ASPECT_GROUPING = True
_C.TRAIN.RESUME = False
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 0
_C.TRAIN.OPTIMIZER = 'SGD'
_C.TRAIN.CLIP_GRAD_NORM = -1
_C.TRAIN.GRAD_ACCUMULATE_STEPS = 1
_C.TRAIN.LR = 0.1
_C.TRAIN.LR_SCHEDULE = 'step'  # step/triangle/plateau
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = ()
_C.TRAIN.WARMUP = False
_C.TRAIN.WARMUP_METHOD = 'linear'
_C.TRAIN.WARMUP_FACTOR = 1.0 / 3
_C.TRAIN.WARMUP_STEPS = 1000
_C.TRAIN.WD = 0.0001
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.FP16 = False
_C.TRAIN.FP16_LOSS_SCALE = 128.0
_C.TRAIN.LOSS_LOGGERS = [('ans_loss', 'AnsLoss'),
                         ('cnn_regularization_loss', 'CNNRegLoss')]

# ------------------------------------------------------------------------------------- #
# Common validation related options
# ------------------------------------------------------------------------------------- #
_C.VAL = edict()
_C.VAL.SHUFFLE = False
_C.VAL.FLIP_PROB = 0
_C.VAL.BATCH_IMAGES = 1

# ------------------------------------------------------------------------------------- #
# Common testing related options
# ------------------------------------------------------------------------------------- #
_C.TEST = edict()
_C.TEST.SHUFFLE = False
_C.TEST.FLIP_PROB = 0
_C.TEST.TEST_EPOCH = 0
_C.TEST.BATCH_IMAGES = 1


def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        if vk in config[k]:
                            if vk == 'LR_STEP':
                                config[k][vk] = tuple(float(s) for s in vv.split(','))
                            elif vk == 'LOSS_LOGGERS':
                                config[k][vk] = [tuple(str(s) for s in vvi.split(',')) for vvi in vv]
                            elif vk == "VLBERT" and isinstance(vv, dict):
                                for vvk, vvv in vv.items():
                                    if vvk in config[k][vk]:
                                        config[k][vk][vvk] = vvv
                                    else:
                                        raise ValueError("key {}.{}.{} not in config.py".format(k, vk, vvk))
                            else:
                                config[k][vk] = vv
                        else:
                            raise ValueError("key {}.{} not in config.py".format(k, vk))
                else:
                    if k == 'SCALES':
                        config[k] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("key {} not in config.py".format(k))
