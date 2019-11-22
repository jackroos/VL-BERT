---
RNG_SEED: 12345
OUTPUT_PATH: './output/pretrain/vl-bert'
MODULE: ResNetVLBERTForPretrainingMultitask
GPUS: '0,1,2,3'
LOG_FREQUENT: 100
VAL_FREQUENT: 1
CHECKPOINT_FREQUENT: 1
MODEL_PREFIX: 'vl-bert_base_res101_pretrain_multitask'
NUM_WORKERS_PER_GPU: 4
SCALES:
- 600
- 1000

DATASET:

- DATASET: conceptual_captions
  APPEND_INDEX: false
  DATASET_PATH: './data/conceptual-captions'
  ROOT_PATH: './'
  TRAIN_IMAGE_SET: 'train'
  VAL_IMAGE_SET: 'val'
  TEST_IMAGE_SET: 'val'
  ADD_IMAGE_AS_A_BOX: true
  ZIP_MODE: false
  CACHE_MODE: false
  IGNORE_DB_CACHE: false
  MASK_SIZE: 14

- DATASET: general_corpus
  TRAIN_ANNOTATION_FILE: './data/en_corpus/bc1g.doc+./data/en_corpus/wiki.doc'
  VAL_ANNOTATION_FILE: './data/en_corpus/bc1g.doc'
  TEST_ANNOTATION_FILE: './data/en_corpus/bc1g.doc'
  SEQ_LEN: 64
  MIN_SEQ_LEN: 64

NETWORK:
  PARTIAL_PRETRAIN: ""
  PARTIAL_PRETRAIN_PREFIX_CHANGES: []
  IMAGE_NUM_LAYERS: 101
  IMAGE_C5_DILATED: true
  IMAGE_STRIDE_IN_1x1: true
  PIXEL_MEANS:
  - 102.9801
  - 115.9465
  - 122.7717
  PIXEL_STDS:
  - 1.0
  - 1.0
  - 1.0
  IMAGE_FEAT_PRECOMPUTED: false
  IMAGE_PRETRAINED: './model/pretrained_model/resnet101-pt-vgbua'
  IMAGE_PRETRAINED_EPOCH: 0
  IMAGE_FROZEN_BACKBONE_STAGES:
  - 1
  - 2
  IMAGE_FROZEN_BN: true
  IMAGE_FINAL_DIM: 768
  IMAGE_SEMANTIC: false
  OUTPUT_CONV5: false
  BERT_MODEL_NAME: './model/pretrained_model/bert-base-uncased'
  BERT_PRETRAINED: ''
  BERT_PRETRAINED_EPOCH: 0
  BERT_FROZEN: false
  ENABLE_CNN_REG_LOSS: false
  MLM_LOSS_NORM_IN_BATCH_FIRST: false
  MVRC_LOSS_NORM_IN_BATCH_FIRST: false
  WITH_REL_LOSS: false
  WITH_MLM_LOSS: true
  WITH_MVRC_LOSS: true

  VLBERT:
    with_pooler: false
    input_transform_type: 1
    visual_size: 768
    hidden_size: 768
    num_hidden_layers: 12
    num_attention_heads: 12
    intermediate_size: 3072
    hidden_act: "gelu"
    hidden_dropout_prob: 0.1
    attention_probs_dropout_prob: 0.1
    max_position_embeddings: 512
    type_vocab_size: 3
    vocab_size: 30522
    initializer_range: 0.02
    visual_scale_text_init: 0.0
    visual_scale_object_init: 0.0
    visual_ln: true
    pos_embedding_frozen: false

TRAIN:
  SHUFFLE: true
  FLIP_PROB: 0.5
  BATCH_IMAGES:
  - 8
  - 8
  ASPECT_GROUPING: false
  RESUME: false
  AUTO_RESUME: true
  BEGIN_EPOCH: 0
  END_EPOCH: 10
  OPTIMIZER: 'AdamW'
  CLIP_GRAD_NORM: 10
  LR: 1.0e-7
  LR_SCHEDULE: 'triangle'
  WD: 0.0001
  WARMUP: true
  WARMUP_METHOD: 'linear'
  WARMUP_FACTOR: 0.0
  WARMUP_STEPS: 16000
  FP16: true
  FP16_LOSS_SCALE: 'dynamic'
  LOSS_LOGGERS:
  - "mlm_loss_wvc,MLMLossWVC"
  - "mlm_loss_aux,MLMLossAUX"
  - "mvrc_loss,MVRCLoss"

VAL:
  SHUFFLE: false
  FLIP_PROB: 0
  BATCH_IMAGES:
  - 8
  - 8

TEST:
  SHUFFLE: false
  FLIP_PROB: 0
  TEST_EPOCH: 0
  BATCH_IMAGES:
  - 8
  - 8