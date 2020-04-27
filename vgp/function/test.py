import os
import pprint
import shutil

import pandas as pd
from tqdm import trange
import numpy as np
import torch

from common.utils.load import smart_load_model_state_dict
from common.trainer import to_cuda
from common.utils.create_logger import create_logger
from common.metrics.vgp_metrics import compute_metrics_sentence_level
from vgp.data.build import make_dataloader
from vgp.modules import *

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as Apex_DDP
except ImportError:
    pass
    #raise ImportError("Please install apex from https://www.github.com/nvidia/apex if you want to use fp16.")

# submit csv should contain following columns:
# annot_id,
# answer_0,answer_1,answer_2,answer_3,
# rationale_conditioned_on_a0_0,rationale_conditioned_on_a0_1,rationale_conditioned_on_a0_2,rationale_conditioned_on_a0_3,
# rationale_conditioned_on_a1_0,rationale_conditioned_on_a1_1,rationale_conditioned_on_a1_2,rationale_conditioned_on_a1_3,
# rationale_conditioned_on_a2_0,rationale_conditioned_on_a2_1,rationale_conditioned_on_a2_2,rationale_conditioned_on_a2_3,
# rationale_conditioned_on_a3_0,rationale_conditioned_on_a3_1,rationale_conditioned_on_a3_2,rationale_conditioned_on_a3_3


@torch.no_grad()
def test_net(args, config, ckpt_path=None, save_path=None, save_name=None):
    if save_path is None:
        logger, test_output_path = create_logger(config.OUTPUT_PATH, args.cfg, config.DATASET.TEST_IMAGE_SET,
                                                 split='test')
        save_path = test_output_path
    if save_name is None:
        save_name = config.MODEL_PREFIX
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    result_csv_path = os.path.join(save_path,
                                   '{}_test_result.csv'.format(save_name))
    if not(args.use_cache and os.path.isfile(result_csv_path)):
        print('test net...')
        pprint.pprint(args)
        pprint.pprint(config)
        device_ids = [int(d) for d in config.GPUS.split(',')]
        # os.environ['CUDA_VISIBLE_DEVICES'] = config.GPUS

        if ckpt_path is None:
            _, train_output_path = create_logger(config.OUTPUT_PATH, args.cfg, config.DATASET.IMAGE_SET,
                                                 split='train')
            model_prefix = os.path.join(train_output_path, config.MODEL_PREFIX)
            ckpt_path = '{}-best.model'.format(model_prefix)
            print('Use best checkpoint {}...'.format(ckpt_path))

        shutil.copy2(ckpt_path, os.path.join(save_path, '{}_test_ckpt_{}.model'.format(config.MODEL_PREFIX, config.DATASET.TASK)))

        # torch.backends.cudnn.enabled = False
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        # get network
        model = eval(config.MODULE)(config)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        else:
            model = model.cuda()
        if args.fp16:
            [model] = amp.initialize([model],
                                     opt_level='O2',
                                     keep_batchnorm_fp32=False)
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        smart_load_model_state_dict(model, checkpoint['state_dict'])

        # loader
        test_loader = make_dataloader(config, mode='test', distributed=False)
        test_dataset = test_loader.dataset
        test_database = test_dataset.database

        # test
        test_probs = []
        test_ids = []
        sentence_labels = []
        cur_id = 0
        model.eval()
        for nbatch, batch in zip(trange(len(test_loader)), test_loader):
            batch = to_cuda(batch)
            output = model(*batch)
            probs = torch.sigmoid(output['sentence_label_logits'].float()).detach().cpu().numpy()
            sentence_labels.append([test_database[cur_id + k]['label'] for k in range(batch_size)])
            batch_size = probs.shape[0]
            test_ids.append([test_database[cur_id + k]['pair_id'] for k in range(batch_size)])
            cur_id += batch_size
        test_probs = np.concatenate(test_probs, axis=0)
        test_ids = np.concatenate(test_ids, axis=0)
        sentence_labels = np.concatenate(sentence_labels, axis=0)

        # generate final result csv
        columns = ["sentence_prob"]
        dataframe = pd.DataFrame(data=test_probs, columns=columns)
        dataframe['pair_id'] = test_ids
        dataframe['sentence_label'] = sentence_labels
        dataframe = dataframe.set_index('pair_id', drop=True)

        # Save predictions
        dataframe.to_csv(result_csv_path)
        print('result csv saved to {}.'.format(result_csv_path))
    else:
        print("Cache found in {}, skip test prediction!".format(result_csv_path))
        dataframe = pd.read_csv(result_csv_path)

    # Evaluate predictions
    pred_sent_probs = np.array(dataframe["sentence_prob"].values)
    sentence_labels = np.array(dataframe["sentence_label"].values)
    for metric in ["accuracy"]:
        result = compute_metrics_sentence_level(metric, pred_sent_probs, sentence_labels)
        print("{} on test set is: {}".format(metric, str(result)))
