import os
import pprint
import shutil

import pandas as pd
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn.functional as F

from common.utils.load import smart_load_model_state_dict
from common.trainer import to_cuda
from common.utils.create_logger import create_logger
from vcr.data.build import make_dataloader
from vcr.modules import *

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
                                   '{}_test_result_{}.csv'.format(save_name, config.DATASET.TASK))
    if args.use_cache and os.path.isfile(result_csv_path):
        print("Cache found in {}, skip test!".format(result_csv_path))
        return result_csv_path

    print('test net...')
    pprint.pprint(args)
    pprint.pprint(config)
    device_ids = [int(d) for d in config.GPUS.split(',')]
    # os.environ['CUDA_VISIBLE_DEVICES'] = config.GPUS

    if ckpt_path is None:
        _, train_output_path = create_logger(config.OUTPUT_PATH, args.cfg, config.DATASET.TRAIN_IMAGE_SET,
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
    cur_id = 0
    model.eval()
    for nbatch, batch in zip(trange(len(test_loader)), test_loader):
    # for nbatch, batch in tqdm(enumerate(test_loader)):
        batch = to_cuda(batch)
        if config.DATASET.TASK == 'Q2A':
            output = model(*batch)
            probs = F.softmax(output['label_logits'].float(), dim=1)
            batch_size = probs.shape[0]
            test_probs.append(probs.float().detach().cpu().numpy())
            test_ids.append([test_database[cur_id + k]['annot_id'] for k in range(batch_size)])
            cur_id += batch_size
        elif config.DATASET.TASK == 'QA2R':
            conditioned_probs = []
            for a_id in range(4):
                q_index_in_batch = test_loader.dataset.data_names.index('question')
                q_align_mat_index_in_batch = test_loader.dataset.data_names.index('question_align_matrix')
                batch_ = [*batch]
                batch_[q_index_in_batch] = batch[q_index_in_batch][:, a_id, :, :]
                batch_[q_align_mat_index_in_batch] = batch[q_align_mat_index_in_batch][:, a_id, :, :]
                output = model(*batch_)
                probs = F.softmax(output['label_logits'].float(), dim=1)
                conditioned_probs.append(probs.float().detach().cpu().numpy())
            conditioned_probs = np.concatenate(conditioned_probs, axis=1)
            test_probs.append(conditioned_probs)
            test_ids.append([test_database[cur_id + k]['annot_id'] for k in range(conditioned_probs.shape[0])])
            cur_id += conditioned_probs.shape[0]
        else:
            raise ValueError('Not Support Task {}'.format(config.DATASET.TASK))
    test_probs = np.concatenate(test_probs, axis=0)
    test_ids = np.concatenate(test_ids, axis=0)

    result_npy_path = os.path.join(save_path, '{}_test_result_{}.npy'.format(save_name, config.DATASET.TASK))
    np.save(result_npy_path, test_probs)
    print('result npy saved to {}.'.format(result_npy_path))

    # generate final result csv
    if config.DATASET.TASK == 'Q2A':
        columns = ['answer_{}'.format(i) for i in range(4)]
    else:
        columns = ['rationale_conditioned_on_a{}_{}'.format(i, j) for i in range(4) for j in range(4)]
    dataframe = pd.DataFrame(data=test_probs, columns=columns)
    dataframe['annot_id'] = test_ids
    dataframe = dataframe.set_index('annot_id', drop=True)

    dataframe.to_csv(result_csv_path)
    print('result csv saved to {}.'.format(result_csv_path))
    return result_csv_path


def merge_result(q2a_result_file, qa2r_result_file, output_file):
    left_df = pd.read_csv(q2a_result_file)
    right_df = pd.read_csv(qa2r_result_file)
    merged_df = pd.merge(left_df, right_df, on='annot_id')
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    merged_df.to_csv(output_file, index=False)
    print('merged result csv saved to {}.'.format(output_file))



