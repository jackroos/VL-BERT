import torch
import os


def smart_load_model_state_dict(model, state_dict):
    parsed_state_dict = {}
    for k, v in state_dict.items():
        if k not in model.state_dict():
            if k.startswith('module.'):
                k = k[len('module.'):]
            else:
                k = 'module.' + k
        if k in model.state_dict():
            parsed_state_dict[k] = v
        else:
            raise ValueError('failed to match key of state dict smartly!')
    model.load_state_dict(parsed_state_dict)


def smart_resume(model, optimizer, validation_monitor, config, model_prefix, logger):
    if config.TRAIN.RESUME:
        print(('continue training from ', config.TRAIN.BEGIN_EPOCH))
        # load model
        model_filename = '{}-{:04d}.model'.format(model_prefix, config.TRAIN.BEGIN_EPOCH - 1)
        check_point = torch.load(model_filename, map_location=lambda storage, loc: storage)
        # model.load_state_dict(check_point['state_dict'])
        smart_load_model_state_dict(model, check_point['state_dict'])
        optimizer.load_state_dict(check_point['optimizer'])
        if 'validation_monitor' in check_point:
            validation_monitor.load_state_dict(check_point['validation_monitor'])
            print(
                'Best Val {}: {}, Epoch: {}'.format(validation_monitor.host_metric_name,
                                                    validation_monitor.best_val,
                                                    validation_monitor.best_epoch)
            )
    elif config.TRAIN.AUTO_RESUME:
        for epoch in range(config.TRAIN.END_EPOCH, config.TRAIN.BEGIN_EPOCH, -1):
            model_filename = '{}-{:04d}.model'.format(model_prefix, epoch - 1)
            if os.path.exists(model_filename):
                config.TRAIN.BEGIN_EPOCH = epoch
                check_point = torch.load(model_filename, map_location=lambda storage, loc: storage)
                # model.load_state_dict(check_point['state_dict'])
                smart_load_model_state_dict(model, check_point['state_dict'])
                optimizer.load_state_dict(check_point['optimizer'])
                if 'validation_monitor' in check_point:
                    validation_monitor.load_state_dict(check_point['validation_monitor'])
                    print(
                        'Best Val {}: {}, Epoch: {}'.format(validation_monitor.host_metric_name,
                                                            validation_monitor.best_val,
                                                            validation_monitor.best_epoch)
                    )
                logger.info("Auto continue training from {0}".format(model_filename))
                print("Auto continue training from {0}".format(model_filename))
                break


def smart_partial_load_model_state_dict(model, state_dict):
    parsed_state_dict = {}
    non_match_keys = []
    pretrained_keys = []
    for k, v in state_dict.items():
        if k not in model.state_dict():
            if k.startswith('module.'):
                k = k[len('module.'):]
            else:
                k = 'module.' + k
        if k in model.state_dict():
            parsed_state_dict[k] = v
            pretrained_keys.append(k)
        else:
            non_match_keys.append(k)
            # raise ValueError('failed to match key of state dict smartly!')

    non_pretrain_keys = [k for k in model.state_dict().keys() if k not in pretrained_keys]

    print("[Partial Load] partial load state dict of keys: {}".format(parsed_state_dict.keys()))
    print("[Partial Load] non matched keys: {}".format(non_match_keys))
    print("[Partial Load] non pretrain keys: {}".format(non_pretrain_keys))
    new_state_dict = model.state_dict()
    new_state_dict.update(parsed_state_dict)
    model.load_state_dict(new_state_dict)

