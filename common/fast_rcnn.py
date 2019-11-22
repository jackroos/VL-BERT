import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from common.backbone.resnet.resnet import *
from common.backbone.resnet.resnet import Bottleneck, BasicBlock
from common.backbone.resnet.resnet import model_urls

from common.lib.roi_pooling.roi_pool import ROIPool
from common.lib.roi_pooling.roi_align import ROIAlign
from common.utils.flatten import Flattener
from common.utils.pad_sequence import pad_sequence
from common.utils.bbox import coordinate_embeddings


class FastRCNN(nn.Module):
    def __init__(self, config, average_pool=True, final_dim=768, enable_cnn_reg_loss=False):
        """
        :param config:
        :param average_pool: whether or not to average pool the representations
        :param final_dim:
        :param is_train:
        """
        super(FastRCNN, self).__init__()
        self.average_pool = average_pool
        self.enable_cnn_reg_loss = enable_cnn_reg_loss
        self.final_dim = final_dim
        self.image_feat_precomputed = config.NETWORK.IMAGE_FEAT_PRECOMPUTED
        if self.image_feat_precomputed:
            if config.NETWORK.IMAGE_SEMANTIC:
                self.object_embed = torch.nn.Embedding(num_embeddings=81, embedding_dim=128)
            else:
                self.object_embed = None
        else:
            self.stride_in_1x1 = config.NETWORK.IMAGE_STRIDE_IN_1x1
            self.c5_dilated = config.NETWORK.IMAGE_C5_DILATED
            self.num_layers = config.NETWORK.IMAGE_NUM_LAYERS
            self.pretrained_model_path = '{}-{:04d}.model'.format(config.NETWORK.IMAGE_PRETRAINED,
                                                                  config.NETWORK.IMAGE_PRETRAINED_EPOCH) if config.NETWORK.IMAGE_PRETRAINED != '' else None
            self.output_conv5 = config.NETWORK.OUTPUT_CONV5
            if self.num_layers == 18:
                self.backbone = resnet18(pretrained=True, pretrained_model_path=self.pretrained_model_path,
                                         expose_stages=[4])
                block = BasicBlock
            elif self.num_layers == 34:
                self.backbone = resnet34(pretrained=True, pretrained_model_path=self.pretrained_model_path,
                                         expose_stages=[4])
                block = BasicBlock
            elif self.num_layers == 50:
                self.backbone = resnet50(pretrained=True, pretrained_model_path=self.pretrained_model_path,
                                         expose_stages=[4], stride_in_1x1=self.stride_in_1x1)
                block = Bottleneck
            elif self.num_layers == 101:
                self.backbone = resnet101(pretrained=True, pretrained_model_path=self.pretrained_model_path,
                                          expose_stages=[4], stride_in_1x1=self.stride_in_1x1)
                block = Bottleneck
            elif self.num_layers == 152:
                self.backbone = resnet152(pretrained=True, pretrained_model_path=self.pretrained_model_path,
                                          expose_stages=[4], stride_in_1x1=self.stride_in_1x1)
                block = Bottleneck
            else:
                raise NotImplemented

            output_size = (14, 14)
            self.roi_align = ROIAlign(output_size=output_size, spatial_scale=1.0 / 16)

            if config.NETWORK.IMAGE_SEMANTIC:
                self.object_embed = torch.nn.Embedding(num_embeddings=81, embedding_dim=128)
            else:
                self.object_embed = None
                self.mask_upsample = None

            self.roi_head_feature_extractor = self.backbone._make_layer(block=block, planes=512, blocks=3,
                                                                        stride=2 if not self.c5_dilated else 1,
                                                                        dilation=1 if not self.c5_dilated else 2,
                                                                        stride_in_1x1=self.stride_in_1x1)

            if average_pool:
                self.head = torch.nn.Sequential(
                    self.roi_head_feature_extractor,
                    nn.AvgPool2d(7 if not self.c5_dilated else 14, stride=1),
                    Flattener()
                )
            else:
                self.head = self.roi_head_feature_extractor

            if config.NETWORK.IMAGE_FROZEN_BN:
                for module in self.roi_head_feature_extractor.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        for param in module.parameters():
                            param.requires_grad = False

            frozen_stages = config.NETWORK.IMAGE_FROZEN_BACKBONE_STAGES
            if 5 in frozen_stages:
                for p in self.roi_head_feature_extractor.parameters():
                    p.requires_grad = False
                frozen_stages = [stage for stage in frozen_stages if stage != 5]
            self.backbone.frozen_parameters(frozen_stages=frozen_stages,
                                            frozen_bn=config.NETWORK.IMAGE_FROZEN_BN)

            if self.enable_cnn_reg_loss:
                self.regularizing_predictor = torch.nn.Linear(2048, 81)

        self.obj_downsample = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(2 * 2048 + (128 if config.NETWORK.IMAGE_SEMANTIC else 0), final_dim),
            torch.nn.ReLU(inplace=True),
        )

    def init_weight(self):
        if not self.image_feat_precomputed:
            if self.pretrained_model_path is None:
                pretrained_model = model_zoo.load_url(model_urls['resnet{}'.format(self.num_layers)])
            else:
                pretrained_model = torch.load(self.pretrained_model_path, map_location=lambda storage, loc: storage)
            roi_head_feat_dict = {k[len('layer4.'):]: v for k, v in pretrained_model.items() if k.startswith('layer4.')}
            self.roi_head_feature_extractor.load_state_dict(roi_head_feat_dict)
            if self.output_conv5:
                self.conv5.load_state_dict(roi_head_feat_dict)

    def bn_eval(self):
        if not self.image_feat_precomputed:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

    def forward(self, images, boxes, box_mask, im_info, classes=None, segms=None, mvrc_ops=None, mask_visual_embed=None):
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param boxes: [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :return: object reps [batch_size, max_num_objects, dim]
        """

        box_inds = box_mask.nonzero()
        obj_labels = classes[box_inds[:, 0], box_inds[:, 1]].type(torch.long) if classes is not None else None
        assert box_inds.shape[0] > 0

        if self.image_feat_precomputed:
            post_roialign = boxes[box_inds[:, 0], box_inds[:, 1]][:, 4:]
            boxes = boxes[:, :, :4]
        else:
            img_feats = self.backbone(images)
            rois = torch.cat((
                box_inds[:, 0, None].type(boxes.dtype),
                boxes[box_inds[:, 0], box_inds[:, 1]],
            ), 1)
            roi_align_res = self.roi_align(img_feats['body4'], rois).type(images.dtype)

            if segms is not None:
                pool_layers = self.head[1:]
                post_roialign = self.roi_head_feature_extractor(roi_align_res)
                post_roialign = post_roialign * segms[box_inds[:, 0], None, box_inds[:, 1]].to(dtype=post_roialign.dtype)
                for _layer in pool_layers:
                    post_roialign = _layer(post_roialign)
            else:
                post_roialign = self.head(roi_align_res)

            # Add some regularization, encouraging the model to keep giving decent enough predictions
            if self.enable_cnn_reg_loss:
                obj_logits = self.regularizing_predictor(post_roialign)
                cnn_regularization = F.cross_entropy(obj_logits, obj_labels)[None]

        feats_to_downsample = post_roialign if (self.object_embed is None or obj_labels is None) else \
            torch.cat((post_roialign, self.object_embed(obj_labels)), -1)
        if mvrc_ops is not None and mask_visual_embed is not None:
            _to_masked = (mvrc_ops == 1)[box_inds[:, 0], box_inds[:, 1]]
            feats_to_downsample[_to_masked] = mask_visual_embed
        coord_embed = coordinate_embeddings(
            torch.cat((boxes[box_inds[:, 0], box_inds[:, 1]], im_info[box_inds[:, 0], :2]), 1),
            256
        )
        feats_to_downsample = torch.cat((coord_embed.view((coord_embed.shape[0], -1)), feats_to_downsample), -1)
        final_feats = self.obj_downsample(feats_to_downsample)

        # Reshape into a padded sequence - this is expensive and annoying but easier to implement and debug...
        obj_reps = pad_sequence(final_feats, box_mask.sum(1).tolist())
        post_roialign = pad_sequence(post_roialign, box_mask.sum(1).tolist())

        # DataParallel compatibility
        obj_reps_padded = obj_reps.new_zeros((obj_reps.shape[0], boxes.shape[1], obj_reps.shape[2]))
        obj_reps_padded[:, :obj_reps.shape[1]] = obj_reps
        obj_reps = obj_reps_padded
        post_roialign_padded = post_roialign.new_zeros((post_roialign.shape[0], boxes.shape[1], post_roialign.shape[2]))
        post_roialign_padded[:, :post_roialign.shape[1]] = post_roialign
        post_roialign = post_roialign_padded

        # Output
        output_dict = {
            'obj_reps_raw': post_roialign,
            'obj_reps': obj_reps,
        }
        if (not self.image_feat_precomputed) and self.enable_cnn_reg_loss:
            output_dict.update({'obj_logits': obj_logits,
                                'obj_labels': obj_labels,
                                'cnn_regularization_loss': cnn_regularization})

        if (not self.image_feat_precomputed) and self.output_conv5:
            image_feature = self.img_head(img_feats['body4'])
            output_dict['image_feature'] = image_feature

        return output_dict
