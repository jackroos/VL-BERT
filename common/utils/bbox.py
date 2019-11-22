import torch


def nonlinear_transform(ex_rois, gt_rois):
    """
    compute bounding box regression targets from ex_rois to gt_rois
    :param ex_rois: [k, 4] ([x1, y1, x2, y2])
    :param gt_rois: [k, 4] (corresponding gt_boxes [x1, y1, x2, y2] )
    :return: bbox_targets: [k, 4]
    """
    assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1.0)

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * (gt_widths - 1.0)
    gt_ctr_y = gt_rois[:, 1] + 0.5 * (gt_heights - 1.0)

    targets_dx = (gt_ctr_x - ex_ctr_x) / (ex_widths + 1e-6)
    targets_dy = (gt_ctr_y - ex_ctr_y) / (ex_heights + 1e-6)
    targets_dw = torch.log(gt_widths / (ex_widths).clamp(min=1e-6))
    targets_dh = torch.log(gt_heights / ((ex_heights).clamp(min=1e-6)))

    targets = torch.cat(
        (targets_dx.view(-1, 1), targets_dy.view(-1, 1), targets_dw.view(-1, 1), targets_dh.view(-1, 1)), dim=-1)
    return targets


def coordinate_embeddings(boxes, dim):
    """
    Coordinate embeddings of bounding boxes
    :param boxes: [K, 6] ([x1, y1, x2, y2, w_image, h_image])
    :param dim: sin/cos embedding dimension
    :return: [K, 4, 2 * dim]
    """

    num_boxes = boxes.shape[0]
    w = boxes[:, 4]
    h = boxes[:, 5]

    # transform to (x_c, y_c, w, h) format
    boxes_ = boxes.new_zeros((num_boxes, 4))
    boxes_[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    boxes_[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    boxes_[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes_[:, 3] = boxes[:, 3] - boxes[:, 1]
    boxes = boxes_

    # position
    pos = boxes.new_zeros((num_boxes, 4))
    pos[:, 0] = boxes[:, 0] / w * 100
    pos[:, 1] = boxes[:, 1] / h * 100
    pos[:, 2] = boxes[:, 2] / w * 100
    pos[:, 3] = boxes[:, 3] / h * 100

    # sin/cos embedding
    dim_mat = 1000 ** (torch.arange(dim, dtype=boxes.dtype, device=boxes.device) / dim)
    sin_embedding = (pos.view((num_boxes, 4, 1)) / dim_mat.view((1, 1, -1))).sin()
    cos_embedding = (pos.view((num_boxes, 4, 1)) / dim_mat.view((1, 1, -1))).cos()

    return torch.cat((sin_embedding, cos_embedding), dim=-1)


def bbox_iou_py_vectorized(boxes, query_boxes):
    n_ = boxes.shape[0]
    k_ = query_boxes.shape[0]
    n_mesh, k_mesh = torch.meshgrid([torch.arange(n_), torch.arange(k_)])
    n_mesh = n_mesh.contiguous().view(-1)
    k_mesh = k_mesh.contiguous().view(-1)
    boxes = boxes[n_mesh]
    query_boxes = query_boxes[k_mesh]

    x11, y11, x12, y12 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x21, y21, x22, y22 = query_boxes[:, 0], query_boxes[:, 1], query_boxes[:, 2], query_boxes[:, 3]
    xA = torch.max(x11, x21)
    yA = torch.max(y11, y21)
    xB = torch.min(x12, x22)
    yB = torch.min(y12, y22)
    interArea = torch.clamp(xB - xA + 1, min=0) * torch.clamp(yB - yA + 1, min=0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + boxBArea - interArea)

    return iou.view(n_, k_).to(boxes.device)






