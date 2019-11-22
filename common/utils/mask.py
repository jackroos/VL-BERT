from skimage.draw import polygon
import torch


def generate_instance_mask(seg_polys, box, mask_size=(14, 14), dtype=torch.float32, copy=True):
    """
    Generate instance mask from polygon
    :param seg_poly: torch.Tensor, (N, 2), (x, y) coordinate of N vertices of segmented foreground polygon
    :param box: array-like, (4, ), (xmin, ymin, xmax, ymax), instance bounding box
    :param mask_size: tuple, (mask_height, mask_weight)
    :param dtype: data type of generated mask
    :param copy: whether copy seg_polys to a new tensor first
    :return: torch.Tensor, of mask_size, instance mask
    """
    mask = torch.zeros(mask_size, dtype=dtype)
    w_ratio = float(mask_size[0]) / (box[2] - box[0] + 1)
    h_ratio = float(mask_size[1]) / (box[3] - box[1] + 1)

    # import IPython
    # IPython.embed()

    for seg_poly in seg_polys:
        if copy:
            seg_poly = seg_poly.detach().clone()
        seg_poly = seg_poly.type(torch.float32)
        seg_poly[:, 0] = (seg_poly[:, 0] - box[0]) * w_ratio
        seg_poly[:, 1] = (seg_poly[:, 1] - box[1]) * h_ratio
        rr, cc = polygon(seg_poly[:, 1].clamp(min=0, max=mask_size[1] - 1),
                         seg_poly[:, 0].clamp(min=0, max=mask_size[0] - 1))

        mask[rr, cc] = 1
    return mask




