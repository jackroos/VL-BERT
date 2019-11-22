import torch
from roi_pool import ROIPool
from roi_align import ROIAlign

align = ROIAlign(output_size=(3, 3), spatial_scale=1.0, sampling_ratio=1)
pool = ROIPool(output_size=(3, 3), spatial_scale=1.0)

device = torch.device("cuda:0")

feature = torch.arange(81*2*3).view((2,3,9,9)).float().to(device)
rois = torch.Tensor([[0,0,0,9,9],[1,0,0,9,9],[1,0,0,7,7]]).to(device)

pooled = pool(feature,rois)
aligned = align(feature,rois)

import IPython
IPython.embed()
