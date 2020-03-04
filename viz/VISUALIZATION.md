# Visualization

The code is based on [bertviz](https://github.com/jessevig/bertviz), a nice tool for BERT visualization.

## Prepare

* Change work directory to this directory.

  ```bash
  cd ./viz
  ```

* Create a soft link to the data folder (If you are working on Windows, please modify the data path in the jupyter notebook by yourself).

  ```bash
  ln -s ../data ./
  ```

* Download and unzip COCO val2017: [images](http://images.cocodataset.org/zips/val2017.zip), [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip), place them under ```./data/coco```.

* (Optional) Download pre-trained models as described in [PREPARE_PRETRAINED_MODELS.md](../model/pretrained_model/PREPARE_PRETRAINED_MODELS.md), if you want to precompute all attention maps by yourself.

## Pre-compute attention maps
* Pre-computing all attention maps on COCO val2017: 
  
  ```bash
  python pretrain/vis_attention_maps.py --cfg cfgs/pretrain/vis_attention_maps_coco.yaml --save-dir ./vl-bert_viz
  ```
* We provide 100 pre-computed attention maps of vl-bert-base-e2e on COCO val2017: [GoogleDrive](https://drive.google.com/file/d/1TFfqArX3lwOPQ8EklZ6px5-gvOvoGdTr/view?usp=sharing) [BaiduPan](https://pan.baidu.com/s/1l0T5vAuklQTrAmD3wbJ7uQ), please download and unzip it into ```./data```.

## Visualization on Jupyter Notebook
* Open Jupyter Notebook  in this directory and select ```model_view_vl-bert_coco.ipynb```.
    ```bash
    jupyter notebook
    ```

* Run all cells in the notebook in order.

* Browse attention maps in the last cell, you can change the image id to visualize other examples.

 
