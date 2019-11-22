# Prepare Pre-trained Models
Download pre-trained models and organize them as following:
```
code_root/
└── model/
    └── pretrained_model/
        ├── vl-bert-base-e2e.model
        ├── vl-bert-large-e2e.model
        ├── vl-bert-base-prec.model
        ├── vl-bert-large-prec.model
        ├── bert-base-uncased/
        │   ├── vocab.txt
        │   ├── bert_config.json
        │   └── pytorch_model.bin
        ├── bert-large-uncased/
        │   ├── vocab.txt
        │   ├── bert_config.json
        │   └── pytorch_model.bin
        └── resnet101-pt-vgbua-0000.model     
```


## VL-BERT

| Model Name         | Download Link    |
| ------------------ | ---------------  |
| vl-bert-base-e2e   | [GoogleDrive](https://drive.google.com/file/d/1jjV1ARYMs37tOaBalhJmwq7LcWeMai96/view?usp=sharing) / [BaiduPan](https://pan.baidu.com/s/1rl0Hl-iZZHL-3fj8hE_Uug) |
| vl-bert-large-e2e  | [GoogleDrive](https://drive.google.com/file/d/1YTHWWyP7Kq6zPySoEcTs3STaQdc5OJ7f/view?usp=sharing) / [BaiduPan](https://pan.baidu.com/s/1yqpDZRuGLsRXpklDgSC_Jw) |
| vl-bert-base-prec  | [GoogleDrive](https://drive.google.com/file/d/1YBFsyoWwz83VPzbimKymSBxE37gYtfgh/view?usp=sharing) / [BaiduPan](https://pan.baidu.com/s/1SvGbE2cjw8jEGWwSfJBFQQ) |
| vl-bert-large-prec | [GoogleDrive](https://drive.google.com/file/d/1REZLN7c3JCHVFoi_nEO-Nn6A4PTKIygG/view?usp=sharing) / [BaiduPan](https://pan.baidu.com/s/1k4eQe2rGGGVD24ZksJteNA) |

***Note***: models with suffix "e2e" means parameters of Fast-RCNN is tuned during pre-training, 
while "prec" means Fast-RCNN is fixed during pre-training and for effeciency the visual features is precomputed using
[bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention). 

## BERT & ResNet

Download following pre-trained BERT and ResNet and place them under this folder.

* BERT: [GoogleDrive](https://drive.google.com/file/d/14VceZht89V5i54-_xWiw58Rosa5NDL2H/view?usp=sharing) / [BaiduPan](https://pan.baidu.com/s/1dyYcw50eZznL02ilG676Yw)
* ResNet101 pretrained on Visual Genome: 
[GoogleDrive](https://drive.google.com/file/d/1qJYtsGw1SfAyvknDZeRBnp2cF4VNjiDE/view?usp=sharing) / [BaiduPan](https://pan.baidu.com/s/1_yfZG8VqbWmp5Kr9w2DKGQ) 
(converted from [caffe model](https://www.dropbox.com/s/wqada4qiv1dz9dk/resnet101_faster_rcnn_final.caffemodel?dl=1))