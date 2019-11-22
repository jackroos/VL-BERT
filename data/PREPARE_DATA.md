# Prepare Data

Download datasets as you need, and organize them as following:
 ```
code_root/
└── data/
    ├── conceptual-captions/
    │   ├── train_image/
    │   ├── val_image/
    │   ├── train_frcnn/
    │   ├── val_frcnn/
    │   ├── train.json
    │   ├── val.json
    │   ├── train_frcnn.json
    │   └── val_frcnn.json
    ├── en_corpus/
    │   ├── wiki.doc
    │   └── bc1g.doc
    ├── vcr/
    │   ├── vcr1images/
    │   ├── train.jsonl
    │   ├── val.jsonl
    │   └── test.jsonl
    └── coco/
        ├── train2014/
        ├── val2014/
        ├── test2015/
        ├── annotations/
        ├── vqa/
        ├── refcoco+/
        │   └── proposal/
        └── vgbua_res101_precomputed/
            ├── trainval2014_resnet101_faster_rcnn_genome
            └── test2015_resnet101_faster_rcnn_genome
        
 ```
## Pre-training Data

### Conceptual Captions
See [ReadMe.txt](./conceptual-captions/ReadMe.txt).

### English Wikipedia & BooksCorpus
* Wikipedia: [GoogleDrive](https://drive.google.com/file/d/1rZJ-Nj_SSqwu85tME3wbN8tfGhljfAsf/view?usp=sharing) / [BaiduPan](https://pan.baidu.com/s/1HSgUZXRESxVnx9ATOHwSrQ)
* BooksCorpus: [GoogleDrive](https://drive.google.com/file/d/16T5EYqIjO-tAj1OFxz6bnnzEABCusCcv/view?usp=sharing) / [BaiduPan](https://pan.baidu.com/s/1797WFFUTnRJakgGxefSrBg)

## Fine-tuning Data

### VCR
* Download and unzip images & annotations from [here](https://visualcommonsense.com/download/).

### VQA & RefCOCO+

#### Common
* Download and unzip COCO 2014 images & annotations from [here](http://cocodataset.org/#download).

#### VQA
* Download and unzip annotations from [here](https://visualqa.org/download.html) (including "VQA Annotations" and "VQA Input Questions"), 
place all these files directly under ```./data/coco/vqa```.
* Download and unzip following precomputed boxes & features into ```./data/coco/vgbua_res101_precomputed```.
    * train2014 + val2014: [GoogleDrive](https://drive.google.com/file/d/1KyLyqTqBsMX7QtLTma0xFrmhAzdQDUed/view?usp=sharing) / [BaiduPan](https://pan.baidu.com/s/1Udtoi2TC-nAimZf-vLC9PQ)
    * test2015: [GoogleDrive](https://drive.google.com/file/d/10nM3kRz2c827aqwVvLnv430YYFp0po6O/view?usp=sharing) / [BaiduPan](https://pan.baidu.com/s/1wd3rWfPWLBhGkEc10N9e1Q)

* Download answer vocabulary from [GoogleDrive](https://drive.google.com/file/d/1CPnYcOgIOP5CZkp_KChuCg54_Ljr6-fp/view?usp=sharing) / [BaiduPan](https://pan.baidu.com/s/1IvPsH-mmqHi2glgznaBuYw), place it under the folder ```./data/coco/vqa/```.
    
#### RefCOCO+

* Download and unzip [annotations](http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip), place all files in ```refcoco+/``` directly under ```./data/coco/refcoco+```.
* Download [region proposals](http://bvision.cs.unc.edu/licheng/MattNet/detections.zip), place all files in ```detections/refcoco+_unc``` directly under ```./data/coco/refcoco+/proposal```.