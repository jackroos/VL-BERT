0. create a python 2.7 conda environment:

   conda create -n cc python=2.7 pip
   conda activate cc
   pip install Cython numpy Pillow

1. download "Train_GCC-training.tsv" and "Validation_GCC-1.1.0-Validation.tsv" from
   https://ai.google.com/research/ConceptualCaptions/download
	
2. move "Train_GCC-training.tsv" and "Validation_GCC-1.1.0-Validation.tsv" into
   conceptual-captions/utils/
   
3. cd to conceptual-captions/utils/

4. python gen_train4download.py
   python gen_val4download.py

5. sh download_train.sh
   sh download_val.sh
   
   * you may need to run these commands multiple times to avoid temporary network failures and download as more images as possible
   * these commands will skip already successfully downloaded images, so don't worry about wasting time

6. 1) zip (without compression) "train_image" by
   
   cd ../train_image
   zip -0 ../train_image.zip ./*
   cd ../utils/
   
   2) zip (without compression) "val_image" by
   
   cd ../val_image
   zip -0 ../val_image.zip ./*
   cd ../utils/
   
7. python gen_train_image_json.py
   python gen_val_image_json.py
   
   
8. git clone https://github.com/jackroos/bottom-up-attention and follow "Installation" :

   1) Build the Cython modules

   cd $REPO_ROOT/lib
   make
   
   2) Build Caffe and pycaffe

   cd $REPO_ROOT/caffe
   # Now follow the Caffe installation instructions here:
   #   http://caffe.berkeleyvision.org/installation.html

   # If you're experienced with Caffe and have all of the requirements installed
   # and your Makefile.config in place, then simply do:
   make -j8 && make pycaffe
   
   3) Download pretrained model (https://www.dropbox.com/s/5xethd2nxa8qrnq/resnet101_faster_rcnn_final.caffemodel?dl=1), and put it under data/faster_rcnn_models.
   
9. python ./tools/generate_tsv_v2.py --gpu 0,1,2,3,4,5,6,7 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split conceptual_captions_train --data_root {Conceptual_Captions_Root} --out {Conceptual_Captions_Root}/train_frcnn/

   python ./tools/generate_tsv_v2.py --gpu 0,1,2,3,4,5,6,7 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split conceptual_captions_val --data_root {Conceptual_Captions_Root} --out {Conceptual_Captions_Root}/val_frcnn/
   
10. zip (without compression) "train_frcnn" and "val_frcnn" similar to step 6.
