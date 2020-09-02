# 3D-ResNets-Action-Recognition-Paddle
## 百度顶会论文复现营 
https://aistudio.baidu.com/aistudio/education/group/info/1340

**This is a PaddlePaddle Implementation of the [paper](https://arxiv.org/abs/2004.04968)**  
Hirokatsu Kataoka, Tenga Wakamiya, Kensho Hara, and Yutaka Satoh, "Would Mega-scale Datasets Further Enhance Spatiotemporal 3D CNNs", arXiv preprint, arXiv:2004.04968, 2020.


## Requirements

* [PaddlePaddle](https://www.paddlepaddle.org.cn/) (1.8.3 required)

```bash
python -m pip install paddlepaddle-gpu==1.8.3.post97 -i https://mirror.baidu.com/pypi/simple
```

* FFmpeg, FFprobe

* Python 3

## Pre-trained models

Pre-trained models are available [here](https://drive.google.com/file/d/1CmLeDqCGs52Ev2GWkoTKvvJ6TQfgsQdc/view).  

```misc
pretrain.pdparams: --model resnet --model_depth 50 --n_pretrain_classes 1039
```
## Preparation

### UCF-101

* Download videos and train/test splits [here](http://crcv.ucf.edu/data/UCF101.php).
* Convert from avi to jpg files using ```util_scripts/generate_video_jpgs.py```

```bash
python -m util_scripts.generate_video_jpgs avi_video_dir_path jpg_video_dir_path ucf101
```

* Generate annotation file in json format similar to ActivityNet using ```util_scripts/ucf101_json.py```
  * ```annotation_dir_path``` includes classInd.txt, trainlist0{1, 2, 3}.txt, testlist0{1, 2, 3}.txt

```bash
python -m util_scripts.ucf101_json annotation_dir_path jpg_video_dir_path dst_json_path
```

## Running the code

Assume the structure of data directories is the following:

```misc
~/
  data/
    UCF-jpg/
      .../ (directories of class names)
        .../ (directories of video names)
          ... (jpg files)
    UCF_annotation/
      ucf101_01.json
results/
   val.json
```

Confirm all options.

```bash
python main.py --root_path ~/ --video_path data/UCF-jpg --annotation_path data/UCF_json/ucf101_01.json \
--result_path results --dataset ucf101 --model resnet --n_pretrain_classes 1039 \
--pretrain_path data/pretrain --model_depth 50 --n_classes 101 --batch_size 128 \
--checkpoint 5 --n_epochs 20 --learning_rate 0.003 --train_crop 'random' --lr_scheduler multistep\
--inference --inference_batch_size 1
```

Evaluate top-1 video accuracy of a recognition result (~/results/val.json).

```bash
# 计算top1 accuracy
python -m util_scripts.eval_accuracy --ground_truth_path data/UCF_json/ucf101_01.json \
--result_path results/val_random.json --subset validation --k 1 --ignore --save
```
