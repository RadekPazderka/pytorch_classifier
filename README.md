#Pytorch classifier

Simple pytorch classifier using custom modified VGG16 architecture.

###Install environment (conda)
1) conda env create --file environment.yml
2) conda activate pytorch_classifier

##Demo

1) Download pretrained model from uloz.to server [here](https://uloz.to/tam/ece5c4e7-8c83-47b9-95c5-f0fe50b26fb5).
2) Save it into **./checkpoints** folder.
3) Run demo:
```bash
    python demo.py
```

##Dataset
1) Download dataset **animals10** from [here](https://www.kaggle.com/alessiocorrado99/animals10/download).
2) Save dataset zip (archive.zip) into **dataset** folder.
3) Run preprocess dataset (splits dataset into TRAIN and VAL folders):

```bash
    python prepare_dataset.py
```

##Train + Val
```bash
    python train.py --train_dir [path/to/train/data/folder] \
                    --val_dir [path/to/val/data/folder] \
                    --checkpoint_dir [path/to/checkpoints/dir] \
                    --pretrained_model [path/to/pretrained/model.pkl]
```
**Example**
```bash
    python train.py --train_dir ./dataset/data/TRAIN \
                    --val_dir ./dataset/data/VAL \
                    --checkpoint_dir ./checkpoints \
                    --pretrained_model ./checkpoints/vgg16_0048.pkl
```

##Test single image
```bash
    python image_tests.py --image_path [path/to/img] \
                          --checkpoint_path [path/to/checkpoint.pkl]
```
**Example**
```bash
    python image_tests.py --image_path ./demo/butterfly.jpg \
                          --checkpoint_path ./checkpoints/vgg16_0048.pkl
```