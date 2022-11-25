# Averting Mode Collapse for Generative Zero-Shot Learning
Averting Mode Collapse for Generative Zero-Shot Learning [paper](https://ieeexplore.ieee.org/document/9420574)

## Data
The code uses the ResNet101 features provided by the paper: Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly, and follows its GZSL settings.

The features can be download here [data](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip)
## Video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/3PxsFIZsc0M/0.jpg)](https://www.youtube.com/watch?v=3PxsFIZsc0M)


## Run
    !python "clswgan.py" --dataroot "xlsa17/data"  --manualSeed 1365 --cls_weight 0.1 --preprocessing --val_every 1 --lr 0.00001 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 45 --syn_num 350 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset AWA2 --batch_size 64 --nz 85 --attSize 85 --resSize 2048 --outname awa --anti 0.2 --recons_weight 0.01

## Ciation

```
@inproceedings{Ramazi,
  title={Averting Mode Collapse for Generative Zero-Shot Learning},
  author={Ramazi, Shayan and Shabani, Setare},
  booktitle={12th International Conference on Computer and Knowledge Engineering (ICCKE)},
  year={2022}
}
```
