# Averting Mode Collapse for Generative Zero-Shot Learning
Averting Mode Collapse for Generative Zero-Shot Learning 

## Data
The code uses the ResNet101 features provided by the paper: Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly, and follows its GZSL settings.

The features can be download here [data](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip)
## Video
[Link](https://iccke.um.ac.ir/Areas/Panel/Hamayesh/2/88f50f46-c077-4da1-a82c-9607928917e0/Asar/6321/iccke-1139___e2DQf.mp4)
## Presentation
[ICCKE.pdf](https://github.com/ShayanRamazi/Averting-Mode-Collapse-for-Generative-Zero-Shot-Learning/files/10094151/ICCKE.pdf)



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
