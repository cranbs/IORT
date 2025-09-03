## IORT-Interactive Object Removal Tool based on SAM and LaMa

## Removal Anything

***

## Fill Anything

***

![ObjectRemoval](./displays/ObjectRemoval.gif)

## Install

***

```
# create environment
conda create -n iort python=3.9

# activate environment
conda activate iort

pip install torch==2.2 torchvision==0.17.0
pip install -r LaMaProject/requirements.txt
```

## Usage

***

Download the model checkpoints provided in [LaMa](https://github.com/advimman/lama) (big-lama) and put them into `./checkpoints`.  For simplicity, you can also go [here](https://drive.google.com/drive/folders/1B2x7eQDgecTL0oh3LSIBDGj0fTxs6Ips) directly download `big-lama` and put them into `./checkpoints`.

```
python main.py
```

## To Do

***

- [ ] add visible botton function
- [ ] add revise mask function
- [ ] output files weidget
- [ ] test other segment anything model
- [ ] LaMa refinement inpainting mode
- [ ] Video object removal
- [ ] SAM box mode
- [ ] shortcut
- [ ] Fix bugs

## Acknowledgment

***

- Thanks [yatengLG](https://github.com/yatengLG) for his awesome project: [An Interactive Semi-Automatic Annotation Tool Based on Segment Anything](https://github.com/yatengLG/ISAT_with_segment_anything)

- Awesome image inpainting method [LaMa](https://github.com/advimman/lama) 