## IORT-Interactive Object Removal Tool based on SAM and LaMa

![ObjectRemoval](./displays/ObjectRemoval.gif)

##### Install

```
# create environment
conda create -n iort python=3.9

# activate environment
conda activate iort

pip install torch==2.2 torchvision==0.17.0
cd LaMaProject
pip install -r requirements.txt
```

##### To Do

- [ ] LaMa refinement inpainting mode
- [ ] Video object removal
- [ ] SAM box mode
- [ ] shortcut
- [ ] Fix bugs

##### Acknowledgment

Thanks yatengLG for the awesome project: [An Interactive Semi-Automatic Annotation Tool Based on Segment Anything](https://github.com/yatengLG/ISAT_with_segment_anything)
