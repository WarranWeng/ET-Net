## Dataset: 
HQF, MVSEC and IJRR datasets can be produced via the instructions in this [repo](https://github.com/TimoStoff/events_contrast_maximization). Note that MVSEC and IJRR are cut for better evaluation, of which the exact cut time can be found in the supplementary material.

## Pretrained model
The pretrained model, which can reproduce the quantitative results in the paper, will be released in this [site](https://drive.google.com/file/d/1V7vj3YkbhAmgzyf6rqrkeF0HSSNwyGkO/view?usp=sharing)

## Inference:
To reconstruct the intensity image using our ET-Net, E2VID, E2VID+, FireNet, FireNet+, one can run the file `scripts/eval.py`. The paths to the pretrained model, dataset and output files should be specified. We provide example in `scripts/eval.py`, one can look into this script for details. 

