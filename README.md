# Event-based Video Reconstruction Using Transformer

**Official implementation** of the following paper:

Event-based Video Reconstruction Using Transformer by Wenming Weng, Yueyi Zhang, Zhiwei Xiong. In ICCV 2021.

## Dataset: 

HQF, MVSEC and IJRR datasets can be produced via the instructions in this [repo](https://github.com/TimoStoff/events_contrast_maximization). Note that MVSEC and IJRR are cut for better evaluation, of which the exact cut time can be found in the supplementary material.

## Pretrained model

The pretrained model, which can reproduce the quantitative results in the paper, will be released in this [site](https://drive.google.com/file/d/1V7vj3YkbhAmgzyf6rqrkeF0HSSNwyGkO/view?usp=sharing)

## Inference:

To reconstruct the intensity image using our ET-Net, E2VID, E2VID+, FireNet, FireNet+, one can run the file `scripts/eval.py`. The paths to the pretrained model, dataset and output files should be specified. We provide example in `scripts/eval.py`, one can look into this script for details. 

## Citation

If you find this work helpful, please consider citing our paper.

```latex
@InProceedings{Weng_2021_ICCV,
    author    = {Weng, Wenming and Zhang, Yueyi and Xiong, Zhiwei},
    title     = {Event-based Video Reconstruction Using Transformer},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2021},
}
```

## Related Projects

[Reducing the Sim-to-Real Gap for Event Cameras, ECCV'20](https://github.com/TimoStoff/event_cnn_minimal)

[High Speed and High Dynamic Range Video with an Event Camera, TPAMI'19](https://github.com/uzh-rpg/rpg_e2vid)

## Contact

If you have any problem about the released code, please do not hesitate to contact me with email (wmweng@mail.ustc.edu.cn).
