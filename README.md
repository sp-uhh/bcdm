# Bone-conduction Guided Multimodal Speech Enhancement with Conditional Diffusion Models

This repository contains the implementation for:
- Sina Khanagha, Bunlong Lay, Timo Gerkmann, [*"Bone-conduction Guided Multimodal Speech Enhancement with Conditional Diffusion Models"*](https://www.arxiv.org/abs/2601.12354), IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), Barcelona, Spain, May 2026.

The code base is mostly adopted from our group's previous work [*SGMSE+*](https://github.com/sp-uhh/sgmse)



## Installation

You can clone the repository and install the required dependencies with:
```bash
git clone https://github.com/sp-uhh/bcdm.git
cd bcdm
pip install -r requirements.txt
```




## Training

You can use the `train.py` script for training the model. For example, to train the BCDM-IC-L model from our paper you can use the following command: 

```bash
python train.py --base_dir <path_to_your_dir> --format conditional_bc --backbone ncsnpp_v2 
```
And for BCDM-DC-L:

```bash
python train.py --base_dir <path_to_your_dir> --format conditional_bc --backbone ncsnpp_v2_decoder_injection 
```
**Note that the some of the available options from `python train.py --help` are not currently implemented here.**

- For resuming training, you can use the `--ckpt` option of `train.py`
- If you do not have a wandb account setup, you can also pass --nolog for offline logging
- where `your_base_dir` should be a path to a folder containing subdirectories `train/` and `valid/` (optionally `test/` as well). Each subdirectory must itself have three subdirectories `clean/`, `noisy/` and `acc/` (containing bone-conducted sensor data), with the same filenames present in all three subdirectories. Alternatively you can modify the sgmse/data_module.py file to match your dataset structure.





## Evaluation

To evaluate on a test set, run
```bash
python enhancement.py --test_dir <your_test_dir> --conditional_dir <your_bone-conducted_dir> --enhanced_dir <enhanced_files_output_dir> --ckpt <path_to_model_checkpoint> --N <num_reverse_steps>
```

to generate the enhanced .wav files, and subsequently run

```bash
python calc_metrics.py --test_dir <your_test_dir> --enhanced_dir <your_enhanced_dir>
```

to calculate and output the instrumental metrics.


## Citations / References

We kindly ask you to cite our papers in your publication when using any of our research or code:
TODO