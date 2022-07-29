#!/bin/bash

conda activate ~/anaconda3/envs/dasp_oop_GAN
python -m main --ganType=dcgan '{"gen_optimizer": "adam", "dis_optimizer": "rmsprop", "adam_lr": 0.0002, "adam_beta_1": 0.5}'