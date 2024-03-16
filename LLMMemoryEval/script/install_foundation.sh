#!/bin/bash

apt update
apt install -y vim git wget screen

pip install transformers sentencepiece torch torchvision accelerate bitsandbytes xformers deepspeed