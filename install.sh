#!/bin/bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

pip install --no-build-isolation transformers accelerate datasets "flash_attn>= 2.1.1, <= 2.8.1" transformer_engine[pytorch]

pip install -r requirements.txt