#!/bin/bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

pip install --no-build-isolation transformers accelerate datasets flash_attn transformer_engine[pytorch]

pip install -r requirements.txt