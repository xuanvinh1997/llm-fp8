#!/bin/bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

pip install --no-build-isolation transformers accelerate datasets flash-attn==2.7.3 transformer_engine[pytorch]

pip install -r requirements.txt