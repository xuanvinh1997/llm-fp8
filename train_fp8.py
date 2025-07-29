# Import necessary packages, methods and variables
from utils import *


hyperparams.model_name = "meta-llama/Llama-3.2-1B"
hyperparams.weights_cache_dir = "/root/.cache/huggingface/hub"
hyperparams.mixed_precision = "fp8"
hyperparams.dataset_name = "nvidia/OpenMathInstruct-2"

# Init the model and accelerator wrapper
model = init_te_llama_model(hyperparams)
print(model)
accelerator, model, optimizer, train_dataloader, lr_scheduler = wrap_with_accelerator(model, hyperparams)

# Finetune the model
finetune_model(model, hyperparams, accelerator, train_dataloader, optimizer, lr_scheduler)