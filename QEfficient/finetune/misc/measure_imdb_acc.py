# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import warnings

import torch
import datasets
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from QEfficient.finetune.configs.training import train_config as TRAIN_CONFIG
from QEfficient.finetune.utils.config_utils import generate_dataset_config
from QEfficient.finetune.utils.dataset_utils import get_preprocessed_dataset

# Suppress all warnings
warnings.filterwarnings("ignore")

#try:
#    import torch_qaic  # noqa: F401
#
#    device = "qaic:32"
#except ImportError as e:
# print(f"Warning: {e}. Moving ahead without these qaic modules.")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_config = TRAIN_CONFIG()
train_config.model_name = "google-bert/bert-base-uncased"
train_config.tokenizer_name = "google-bert/bert-base-uncased"
train_config.task_type = "seq_classification"
train_config.dataset = "imdb_dataset"
train_config.output_dir = "./bert_imdb_qaic_ep_3_lr_1e_3_grad_acc_16_sdk_1.20.0.77_1_soc_w_sorting/complete_epoch_3"

dataset_config = generate_dataset_config(train_config, kwargs={})

# Load the tokenizer and add special tokens
tokenizer = AutoTokenizer.from_pretrained(
    train_config.model_name if train_config.tokenizer_name is None else train_config.tokenizer_name
)
if not tokenizer.pad_token_id:
    tokenizer.pad_token_id = tokenizer.eos_token_id

dataset = datasets.load_dataset("stanfordnlp/imdb", split="test", trust_remote_code=True)


# trained_weights_path = os.path.join(train_config.output_dir, "trained_weights")
# list_paths = [d for d in os.listdir(trained_weights_path) if os.path.isdir(os.path.join(trained_weights_path, d))]
# max_index = max([int(path[5:]) for path in list_paths])

# save_dir = os.path.join(trained_weights_path, "step_" + str(max_index))

# model_ft = AutoModelForSequenceClassification.from_pretrained(save_dir, attn_implementation="sdpa")
model_ft = AutoModelForSequenceClassification.from_pretrained(train_config.output_dir, attn_implementation="sdpa")
model_ft.to(device)
model_ft.eval()

print("Original model output:")
with torch.inference_mode():  
    correct = 0  
    for input in tqdm(dataset):
        text = input["text"]
        model_input = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        output = model_ft(**model_input)
        pred_id = torch.argmax(output.logits, dim=1).detach().to("cpu")
        if input['label'] == pred_id:
            correct += 1
            
    acc = correct / len(dataset)
    print(f"Test accuracy: {acc:.4f}")
    
