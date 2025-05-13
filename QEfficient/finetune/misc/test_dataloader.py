# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import shutil

import pytest
import torch
import random
import numpy as np 

import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
import QEfficient
import QEfficient.cloud.finetune
from QEfficient.cloud.finetune import TRAIN_CONFIG, get_preprocessed_dataset, get_dataloader_kwargs, get_custom_data_collator
from QEfficient.cloud.finetune import main as finetune
from QEfficient.finetune.utils.config_utils import generate_dataset_config

def clean_up(path):
    if os.path.exists(path):
        shutil.rmtree(path)


configs = [
    pytest.param("seq_classification", "imdb_dataset", "train", None, id="imdb"),
    pytest.param("seq_classification", "imdb_dataset", "test", 128, id="imdb"),
    pytest.param("generation", "samsum_dataset", "train", None, id="samsum"),
    pytest.param("generation", "samsum_dataset", "test", 128, id="samsum"),
    pytest.param("generation", "gsm8k_dataset", "train", None, id="gsm8k"),
    pytest.param("generation", "gsm8k_dataset", "test", 128, id="gsm8k"),
    pytest.param("generation", "grammar_dataset", "train", None, id="grammar"),
    pytest.param("generation", "grammar_dataset", "test", 128, id="grammar"),
    pytest.param("generation", "alpaca_dataset", "train", None, id="alpaca"),
    pytest.param("generation", "alpaca_dataset", "test", 128, id="alpaca"),
]


# TODO:enable this once docker is available
@pytest.mark.parametrize(
    "task_type,dataset_name,split_name,context_length",
    configs,
)
def test_finetune(
    task_type,
    dataset_name,
    split_name,
    context_length,
    mocker,
):
    # train_config_spy = mocker.spy(QEfficient.cloud.finetune, "TRAIN_CONFIG")
    generate_dataset_config_spy = mocker.spy(QEfficient.cloud.finetune, "generate_dataset_config")
    # generate_peft_config_spy = mocker.spy(QEfficient.cloud.finetune, "generate_peft_config")
    get_dataloader_kwargs_spy = mocker.spy(QEfficient.cloud.finetune, "get_dataloader_kwargs")
    # update_config_spy = mocker.spy(QEfficient.cloud.finetune, "update_config")
    # get_custom_data_collator_spy = mocker.spy(QEfficient.cloud.finetune, "get_custom_data_collator")
    get_preprocessed_dataset_spy = mocker.spy(QEfficient.cloud.finetune, "get_preprocessed_dataset")
    # get_longest_seq_length_spy = mocker.spy(QEfficient.cloud.finetune, "get_longest_seq_length")
    # print_model_size_spy = mocker.spy(QEfficient.cloud.finetune, "print_model_size")
    # train_spy = mocker.spy(QEfficient.cloud.finetune, "train")

    # kwargs = {
    #     "model_name": model_name,
    #     "max_eval_step": max_eval_step,
    #     "max_train_step": max_train_step,
    #     "intermediate_step_save": intermediate_step_save,
    #     "context_length": context_length,
    #     "run_validation": run_validation,
    #     "use_peft": use_peft,
    #     "device": device,
    # }

    if task_type == "seq_classification":
        tokenizer_name = "google-bert/bert-base-uncased"
        dataset_keys = {'input_ids', 'token_type_ids', 'attention_mask', 'labels'}
    elif task_type == "generation":
        tokenizer_name = "meta-llama/Llama-3.2-1B"
        dataset_keys = {'input_ids', 'token_type_ids', 'attention_mask', 'labels'}
    else:
        raise NotImplementedError

    # from QEfficient.cloud.finetune import TRAIN_CONFIG()


    # update the configuration for the training process
    # train_config = TRAIN_CONFIG()
    # update_config(train_config, **kwargs)
    train_config = mocker.Mock(spec=TRAIN_CONFIG)
    train_config.dataset = dataset_name
    dataset_config = generate_dataset_config(train_config, {})
    # Set the seeds for reproducibility
    # torch.manual_seed(train_config.seed)
    # random.seed(train_config.seed)
    # np.random.seed(train_config.seed)

    # Load the pre-trained model and setup its configuration
    # config = AutoConfig.from_pretrained(train_config.model_name)

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Get the dataset utils
    dataset_processer = tokenizer

    # Load and preprocess the dataset for training and validation
    dataset = get_preprocessed_dataset(dataset_processer, dataset_config, split=split_name, context_length=context_length)
    assert len(set(dataset.features.keys()).difference(dataset_keys)) == 0, "Dataset keys are not matching."
    
    train_config.context_length = context_length
    for enable_ddp in [True, False]:
        for enable_sorting_for_ddp in [True, False]:
            train_config.enable_ddp = enable_ddp
            train_config.enable_sorting_for_ddp = enable_sorting_for_ddp
    
            import pdb; pdb.set_trace()
            if enable_ddp:
                train_config.dist_backend = "qccl"
                import torch.distributed as dist
                dist.init_process_group(backend=train_config.dist_backend)
                # from here onward "qaic/cuda" will automatically map to "qaic:i/cuda:i", where i = process rank
                torch_device = torch.device("qaic")
                getattr(torch, torch_device.type).set_device(dist.get_rank())

            train_dl_kwargs = get_dataloader_kwargs(train_config, dataset, dataset_processer, split_name)
    
            print("length of dataset_train", len(dataset))
    
            custom_data_collator = get_custom_data_collator(dataset_processer, dataset_config)
            if custom_data_collator:
                print("custom_data_collator is used")
                train_dl_kwargs["collate_fn"] = custom_data_collator

            # Create DataLoaders for the training and validation dataset
            dataloader = torch.utils.data.DataLoader(
                dataset,
                num_workers=0,
                pin_memory=True,
                **train_dl_kwargs,
            )
            print(f"--> Num of Training Set Batches loaded = {len(dataloader)}")

            if enable_sorting_for_ddp:
                dist.barrier()
                dist.destroy_process_group()

            longest_seq_length, _ = get_longest_seq_length(dataloader.dataset)



    for data in dataset:
        print(data)


    generate_dataset_config_spy.assert_called_once()

    finetune(**kwargs)

    train_config_spy.assert_called_once()
    generate_dataset_config_spy.assert_called_once()
    generate_peft_config_spy.assert_called_once()
    update_config_spy.assert_called_once()
    get_custom_data_collator_spy.assert_called_once()
    get_longest_seq_length_spy.assert_called_once()
    print_model_size_spy.assert_called_once()
    train_spy.assert_called_once()

    assert get_dataloader_kwargs_spy.call_count == 2
    assert get_preprocessed_dataset_spy.call_count == 2

    args, kwargs = train_spy.call_args
    train_dataloader = args[1]
    eval_dataloader = args[2]
    optimizer = args[4]

    batch = next(iter(train_dataloader))
    assert "labels" in batch.keys()
    assert "input_ids" in batch.keys()
    assert "attention_mask" in batch.keys()

    assert isinstance(optimizer, optim.AdamW)

    assert isinstance(train_dataloader, DataLoader)
    if run_validation:
        assert isinstance(eval_dataloader, DataLoader)
    else:
        assert eval_dataloader is None

    args, kwargs = update_config_spy.call_args
    train_config = args[0]

    saved_file = os.path.join(train_config.output_dir, "adapter_model.safetensors")
    assert os.path.isfile(saved_file)

    clean_up(train_config.output_dir)
    clean_up("runs")
    clean_up(train_config.dump_root_dir)
