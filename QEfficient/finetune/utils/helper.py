# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

TASK_TYPE = ["generation", "seq_classification"]
PEFT_METHOD = ["lora"]
DEVICE = ["qaic", "cpu", "cuda"]
BATCHING_STRATEGY = ["padding", "packing"]


def parse_unk_args(unk_args_str):
    if len(unk_args_str) % 2 != 0:
        raise RuntimeError("Unknown arguments must be in pairs")
    unk_args_dict = {unk_args_str[i].replace("--", ""): unk_args_str[i + 1] for i in range(0, len(unk_args_str), 2)}
    return unk_args_dict
