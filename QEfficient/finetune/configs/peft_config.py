# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class BaseConfig:
    """Base class for PEFT finetuning config"""

    def to_dict(self) -> Dict:
        """Convert the members of the dataclass to a dict.

        Returns:
            Dict: Dict representing the members of dataclass.
        """
        return asdict(self)


@dataclass
class LoraConfig:
    """LoRA-specific configuration for parameter-efficient fine-tuning.

    Attributes:
        lora_r (int): LoRA rank (default: 8).
        lora_alpha (int): LoRA scaling factor (default: 32).
        target_modules (List[str]): Modules to apply LoRA to (default: ["q_proj", "v_proj"]).
        bias (str): Bias handling in LoRA (default: "none").
        task_type (str): Task type for LoRA (default: "CAUSAL_LM").
        lora_dropout (float): Dropout rate for LoRA (default: 0.0).
        inference_mode (bool): Whether model is in inference mode (default: False).
    """

    lora_r: int = 8
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05
    inference_mode: bool = False  # should be False for finetuning

    def to_dict(self):
        """Converts the dataclass members to dict."""
        data = asdict(self)
        # PEFT module takes r value directly instead of lora_r.
        # Using --r conflicts with torchrun options. Hence, updated to --lora_r.
        data["r"] = data.pop("lora_r")
        return data


# CAUTION prefix tuning is currently not supported
@dataclass
class PrefixConfig:
    num_virtual_tokens: int = 30
    task_type: str = "CAUSAL_LM"
