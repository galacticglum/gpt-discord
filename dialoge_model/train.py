"""Train the agent."""
import math
import logging
import argparse
from typing import Dict
from pathlib import Path
from pprint import pformat
from itertools import chain
from collections import defaultdict

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import (
    TensorboardLogger,
    OutputHandler,
    OptimizerParamsHandler
)

from transformers import (
    AdamW,
    OpenAIGPTTokenizer
    OpenAIGPTDoubleHeadsModel,
    GPT2Tokenizer,
    GPT2DoubleHeadsModel,
    WEIGHTS_NAME,
    CONFIG_NAME
)


class SpecialTokens:
    """Special tokens used for embedding additional information.

    Instance Attributes:
        - bos_token: The token used to mark the beginning of the sequence.
        - eos_token: The token used to mark the end of the sequence.
        - person_a_token: The token used to mark when the first person starts speaking.
        - person_b_token: The token used to mark when the second person start speaking.
        - pad_token: The token used to pad sequences.
    """
    bos_token: str = '<bos>'
    eos_token: str = '<eos>'
    person_a_token: str = '<p1>'
    person_b_token: str = '<p2>'
    pad_token: str = '<pad>'

    def to_attr_dict(self) -> Dict[str, str]:
        """Return a dict mapping the attribute name (in the tokenizer) of each token to its value."""
        return {
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'pad_token': self.pad_token,
            'additional_sepcial_tokens': [
                self.person_a_token, self.person_b_token
            ]
        }