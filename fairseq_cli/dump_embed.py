#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable

import numpy as np
import torch

from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from omegaconf import DictConfig
from fairseq.trainer import Trainer


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")


def main(cfg: DictConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    metrics.reset()

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)
    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in cfg.dataset.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # Build model and criterion
    model = task.build_model(cfg.model)

    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(name)
    # change_model_mode(model, mode="eval")
    # for name, module in model.named_modules():
    #     try:
    #         print(module.training_or_inference)
    #     except:
    #         pass
    # print('*'*100)
    # print(cfg.common.no_progress_bar)
    # print(cfg)

    criterion = task.build_criterion(cfg.criterion)
    # logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {})".format(criterion.__class__.__name__))
    logger.info(
        "num. model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )

    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)

    checkpoint_path = os.path.join(
            cfg.checkpoint.save_dir, "checkpoint_best.pt"
        )
    
    state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_path)

    print(state.keys())
    trainer.get_model().load_state_dict(
                    state["model"], strict=True, model_cfg=cfg.model
                )


    def dump_embed_file(embed, embed_dict, embed_file):
        """Parse embedding text file into a dictionary of word and embedding tensors.

            The first line can have vocabulary size and dimension. The following lines
            should contain word and embedding separated by spaces.

            Example:
                2 5
                the -0.0230 -0.0264  0.0287  0.0171  0.1403
                at -0.0395 -0.1286  0.0275  0.0254 -0.0932
            """
        with open(embed_file, "w") as file:
            file.write( "{} {}\n".format(embed.size(0), embed.size(1)) )
            for symbol_emb, symbol in zip(embed.cpu().numpy().tolist(), task.src_dict.symbols):
                file.write("{}".format(symbol))
                for emb_ in symbol_emb:
                    file.write(" {}".format(float(emb_)))
                file.write("\n")

    # dump the encoder embedding
    encoder_embed_path = os.path.join(
            cfg.checkpoint.save_dir, "encoder_embed.txt"
        )
    decoder_embed_path = os.path.join(
            cfg.checkpoint.save_dir, "decoder_embed.txt"
        )
    print(model.encoder.embed_tokens, task.src_dict, len(task.src_dict.symbols))
    dump_embed_file( model.encoder.embed_tokens.weight.data, task.src_dict, encoder_embed_path )

    # dump the decoder embedding
    print(model.decoder.embed_tokens, task.tgt_dict, len(task.tgt_dict.symbols))
    dump_embed_file( model.decoder.embed_tokens.weight.data, task.tgt_dict, decoder_embed_path )

def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)


if __name__ == "__main__":
    cli_main()
