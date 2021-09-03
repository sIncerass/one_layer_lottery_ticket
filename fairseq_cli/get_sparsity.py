#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast
import logging
import math
import os
import sys
from argparse import Namespace
from itertools import chain

import numpy as np
import torch
from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.data import encoders
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from omegaconf import DictConfig

from fairseq.modules.masked_modules import ThresholdBinarizer

def change_model_mode(model, mode="training"):
    for name, module in model.named_modules():
        try:
            module.change_mode(mode)
        except:
            pass
    print(f'Successfule change to model to {mode} mode.')

def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.txt".format(cfg.dataset.gen_subset),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    else:
        return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    task = tasks.setup_task(cfg.task)


    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )
    # print(models)
    # NOTE: change the mode 
    for _model in models:
        change_model_mode(_model, mode="eval")

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    model_mask = models[0] # we only need one model to compute the sparsity
    total_num_params = 0
    total_masked_params = 0

    for name, param in model_mask.named_parameters():
        if "layer_norm" in name:
            print(name)
            print(param.mean(), param.var().sqrt())
    

    # for name, param in model_mask.named_parameters():
    #     if "scores" in name:
    #         total_num_params += param.numel()
    #         mask_score = ThresholdBinarizer.apply(param, model_mask.encoder.embed_tokens.prune_ratio, True)
    #         total_masked_params += mask_score.sum()
    #         print(f"Sparisity of {name}: ", 1 - mask_score.sum() / param.numel())
    
    # print('Sapristy level is: ', 1 - total_masked_params / total_num_params)


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
