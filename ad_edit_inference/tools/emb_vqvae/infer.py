# !/usr/bin/env python
# -*-coding:utf-8 -*-

import argparse
from loguru import logger
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from .modeling.vq_vae import RQVAE

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger.remove()
logger.add(sys.stderr, enqueue=True, level="INFO")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_config(config_path):
    conf = OmegaConf.load(config_path)
    return conf


def get_model(config_path):
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    config = get_config(config_path)
    model_path = os.path.join(
        current_dir,
        config.load.model_path,
        f'{config.load.epoch_id}.pth')
    assert os.path.exists(model_path), f"Model {model_path} does not exist"
    model = RQVAE(
        input_dim=config.model.input_feature_size,
        codebook_dim=config.model.codebook_dim,
        codebook_size=config.model.codebook_size,
        num_quantizers=config.model.num_head,
        encoder_mlp_size=config.model.encoder_mlp_size,
        decoder_mlp_size=config.model.decoder_mlp_size,
        shared_codebook=config.model.shared_codebook
    )
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    state_dicts = torch.load(model_path, map_location="cpu")
    lacked_key, unexpected_key = model.load_state_dict(
        state_dicts, strict=False)
    model = model.to(device)
    logger.info("lacked_key is ")
    logger.info(lacked_key)
    logger.info("unexpected_key is ")
    logger.info(unexpected_key)
    return model


def emb2token(model, embeddings):
    """embeddings (B, D) -> tokens (B, H)"""
    model.eval()
    model = model.to(device)
    embeddings = embeddings.to(device)
    with torch.no_grad():
        rec_emb, [indices, commit_loss] = model(embeddings)
    return indices


def token2emb(model, tokens):
    """tokens (B, H) -> embeddings (B, D)"""
    model.eval()
    model = model.to(device)
    tokens = tokens.to(device)
    with torch.no_grad():
        output = model.quantizer.get_output_from_indices(tokens)
        new_emb = model.decoder(output)
    return new_emb
