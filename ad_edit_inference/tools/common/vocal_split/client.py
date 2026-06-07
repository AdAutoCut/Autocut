# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# Author     ：Bo Wang
# File       : aigc_v2_client.py
# Time       ：2024/3/12 17:37
"""
import os
import random
import sys
import time
import traceback

import argparse
from kess.framework import ClientOption, GrpcClient
from google.protobuf.json_format import MessageToDict
from .Vocal_split_pb2_grpc import VocalSplitServingStub


# cdn_domain = "http://s1-11661.kwimgs.com"

class VocalSplitClient:
    def __init__(self, grpc_service_name='ad-Vocal-Split'):
        self.client_option = ClientOption(
            biz_def='ad',
            grpc_service_name=grpc_service_name,
            grpc_stub_class=VocalSplitServingStub,
        )
        self.client = GrpcClient(self.client_option)

    def sync_run(self, req, timeout=1200):
        try:
            resp = self.client.GetSplitVocal(req, timeout=timeout)
            res = MessageToDict(resp)
            # print(res)
            return res
        except:
            print(traceback.format_exc())
            return None
