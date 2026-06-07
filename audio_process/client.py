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
from Vocal_split_pb2_grpc import VocalSplitServingStub
from Vocal_split_pb2 import VocalSplitRequest

cdn_domain = "http://s1-11661.kwimgs.com"

class Client:
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

def batch_demo_8(grpc_service_name):
    # 小说音频复刻
    req = VocalSplitRequest(vocal_blob_key={"db": "ad", "table": "nieuwland-material", "key": "127351952270.wav"})
    client = Client(grpc_service_name)
    resp = client.sync_run(req)
    print(resp)


def parse_args():
    parser = argparse.ArgumentParser(description='A servers for video generation')
    parser.add_argument('--service_name', '-s', default='ad-Vocal-Split', type=str, help="service name")
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    # batch_demo_2(grpc_service_name=args.service_name)
    print(args.service_name)
    batch_demo_8(grpc_service_name=args.service_name)
