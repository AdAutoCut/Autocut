#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import pickle
import random
import threading
import numpy as np
import time
import traceback
from google.protobuf.json_format import MessageToDict
from infer_server_sdk.blobstore import *
from kess.framework import GrpcClient, ClientOption
from mars.protos.model_serving_pb2 import PredictRequest, MetaInfo
from mars.protos.model_serving_pb2_grpc import ModelServingStub
from mars.service.utils import TensorDictToNumpyArrayData

logger = logging.getLogger(__name__)
fmt_str = ('%(asctime)s.%(msecs)03d %(levelname)7s '
           '[%(thread)d][%(process)d] %(message)s')
fmt = logging.Formatter(fmt_str, datefmt='%H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def predict_audio(grpc_client, timeout):
    try:
        start = time.perf_counter()
        req = PredictRequest(id="132978328786", meta=MetaInfo(str_val=json.dumps({"audio_list":[["ad_nieuwland-material_BGM_retrieval/music_foder/160936258826.wav", 0, 0]], "no_vocal":True})))  # input: video_id
        resp = grpc_client.Predict(req, timeout=timeout)  # output: [music_path, music_start_time, music_end_time]
        print(f"resp:{MessageToDict(resp)}")
        return
        # music_id = resp.medias[0].meta.str_array.str_elems[0]
        # music_path = resp.medias[0].meta.str_array.str_elems[1]
        # music_start_time = resp.medias[0].meta.str_array.str_elems[2]
        # music_end_time = resp.medias[0].meta.str_array.str_elems[3]
        # print(f"music_id:{music_id}, music_path:{music_path}, music_start_time:{music_start_time}, music_end_time:{music_end_time}")
        # print(f"current cost time:{time.perf_counter() - start}")
    except Exception as e:
        traceback.print_exc()
        logger.error('error:{}'.format(e))

def predict_video(grpc_client, timeout):
    try:
        start = time.perf_counter()
        req = PredictRequest(id="132978328786", meta=MetaInfo(str_val=json.dumps({"video_list":[["video_def_160936258826.mp4", 0, 0]], "no_vocal":True})))  # input: video_id
        resp = grpc_client.Predict(req, timeout=timeout)  # output: [music_path, music_start_time, music_end_time]
        print(f"resp:{MessageToDict(resp)}")
        return
        # music_id = resp.medias[0].meta.str_array.str_elems[0]
        # music_path = resp.medias[0].meta.str_array.str_elems[1]
        # music_start_time = resp.medias[0].meta.str_array.str_elems[2]
        # music_end_time = resp.medias[0].meta.str_array.str_elems[3]
        # print(f"music_id:{music_id}, music_path:{music_path}, music_start_time:{music_start_time}, music_end_time:{music_end_time}")
        # print(f"current cost time:{time.perf_counter() - start}")
    except Exception as e:
        traceback.print_exc()
        logger.error('error:{}'.format(e))


if __name__ == "__main__":
    # channel = grpc.insecure_channel('10.106.25.215:21164')
    # client = ModelServingStub(channel=channel)
    client_option = ClientOption(
        biz_def='ad',
        grpc_service_name='ad-creative-mars-BGM-retrieval-get-feature-online',  #
        grpc_stub_class=ModelServingStub
    )
    client = GrpcClient(client_option)  # ???

    predict_audio(client, 2000)

    predict_video(client, 2000)