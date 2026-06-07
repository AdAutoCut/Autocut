from google.protobuf.json_format import MessageToDict
from kess.framework import ClientOption, KessOption, GrpcClient
from mmu.zt_speech_recognition_pb2 import MmuPhotoSpeechQueryRequest
from mmu.zt_speech_recognition_pb2_grpc import MmuZtSpeechRecognitionServiceStub

from video.client.base_client import BaseClient


class PhotoAsrMmuClient(BaseClient):
    def __init__(self, servicer_name="video-graph-server", client_service_name="grpc_mmuZtSpeechRecognition"):
        super().__init__(servicer_name)
        client_option = ClientOption(
            biz_def='mmu',
            grpc_service_name=client_service_name,
            grpc_stub_class=MmuZtSpeechRecognitionServiceStub,
            servicer_option=KessOption(biz_def="ad", name=servicer_name, port=20101)
        )
        self.client = GrpcClient(client_option)

    def _sync_run(self, *args, **kwargs):
        future = self._async_run(*args, **kwargs)
        return self._async_wait(future)

    def _async_run(self, photo_id, biz_def="AD-AIGC"):
        req = MmuPhotoSpeechQueryRequest(
            biz=biz_def, photo_id=[photo_id]
        )
        return self.client.QueryPhotoSpeechResult.future(req, timeout=self.timeout_1)

    def _async_wait(self, future):
        ret = None
        if future:
            resp = future.result()
            ret = MessageToDict(resp)
        return ret
