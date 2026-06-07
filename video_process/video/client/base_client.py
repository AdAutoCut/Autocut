import time
import traceback
from abc import abstractmethod

from infra.kafka import KsKafkaConsumer, MessageContext, ConsumerParameter
from typing_extensions import final

from ..utils.logger import logger
from ..utils.perf import add_perf

BIG_RESPONSE_SERVICE_CLIENT = ["TTSMiniMaxClient"]


class CommonResponseConsumer(KsKafkaConsumer):
    def __init__(self, params, func):
        super().__init__(params)
        self.response_func = func

    def consume(self, message: bytes, context: MessageContext):
        self.response_func(message, context)


class BaseClient(object):
    def __init__(self, servicer_name="video-graph-server"):
        self.client = None
        self.class_name = self.__class__.__name__
        self.servicer_name = servicer_name
        self.resp_consumer = None
        self.timeout_1 = 10
        self.timeout_2 = 10
        self.start_consumer = False

    def set_timeout(self, timeout_1, timeout_2):
        self.timeout_1 = timeout_1
        self.timeout_2 = timeout_2

    def _start_consumer(self, topic, group_id):
        if self.start_consumer:
            return
        parameter = ConsumerParameter(topic, group_id, auto_offset_reset_to_latest=True)
        self.resp_consumer = CommonResponseConsumer(parameter, self._response_func)
        self.resp_consumer.start(block=False)
        self.start_consumer = True

    def _response_func(self, message: bytes, context: MessageContext):
        pass

    @abstractmethod
    def _sync_run(self, *args, **kwargs):
        pass

    @final
    def sync_req(self, *args, **kwargs):
        start_time = int(time.time() * 1000)
        add_perf(self.class_name, extra1="request", extra2="sync", extra3=self.servicer_name)
        resp = None
        try:
            resp = self._sync_run(*args, **kwargs)
            if resp is None:
                add_perf(self.class_name, extra1="res_is_none", extra2="sync", extra3=self.servicer_name)
        except:
            logger.info(f"{self.class_name} req failed, err:{traceback.format_exc()}")
            add_perf(self.class_name, extra1="req_failed", extra2="sync", extra3=self.servicer_name)
        cost_time = int(time.time() * 1000 - start_time)
        logger.debug(f"{self.class_name} cost time:{cost_time}")
        if self.class_name not in BIG_RESPONSE_SERVICE_CLIENT:
            logger.debug(f"{self.class_name} resp:{resp}")
        add_perf("rpc_cost_time_stat", micros=cost_time, extra1=self.class_name, extra2=self.servicer_name)
        return resp

    @abstractmethod
    def _async_run(self, *args, **kwargs):
        pass

    @abstractmethod
    def _async_wait(self, *args, **kwargs):
        pass

    @final
    def async_req(self, *args, **kwargs):
        add_perf(self.class_name, extra1="request", extra2="async", extra3=self.servicer_name)
        future = None
        try:
            future = self._async_run(*args, **kwargs)
            if future is None:
                add_perf(self.class_name, extra1="future_is_none", extra2="async", extra3=self.servicer_name)
        except:
            logger.info(f'{self.class_name} req failed, err: {traceback.format_exc()}')
            add_perf(self.class_name, extra1="req_failed", extra2="async", extra3=self.servicer_name)
        return future

    @final
    def async_req_wait(self, *args, **kwargs):
        start_time = int(time.time() * 1000)
        resp = None
        try:
            resp = self._async_wait(*args, **kwargs)
            if resp is None:
                add_perf(self.class_name, extra1="res_is_none", extra2="async", extra3=self.servicer_name)
        except:
            logger.info(f"{self.class_name} get res error, err:{traceback.format_exc()}")
            add_perf(self.class_name, extra1="get_res_error", extra2="async", extra3=self.servicer_name)
        cost_time = int(time.time() * 1000 - start_time)
        logger.debug(f"{self.class_name} cost time:{cost_time}")
        if self.class_name not in BIG_RESPONSE_SERVICE_CLIENT:
            logger.debug(f"{self.class_name} resp:{resp}")
        add_perf("rpc_cost_time_stat", micros=cost_time, extra1=self.class_name, extra2=self.servicer_name)
        return resp
