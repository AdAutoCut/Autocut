import threading
import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from infra import kws

from .singleton_meta import SingletonMeta
from .logger import logger

s3_client_config = Config(s3={"addressing_style": "path"}, signature_version=UNSIGNED)  # 固定配置，不可更改

s3_endpoint_url = "http://bs3-hb1.internal"                       # 国内线上环境
s3_endpoint_url_sgp = "http://bs3-sgp.internal"                   # 海外线上环境
#s3_endpoint_url = "http://bs3-hb1.staging.kuaishou.com"          # Staging 环境

oversea_bucket_list = ['ad-i18n-dsp', 's3-photo-oversea']


class BlobStoreClientManager(metaclass=SingletonMeta):
    def __init__(self):
        self.bucket_client_map = {}

    def get_client(self, bucket):
        client = self.bucket_client_map.get(bucket, None)
        if client:
            return client
        else:
            limit = 10
            while not client and limit > 0:
                try:
                    limit -= 1
                    client = BlobStoreClient(bucket)
                except:
                    client = None
            self.bucket_client_map.update({
                bucket: client
            })
            return client


class BlobStoreClient(object):
    def __init__(self, bucket_name):
        for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
            os.environ.pop(var, None)
        ksn = kws.get_kws_info()
        endpoint_url = s3_endpoint_url
        if (ksn and ksn.kws_service_region == 'SGP') or bucket_name in oversea_bucket_list:
            endpoint_url = s3_endpoint_url_sgp
        self.s3_bucket = bucket_name
        self.s3_client = boto3.client(service_name="s3",
                                      endpoint_url=endpoint_url,
                                      config=s3_client_config,
                                      use_ssl=False)
        event_system = self.s3_client.meta.events
        event_system.register('before-sign.*.*', self._add_service_header)

    @staticmethod
    def _add_service_header(request, **kwargs):
        ksn = kws.get_kws_info()  # 获取服务名标识
        service_name = 'unknown' if ksn is None else ksn.kws_service_name
        request.headers.add_header('service', service_name)

    def set_bucket_name(self, bucket_name):
        self.s3_bucket = bucket_name

    def upload_file_with_retry(self, file_path, key, retry=5):
        while retry:
            ret = self.upload_binary_to_s3(file_path, key)
            if ret:
                return True
            retry = retry - 1
            logger.debug(f"upload_file_with_retry retry:{retry}")
        return False

    # see: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.put_object
    def upload_binary_to_s3(self, file_path, key, metadata=None):
        if metadata is None:
            metadata = dict()
        try:
            with open(file_path, 'rb') as binfile:
                self.s3_client.put_object(Bucket=self.s3_bucket, Body=binfile, Key=key, Metadata=metadata)
                logger.debug(f'upload_succ:{key}')
        except Exception as e:
            logger.error(e)
            return False
        return True

    def download_file_with_retry(self, key, dest_key, retry=5):
        while retry:
            ret = self.download_binary_from_s3(key, dest_key)
            if ret:
                return True
            retry = retry - 1
            logger.debug(f"download_file_with_retry retry:{retry}")
        return False

    # see: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.get_object
    def download_binary_from_s3(self, key, dest_key, bucket=""):
        try:
            real_bucket = bucket if len(bucket) > 0 else self.s3_bucket
            response = self.s3_client.get_object(Bucket=real_bucket, Key=key)
            if response['ResponseMetadata']['HTTPStatusCode'] != 200:
                logger.error(f"error response:{response}")
                return False
            with open(dest_key, 'wb') as binfile:
                binfile.write(response["Body"].read())
            logger.debug(f'download and write succ: {key}, {dest_key}')
        except Exception as e:
            logger.error(e)
            return False
        return True

    def upload_bytes_to_s3(self, string, key, metadata=None):
        if metadata is None:
            metadata = dict()
        try:
            self.s3_client.put_object(Bucket=self.s3_bucket, Body=string, Key=key, Metadata=metadata)
            logger.debug(f'upload_succ: {key}')
        except Exception as e:
            logger.error(f'error:{e}')
            return False
        return True

    def download_bytes_from_s3(self, key, bucket=""):
        byte_val = None
        try:
            real_bucket = bucket if len(bucket) > 0 else self.s3_bucket
            response = self.s3_client.get_object(Bucket=real_bucket, Key=key)
            if response['ResponseMetadata']['HTTPStatusCode'] != 200:
                logger.error(f"error response:{response}")
                return False, byte_val
            byte_val = response['Body'].read()
            logger.debug(f'download_succ: {key}')
        except Exception as e:
            logger.error(f'error:{e}')
            return False, byte_val
        return True, byte_val

    def batch_download_binary_from_s3(self, item_map):
        for k, v in item_map.items():
            self.download_binary_from_s3(k, v[0], v[1])

    def concurrency_download(self, item_map, batch_size=1):
        count = 0
        batch_list = []
        thread_list = []
        for k, v in item_map.items():
            if count % batch_size == 0:
                batch_list.append({})
            batch_list[-1].update({k: v})
            count += 1

        for batch in batch_list:
            thread_list.append(threading.Thread(target=self.batch_download_binary_from_s3, args=(batch, )))

        for t in thread_list:
            t.start()

        for t in thread_list:
            t.join()

    def batch_download_bytes_from_s3(self, item_map, res_map):
        for k, v in item_map.items():
            status, res = self.download_bytes_from_s3(k, v)
            if status:
                res_map.update({k: res})

    def concurrency_download_bytes(self, item_map, res_map, batch_size=1):
        count = 0
        batch_list = []
        thread_list = []
        for k, v in item_map.items():
            if count % batch_size == 0:
                batch_list.append({})
            batch_list[-1].update({k: v})
            count += 1

        for batch in batch_list:
            thread_list.append(threading.Thread(target=self.batch_download_bytes_from_s3, args=(batch, res_map)))

        for t in thread_list:
            t.start()

        for t in thread_list:
            t.join()

    def object_exists(self, key):
        try:
            # response = self.s3_client.object_exists(Bucket=self.s3_bucket, Key=key)
            response = self.s3_client.head_object(Bucket=self.s3_bucket, Key=key)
            logger.debug(f'response:{response}')
            return True
        except Exception as e:
            logger.debug(e)
            return False

    def get_head_object(self, key):
        try:
            response = self.s3_client.head_object(Bucket=self.s3_bucket, Key=key)
            logger.debug(f'response:{response}')
            return True, response
        except Exception as e:
            logger.error(e)
            return False, None

if __name__ == '__main__':
    photo_id = '167107388360'
    key = f"{photo_id}.mp4"
    client = BlobStoreClient('video-def')
    client.download_file_with_retry(key, 'test_video.mp4')
