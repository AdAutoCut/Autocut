import logging
import sys

from loguru import logger as loguru_logger

logger = logging.getLogger("ad_video_graph")


class InterceptHandler(logging.Handler):
    def emit(self, record):
        logger_opt = loguru_logger.opt(depth=6, exception=record.exc_info)
        logger_opt.log(record.levelname, record.getMessage())


def setup_logger(log_name: str, log_level: str = "INFO"):
    loguru_logger.remove()  # 移除默认的handler
    fmt_str = ('<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | '
               '<level>{level: <8}</level> | '
               '<cyan>[{thread.id}][{process.id}]</cyan> '
               '[{file.path}:{line}:{function}] {message}')
    loguru_logger.add(f'{log_name}.log', level=log_level, format=fmt_str, rotation="1 hour", retention="7 days", enqueue=True)
    loguru_logger.add(sys.stdout, level=log_level, format=fmt_str)

    loguru_handler = InterceptHandler()
    logger.addHandler(loguru_handler)
    logger.setLevel(log_level)

    infra_logger = logging.getLogger('infra')
    infra_logger.addHandler(loguru_handler)
    infra_logger.setLevel(logging.WARNING)

    kafka_logger = logging.getLogger('kafka')
    kafka_logger.addHandler(loguru_handler)
    kafka_logger.setLevel(logging.WARNING)

    # infra.perflog.enable_local()
