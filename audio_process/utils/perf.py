from infra.perflog import create_perf_context


class PerfContext:
    def __init__(self, extra5, extra6):
        self.extra5 = extra5
        self.extra6 = extra6

    def __call__(self, subtag, micros=0, count=1, extra1="", extra2="", extra3="", extra4=""):
        add_perf(subtag, micros, count, extra1, extra2, extra3, extra4, self.extra5, self.extra6)


def add_perf(subtag, micros=0, count=1, extra1="", extra2="", extra3="", extra4="", extra5="", extra6=""):
    if subtag and micros > 0:
        log_ctx = create_perf_context("ad.aigc.video_graph", subtag,
                                      extra1, extra2, extra3, extra4, biz_def='ad',
                                      extra5=extra5, extra6=extra6
                                      )
        log_ctx.logstash_only(micros=micros)
        # 立刻保存上报数据
        # log_ctx.persist_data()
    elif subtag:
        log_ctx = create_perf_context("ad.aigc.video_graph", subtag,
                                      extra1, extra2, extra3, extra4, biz_def='ad',
                                      extra5=extra5, extra6=extra6
                                      )
        log_ctx.logstash_only(count=count)
        # 立刻保存上报数据
        # log_ctx.persist_data()
    else:
        pass

def add_llm_script_perf(subtag, micros=0, extra1="", extra2="", extra3="", extra4=""):
    if subtag and micros > 0:
        log_ctx = create_perf_context('ad.aigc.script_creation', subtag,
                                      extra1, extra2, extra3, extra4, biz_def='ad')
        log_ctx.logstash_only(micros=micros)
        # 立刻保存上报数据
        # log_ctx.persist_data()
    elif subtag:
        log_ctx = create_perf_context('ad.aigc.script_creation', subtag,
                                      extra1, extra2, extra3, extra4, biz_def='ad')
        log_ctx.logstash_only(count=1)
        # 立刻保存上报数据
        # log_ctx.persist_data()
    else:
        pass

def add_script_creation_perf(subtag, micros=0, extra1="", extra2="", extra3="", extra4=""):
    if subtag and micros > 0:
        log_ctx = create_perf_context('ad.aigc.script_creation', subtag,
                                      extra1, extra2, extra3, extra4, biz_def='ad')
        log_ctx.logstash_only(micros=micros)
        # 立刻保存上报数据
        # log_ctx.persist_data()
    elif subtag:
        log_ctx = create_perf_context('ad.aigc.script_creation', subtag,
                                      extra1, extra2, extra3, extra4, biz_def='ad')
        log_ctx.logstash_only(count=1)
        # 立刻保存上报数据
        # log_ctx.persist_data()
    else:
        pass
