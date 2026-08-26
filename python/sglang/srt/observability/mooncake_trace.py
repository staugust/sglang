import time
from typing import Union

from sglang.srt.observability.req_time_stats import (
    RequestStageConfig,
    convert_time_to_realtime_ns,
)
from sglang.srt.observability.trace import TraceNullContext, TraceReqContext


class MooncakeRequestStage:
    MOONCAKE_SEND = RequestStageConfig(
        "mooncake_send",
        level=1,
    )
    MOONCAKE_RECV = RequestStageConfig(
        "mooncake_recv",
        level=1,
    )
    MOONCAKE_WORKER_SEND = RequestStageConfig(
        "mooncake_worker_send",
        level=1,
    )
    MOONCAKE_WORKER_SEND_SESSION = RequestStageConfig(
        "mooncake_worker_send_session",
        level=2,
    )
    MOONCAKE_WORKER_RECV = RequestStageConfig(
        "mooncake_worker_recv",
        level=1,
    )
    # HiCache (L3) remote KV recall stages.  Storage-hit probe (batch_exists)
    # and the actual remote fetch (MooncakeStore.batch_get_v1/v2) get their
    # own stage names so the schedules can be aggregated per remote fetch.
    # Both sit above the default trace_level (3), so they are opt-in:
    #   trace_level >= 4 -> emit the batch_get fetch slice
    #   trace_level >= 5 -> also emit the batch_exists hit-probe slice
    HICACHE_STORAGE_HIT_QUERY = RequestStageConfig(
        "hicache_storage_hit_query",
        level=5,
    )
    HICACHE_MOONCAKE_FETCH = RequestStageConfig(
        "hicache_mooncake_fetch",
        level=4,
    )


def mooncake_trace_slice(
    trace_ctx: Union[TraceReqContext, TraceNullContext],
    stage: RequestStageConfig,
    start_ts: float,
    thread_finish_flag=False,
):
    if trace_ctx is None:
        return

    if not trace_ctx.tracing_enable:
        return

    start_ts = convert_time_to_realtime_ns(start_ts)
    trace_ctx.trace_slice_start(stage.stage_name, stage.level, start_ts)
    trace_ctx.trace_slice_end(
        stage.stage_name,
        stage.level,
        thread_finish_flag=thread_finish_flag,
    )


def mooncake_trace_func(stage: RequestStageConfig):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if self.trace_ctx is None:
                return func(self, *args, **kwargs)
            start_ts = convert_time_to_realtime_ns(time.perf_counter())
            self.trace_ctx.trace_slice_start(stage.stage_name, stage.level, start_ts)
            ret = func(self, *args, **kwargs)
            self.trace_ctx.trace_slice_end(stage.stage_name, stage.level)
            return ret

        return wrapper

    return decorator
