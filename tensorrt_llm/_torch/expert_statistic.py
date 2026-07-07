import atexit
import json
import os

import safetensors
import torch

from tensorrt_llm.logger import logger


class ExpertStatistic:
    expert_statistic_obj = None

    @staticmethod
    def get():
        return ExpertStatistic.expert_statistic_obj

    @staticmethod
    def create(rank_id: int):
        # Enabled if EXPERT_STATISTIC_ITER_RANGE is set.
        span = os.environ.get('EXPERT_STATISTIC_ITER_RANGE', None)
        if span is None:
            return
        try:
            start, stop = span.strip().split('-')
            start, stop = int(start), int(stop)
        except ValueError as e:
            raise ValueError(str(e))
        ExpertStatistic.expert_statistic_obj = ExpertStatistic(
            rank_id, start, stop)
        logger.info(
            f'[ExpertStatistic] enabled rank={rank_id} range={start}-{stop} '
            f'dump_dispatch={ExpertStatistic.expert_statistic_obj._dump_dispatch}')

    @staticmethod
    def should_record() -> bool:
        if ExpertStatistic.expert_statistic_obj is not None:
            return ExpertStatistic.expert_statistic_obj._should_record
        return False

    @staticmethod
    def set_iter(iter_id: int) -> None:
        if ExpertStatistic.expert_statistic_obj is not None:
            ExpertStatistic.expert_statistic_obj._set_iter(iter_id)

    @staticmethod
    def set_layer(layer_id: int) -> None:
        if ExpertStatistic.expert_statistic_obj is not None:
            ExpertStatistic.expert_statistic_obj._set_layer(layer_id)

    @staticmethod
    def maybe_add_info(expert_count: int,
                       token_selected_experts: torch.Tensor) -> None:
        if ExpertStatistic.expert_statistic_obj is not None:
            ExpertStatistic.expert_statistic_obj._maybe_add_info(
                expert_count, token_selected_experts)

    def __init__(self, rank_id: int, start: int, stop: int) -> None:
        self.current_iter_id = None
        self.current_layer = None
        self.rank_id = rank_id
        self.start = start
        self.stop = stop
        self._meta_info = None
        self._records = {}
        # Router dispatch-matrix dump (for scripts/router-analysis). Enabled by
        # EXPERT_STATISTIC_DUMP_DISPATCH; records per-(iter,layer) token counts so
        # the offline converter can reconstruct per_rank_num_tokens, and stamps the
        # EP layout (ep_size / experts_per_rank) into meta_info.
        self._dump_dispatch = os.environ.get('EXPERT_STATISTIC_DUMP_DISPATCH',
                                              None) is not None
        try:
            self._ep_size = int(os.environ['EXPERT_STATISTIC_EP_SIZE'])
        except (KeyError, ValueError):
            self._ep_size = None
        self._num_tokens = {}
        self._last_written_iter = None
        # Best-effort flush on clean exit. NOTE: atexit does NOT run on SIGTERM,
        # which is how disagg graceful_cleanup stops workers — so in dispatch-dump
        # mode we also flush incrementally per-iter (see _set_iter) so data
        # survives a SIGTERM/SIGKILL of a short-lived prefill (ctx) worker.
        atexit.register(self._write)

    @property
    def _should_record(self) -> bool:
        return self.current_iter_id is not None and self.start <= self.current_iter_id < self.stop

    def _write(self) -> None:
        if not self._records:
            return
        path = os.environ.get('EXPERT_STATISTIC_PATH', 'expert_statistic')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        if self.rank_id == 0:
            meta_info = dict(self._meta_info or {})
            if self._dump_dispatch and self._ep_size is not None:
                meta_info["ep_size"] = self._ep_size
                num_experts = meta_info.get("num_experts")
                if num_experts is not None and self._ep_size > 0:
                    meta_info["experts_per_rank"] = num_experts // self._ep_size
            with open(f"{path}/meta_info.json", "w") as f:
                json.dump(meta_info, f)
        safetensors.torch.save_file(
            self._records, f"{path}/rank{self.rank_id}.safetensors")
        if self._dump_dispatch:
            with open(f"{path}/rank{self.rank_id}_numtokens.json", "w") as f:
                json.dump(self._num_tokens, f)

    def _set_iter(self, iter_id: int) -> None:
        self.current_iter_id = iter_id
        if self._dump_dispatch:
            # Flush completed iters incrementally (idempotent overwrite) so the
            # dump survives SIGTERM. At most the final in-flight iter is lost.
            if self._records and self._last_written_iter != iter_id:
                logger.info(
                    f'[ExpertStatistic] Rank={self.rank_id} flushing '
                    f'{len(self._records)} records at iter={iter_id}')
                self._write()
                self._last_written_iter = iter_id
        elif iter_id == self.stop:
            self._write()

    def _set_layer(self, layer: int) -> None:
        self.current_layer = layer

    def _maybe_add_info(self, expert_count: int,
                        token_selected_experts: torch.Tensor) -> None:
        if not self._should_record:
            return

        if self._meta_info is None:
            self._meta_info = {
                "num_experts": expert_count,
                "num_experts_per_token": token_selected_experts.size(-1)
            }
        counts = torch.bincount(token_selected_experts.flatten(),
                                minlength=expert_count)
        key = f"{self.current_iter_id}_{self.current_layer}"
        if key not in self._records:
            self._records[key] = counts.cpu()
        else:
            self._records[key] += counts.cpu()
        if self._dump_dispatch:
            num_tokens = token_selected_experts.size(0)
            self._num_tokens[key] = self._num_tokens.get(key, 0) + num_tokens
