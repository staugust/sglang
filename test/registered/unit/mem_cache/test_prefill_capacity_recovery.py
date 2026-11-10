"""CPU regressions for retrying NO_TOKEN after asynchronous KV unlocks."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.prefill_budget import PrefillBudget, SWAPrefillBudget
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class AdmissionRetried(Exception):
    pass


class TestPrefillCapacityRecovery(unittest.TestCase):
    def setUp(self):
        self.capacity = dict(free=79104, evictable=200512)
        self.scheduler = Scheduler.__new__(Scheduler)
        self.scheduler.tree_cache = SimpleNamespace(
            supports_mamba=lambda: False,
            evictable_size=lambda: self.capacity["evictable"],
        )
        allocator = SimpleNamespace(
            page_size=64, available_size=lambda: self.capacity["free"]
        )
        allocator.create_prefill_budget = lambda cache: PrefillBudget(allocator, cache)
        self.scheduler.token_to_kv_pool_allocator = allocator
        self.scheduler.attn_tp_group = SimpleNamespace(world_size=1)
        self.scheduler.attn_cp_group = SimpleNamespace(world_size=1)
        self.batch = ScheduleBatch(reqs=[object()], batch_is_full=True)
        self.scheduler.grammar_manager = SimpleNamespace(
            has_waiting_grammars=lambda: False
        )
        self.scheduler.enable_priority_preemption = False
        self.scheduler.is_hybrid_swa = False
        self.scheduler.waiting_queue = [object()]
        self.scheduler.chunked_req = None
        self.scheduler.min_free_slots_delayer = None
        self.scheduler.get_num_allocatable_reqs = lambda *args, **kwargs: 1
        self.scheduler.processed_tokens_counter = 0
        self.scheduler.policy = SimpleNamespace(
            calc_priority=MagicMock(side_effect=AdmissionRetried)
        )

    def try_prefill(self):
        return self.scheduler._get_new_batch_prefill_raw(None, self.batch)

    def test_backup_unlock_retries_without_another_request_finishing(self):
        self.scheduler._record_prefill_no_token(self.batch)
        self.assertEqual(self.batch.prefill_no_token_capacity, 279616)
        self.assertEqual(self.try_prefill(), (None, self.batch))
        # ACK drops the final backup lock: free stays fixed, protected KV
        # becomes evictable. The running requests and request slots do not change.
        self.capacity["evictable"] += 285952
        with self.assertRaises(AdmissionRetried):
            self.try_prefill()
        self.assertFalse(self.batch.batch_is_full)
        self.assertIsNone(self.batch.prefill_no_token_capacity)
        self.scheduler.policy.calc_priority.assert_called_once()

    def test_no_retry_until_effective_capacity_increases(self):
        self.scheduler._record_prefill_no_token(self.batch)
        for _ in range(3):
            self.assertEqual(self.try_prefill(), (None, self.batch))
        # Eviction merely moves capacity from evictable to free.
        self.capacity["free"] += 64
        self.capacity["evictable"] -= 64
        self.assertEqual(self.try_prefill(), (None, self.batch))
        self.scheduler.policy.calc_priority.assert_not_called()

    def test_failed_retry_records_new_capacity(self):
        self.scheduler._record_prefill_no_token(self.batch)
        self.capacity["free"] += 64
        self.scheduler._retry_prefill_after_capacity_recovery(self.batch)
        self.assertFalse(self.batch.batch_is_full)
        # Normal admission may still reject the request; its new failure
        # snapshot prevents retrying every subsequent decode step.
        self.batch.batch_is_full = True
        self.scheduler._record_prefill_no_token(self.batch)
        self.scheduler._retry_prefill_after_capacity_recovery(self.batch)
        self.assertTrue(self.batch.batch_is_full)

    def test_request_slot_full_is_not_cleared(self):
        self.capacity["free"] += 1024
        self.assertEqual(self.try_prefill(), (None, self.batch))
        self.assertTrue(self.batch.batch_is_full)

    def test_cleared_full_discards_old_failure(self):
        self.scheduler._record_prefill_no_token(self.batch)
        self.batch.batch_is_full = False
        self.scheduler._retry_prefill_after_capacity_recovery(self.batch)
        self.assertIsNone(self.batch.prefill_no_token_capacity)

    def test_multi_resource_budget_is_not_tracked(self):
        allocator = self.scheduler.token_to_kv_pool_allocator
        allocator.create_prefill_budget = lambda cache: object.__new__(SWAPrefillBudget)
        self.scheduler._record_prefill_no_token(self.batch)
        self.assertIsNone(self.batch.prefill_no_token_capacity)

    def test_peer_capacity_recovery_retries_on_both_attention_groups(self):
        self.scheduler._record_prefill_no_token(self.batch)
        self.scheduler.attn_cp_group = SimpleNamespace(world_size=2, cpu_group="cp")
        self.scheduler.attn_tp_group = SimpleNamespace(world_size=2, cpu_group="tp")

        def peer_recovered(tensor, *, op, group):
            self.assertEqual(op, torch.distributed.ReduceOp.MAX)
            tensor.fill_(1)

        with patch(
            "torch.distributed.all_reduce", side_effect=peer_recovered
        ) as reduce:
            self.scheduler._retry_prefill_after_capacity_recovery(self.batch)
        self.assertEqual(
            [c.kwargs["group"] for c in reduce.call_args_list], ["cp", "tp"]
        )
        self.assertFalse(self.batch.batch_is_full)


if __name__ == "__main__":
    unittest.main()
