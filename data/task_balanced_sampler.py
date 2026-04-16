"""
TaskBalancedSampler

batch 내 task 다양성을 보장하는 sampler.
num_tasks_per_batch개의 task를 랜덤 선택하고, 각 task에서 samples_per_task개를 샘플링.
→ batch_size = num_tasks_per_batch * samples_per_task (config의 batch_size와 일치해야 함)

DDP 지원: rank / world_size 지정 시 각 rank에 겹치지 않는 배치를 할당.
"""

from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class TaskBalancedSampler(Sampler):
    def __init__(
        self,
        dataset,
        num_tasks_per_batch: int = 4,
        samples_per_task: int = 4,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        """
        Args:
            dataset: LiberoDataset (dataset.samples = [(file_idx, demo_key, t), ...])
            num_tasks_per_batch: batch당 포함할 task 수
            samples_per_task: task당 샘플 수
            rank: DDP rank (단일 GPU면 0)
            world_size: DDP world size (단일 GPU면 1)
            seed: 재현성용 시드
        """
        self.num_tasks_per_batch = num_tasks_per_batch
        self.samples_per_task = samples_per_task
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

        # task_id(=file_idx) → 샘플 인덱스 목록
        self.task_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, (file_idx, _, _) in enumerate(dataset.samples):
            self.task_to_indices[file_idx].append(idx)
        self.task_ids = sorted(self.task_to_indices.keys())
        self.num_tasks = len(self.task_ids)

        assert self.num_tasks >= num_tasks_per_batch, (
            f"task 수({self.num_tasks})가 num_tasks_per_batch({num_tasks_per_batch})보다 작음"
        )

        # epoch당 총 배치 수: 전체 샘플 수 기준으로 결정, DDP로 나눔
        total_samples = len(dataset.samples)
        batch_size = num_tasks_per_batch * samples_per_task
        total_batches = total_samples // (batch_size * world_size)
        self.num_batches = total_batches  # 이 rank가 처리할 배치 수

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch * 1000 + self.rank)

        indices = []
        for _ in range(self.num_batches):
            chosen_tasks = rng.choice(self.task_ids, size=self.num_tasks_per_batch, replace=False)
            for task_id in chosen_tasks:
                pool = self.task_to_indices[task_id]
                replace = len(pool) < self.samples_per_task
                chosen = rng.choice(pool, size=self.samples_per_task, replace=replace)
                indices.extend(chosen.tolist())

        return iter(indices)

    def __len__(self) -> int:
        return self.num_batches * self.num_tasks_per_batch * self.samples_per_task
