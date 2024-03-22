import time
from contextlib import contextmanager
from typing import Any, Dict, TypedDict

import GPUtil
import torch


def noise(batch: torch.Tensor, alpha_bars: torch.Tensor, t: int) -> torch.Tensor:
    epsilons = torch.randn(size=batch.shape)
    noised = 1 / alpha_bars[t]
    return


def get_alpha() -> torch.Tensor:
    return


def forward_sampling():
    return


@contextmanager
def timed(wall_times: Dict, key: str):
    start = time.time()
    torch.cuda.synchronize()
    yield
    torch.cuda.synchronize()
    end = time.time()
    elapsed_time = end - start
    wall_times[key].append(elapsed_time)

