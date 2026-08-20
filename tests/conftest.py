import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def randomize_(module: torch.nn.Module, std: float = 3.0) -> torch.nn.Module:
    """Adversarially large random weights.

    Every structural guarantee in this project must hold for *arbitrary* weights,
    not just trained ones — so the invariant tests use this instead of a trained
    model. If a guarantee only holds after training, it was a penalty in disguise.
    """
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(0.0, std)
    return module
