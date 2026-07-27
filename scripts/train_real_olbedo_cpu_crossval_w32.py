"""Run the group-held-out cross-validation with the full-search winner."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_real_olbedo_cpu_crossval as crossval  # noqa: E402


_original_train_candidate = crossval._train_candidate


def _width32_train_candidate(config, *args, **kwargs):
    # Mutate the shared config so checkpoints and the final report truthfully
    # record the architecture selected by the full 24/6 search.
    config.update({"architecture": "intrinsic_v3", "width": 32, "depth": 8})
    return _original_train_candidate(config, *args, **kwargs)


crossval._train_candidate = _width32_train_candidate


if __name__ == "__main__":
    crossval.main()
