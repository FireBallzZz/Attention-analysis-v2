"""Skip legacy tests whose private inputs are not in a public clone.

Some older test modules check historical experiment artefacts: extracted
classroom features (data/processed/, made from videos of real students) and
earlier candidate checkpoints. Those files are not published, and the
released model does not need them, so the tests are skipped when the files
are absent instead of failing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PRIVATE = {
    "test_v3_hardened_datasets.py": [ROOT / "data/processed/classroom_clip_001_v3_hardened.npz"],
    "test_phase6g4c_candidate.py": [ROOT / "data/processed/classroom_clip_001_v3_hardened.npz",
                                    ROOT / "checkpoints/classroom_manual_binary_model_v2_clean.pt"],
    "test_phase7_candidate.py": [ROOT / "checkpoints/candidates/phase7_public_trained_binary_candidate.pt"],
    "test_phase8a_rebuild.py": [ROOT / "checkpoints/classroom_manual_binary_model_v2_clean.pt",
                                ROOT / "data/processed"],
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        needed = PRIVATE.get(Path(str(item.fspath)).name)
        if needed and not all(p.exists() for p in needed):
            item.add_marker(pytest.mark.skip(reason="needs private data / historical checkpoints not in the public repository"))
