import pytest

from scripts.audit_lock import normalized_lock


def test_normalizes_only_one_pytorch_cpu_release():
    assert normalized_lock("torch==2.15.0+cpu\nnumpy==1.26.4\n") == (
        "torch==2.15.0\nnumpy==1.26.4\n"
    )
    with pytest.raises(ValueError, match="exactly one"):
        normalized_lock("numpy==1.26.4\n")
    with pytest.raises(ValueError, match="exactly one"):
        normalized_lock("torch==2.15.0+cpu\ntorch==2.14.0+cpu\n")
