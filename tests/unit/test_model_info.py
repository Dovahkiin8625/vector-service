"""Unit tests for ``core.model_info.describe_instance``.

The helper introspects a loaded backend wrapper without knowing its
concrete attribute layout: it must find ``torch.nn.Module`` objects
behind nested wrappers (instance -> impl -> flag model -> module),
de-duplicate shared/child parameters, and degrade to ``None`` for
objects that hold no module.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from vector_service.core import model_info
from vector_service.core.model_info import describe_instance


@pytest.fixture(autouse=True)
def _clear_cache():
    model_info._cache.clear()
    yield
    model_info._cache.clear()


class _Wrapper:
    """Mimics the real backends' wrapper layout (``_model`` / ``_impl``)."""

    def __init__(self, **attrs):
        self._device = "cpu"
        for name, value in attrs.items():
            setattr(self, name, value)


def test_none_and_module_free_object_return_none():
    assert describe_instance(None) is None
    assert describe_instance(object()) is None


def test_finds_direct_module_and_reports_counts():
    linear = nn.Linear(10, 5)  # weight 5x10 + bias 5 -> 55 fp32 params
    wrapper = _Wrapper(_model=linear)

    info = describe_instance(wrapper)

    assert info is not None
    assert info["param_count"] == 55
    # fp32 -> 4 bytes per parameter, no buffers on nn.Linear
    assert info["memory_bytes"] == 55 * 4
    assert info["dtype"] == "float32"
    assert info["device"] == "cpu"


def test_finds_module_behind_nested_impl():
    # BGE-M3 layout: instance._impl._model.model (+ sibling heads)
    head = nn.Linear(10, 2, bias=False)
    inner = type("FlagModel", (), {})()
    inner.model = nn.Sequential(nn.Linear(10, 4))
    inner.colbert_linear = head
    impl = _Wrapper(_model=inner)
    root = _Wrapper(_impl=impl)

    info = describe_instance(root)

    assert info is not None
    # Sequential's Linear: 4*10 + 4 = 44; sibling head: 2*10 = 20
    assert info["param_count"] == 64
    assert info["memory_bytes"] == 64 * 4


def test_shared_module_reachable_twice_is_counted_once():
    linear = nn.Linear(3, 2)  # 8 params
    wrapper = _Wrapper(_model=linear)
    # Same module exposed through a second attribute path.
    wrapper._alias = type("Alias", (), {})()
    wrapper._alias.model = linear

    info = describe_instance(wrapper)

    assert info is not None
    assert info["param_count"] == 8


def test_parent_and_child_modules_do_not_double_count():
    parent = nn.Sequential(nn.Linear(4, 2))  # 2*4 + 2 = 10
    wrapper = _Wrapper(_model=parent, _child=parent[0])

    info = describe_instance(wrapper)

    assert info is not None
    assert info["param_count"] == 10


def test_explicit_device_attribute_wins():
    wrapper = _Wrapper(_model=nn.Linear(2, 2))
    wrapper._device = "cuda"
    # Weights actually live on CPU, but the wrapper's dispatch device
    # (what the card should show) is the ``_device`` record.
    assert describe_instance(wrapper)["device"] == "cuda"


def test_result_is_cached_per_instance():
    wrapper = _Wrapper(_model=nn.Linear(4, 1))
    first = describe_instance(wrapper)
    # Swap the module after the first introspection: the cached entry
    # for the same instance must be returned unchanged.
    wrapper._model = nn.Linear(400, 200)
    second = describe_instance(wrapper)
    assert first is second
    assert second["param_count"] == 4 + 1
