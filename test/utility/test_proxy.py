import pytest
from mindpark.utility import Proxy


class Foo:

    def __init__(self, value):
        self.value = value


@pytest.fixture
def value():
    return 13


@pytest.fixture(params=['normal', 'changed', 'nested'])
def proxy(request, value):
    if request.param == 'normal':
        return Proxy(Foo(value))
    if request.param == 'changed':
        proxy = Proxy(123)
        proxy.change(Foo(value))
        return proxy
    if request.param == 'nested':
        return Proxy(Proxy(Foo(value)))
    assert False


class TestProxy:

    def test_access_inner_attribute(self, value, proxy):
        first = 13
        proxy = Proxy(Foo(first))
        assert proxy.value is first
        assert proxy.value == 13

    def test_override_attribute(self, proxy):
        proxy = Proxy(Foo(13))
        second = 42
        proxy.value = second
        assert proxy.value is second
        assert proxy.value == 42

    def test_restore_original_attribute(self, value, proxy):
        first = 13
        proxy = Proxy(Foo(first))
        proxy.value = 42
        del proxy.value
        assert proxy.value is first
        assert proxy.value == 13

    def test_access_unknown_attribute(self, proxy):
        with pytest.raises(AttributeError):
            proxy.something_else
        with pytest.raises(AttributeError):
            proxy.something_else = 42
        with pytest.raises(AttributeError):
            del proxy.something_else

    def test_nested_override_outer(self):
        inner = Proxy(Foo(13))
        proxy = Proxy(inner)
        first = 13
        proxy.value = first
        second = 42
        inner.change(Foo(second))
        assert proxy.value is first
        assert proxy.value == 13
        del proxy.value
        assert proxy.value is second
        assert proxy.value == 42
