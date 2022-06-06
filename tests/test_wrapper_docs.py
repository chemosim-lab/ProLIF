import pytest

from prolif.fingerprint import _Docstring, Fingerprint
import prolif.interactions


class Wrapper:
    __doc__ = _Docstring()
    _current_func = ""


class Dummy:
    """Dummy class docs"""
    def do_something(self):
        """Method docstring"""
        return 1


@pytest.fixture
def wrap():
    method = Dummy().do_something
    wrap = Wrapper()
    wrap.__doc__ = method
    return wrap


def test_getter(wrap):
    assert Wrapper._current_func == ""
    Wrapper._current_func = "Dummy"  # simulate __getattribute__
    wrap.__doc__
    assert Wrapper.__doc__._docs["Dummy"] == "Dummy class docs"


@pytest.fixture(scope="module")
def fp():
    return Fingerprint()


@pytest.mark.parametrize("int_name", Fingerprint.list_available())
def test_fp_docs(fp, int_name):
    meth = getattr(fp, int_name.lower())
    assert type(meth)._current_func == int_name
    cls = getattr(prolif.interactions, int_name)
    assert type(meth).__doc__._docs[int_name] == cls.__doc__
