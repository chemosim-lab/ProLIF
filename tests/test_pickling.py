import pickle

import pytest
from rdkit import Chem

from prolif.pickling import PROLIF_PICKLE_OPTIONS, RDKitPickleHandler


@pytest.fixture(autouse=True)
def reset_default_pickle_properties():
    default = Chem.GetDefaultPickleProperties()
    yield
    Chem.SetDefaultPickleProperties(default)


def test_default():
    assert (
        RDKitPickleHandler(0).default_pickle
        == RDKitPickleHandler.get()
        == Chem.GetDefaultPickleProperties()
    )


def test_pickling():
    mol = Chem.MolFromSmiles("CCO")
    mol.SetProp("foo", "bar")
    atom = mol.GetAtomWithIdx(0)
    atom.SetProp("spam", "egg")
    assert mol.HasProp("foo") and atom.HasProp("spam")
    # should discard properties
    RDKitPickleHandler(Chem.PropertyPickleOptions.NoProps).set()
    pickled = pickle.loads(pickle.dumps(mol))
    assert not pickled.HasProp("foo")
    # should keep mol and atom properties
    handler = RDKitPickleHandler(PROLIF_PICKLE_OPTIONS)
    handler.set()
    pickled = pickle.loads(pickle.dumps(mol))
    assert pickled.GetProp("foo") == mol.GetProp("foo")
    assert pickled.GetAtomWithIdx(0).GetProp("spam") == atom.GetProp("spam")
    # should go back to previous pickle options (discard properties)
    handler.reset()
    pickled = pickle.loads(pickle.dumps(mol))
    assert not pickled.HasProp("foo")
