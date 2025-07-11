from sarpy.io.complex.sicd import AmpLookupFunction
from sarpy.io.general.format_function import ComplexFormatFunction, FormatFunction

import numpy
import pytest
import re

def test_amp_lookup_init():
    acceptable_raw_dtypes = "uint8"
    magnitude_lookup_table = numpy.array([0.0]*256, numpy.float32)

    AmpLookupFunction(acceptable_raw_dtypes, magnitude_lookup_table)

def test_amp_lookup_init_failure_no_input():
    with pytest.raises(TypeError, 
                        match = re.escape("AmpLookupFunction.__init__() missing 2 required positional arguments: 'raw_dtype' and 'magnitude_lookup_table'")):
        AmpLookupFunction()

def test_amp_lookup_init_failure_one_argument():
    with pytest.raises(TypeError, 
                        match = re.escape("AmpLookupFunction.__init__() missing 1 required positional argument: 'magnitude_lookup_table'")):
        AmpLookupFunction(numpy.array([0.0]*256, numpy.float32)) # assumes that the value passed is the raw_dtype so it thinks the magnitude lookup table is missing

def test_amp_lookup_init_failure_raw_dtype_float32():
    with pytest.raises(ValueError, 
                        match = re.escape("A magnitude lookup table has been supplied, but the raw datatype is not `uint8`.")):
        AmpLookupFunction("float32", numpy.array([0.0]*256, numpy.float32))

def test_amp_lookup_init_failure_magnitude_lookup_table_list():
    with pytest.raises(ValueError, 
                        match = re.escape("requires a numpy.ndarray, got <class 'list'>")):
        AmpLookupFunction("float32", [0,1,2,3])

def test_amp_lookup_init_failure_magnitude_lookup_numpy_uint8():
    with pytest.raises(ValueError, 
                        match = re.escape("requires a numpy.ndarray of float32 or 64 dtype, got uint8")):
        AmpLookupFunction("uint8", numpy.array([0.0]*256, numpy.uint8))

def test_amp_lookup_init_failure_magnitude_lookup_not_256():
    with pytest.raises(ValueError, 
                        match = re.escape("Requires a one-dimensional numpy.ndarray with 256 elements, got shape (255,)")):
        AmpLookupFunction("uint8", numpy.array([0.0]*255, numpy.float32))

def test_amp_lookup_init_failure_magnitude_lookup_two_dimensional():
    with pytest.raises(ValueError, 
                        match = re.escape("Requires a one-dimensional numpy.ndarray with 256 elements, got shape (2, 3)")):
        AmpLookupFunction("uint8", numpy.array([[1,2,3], [4,5,6]], numpy.float32))

@pytest.fixture()
def setup_amplookup():
    acceptable_raw_dtypes = "uint8"
    magnitude_lookup_table = numpy.array([0.0]*256, numpy.float32)

    yield AmpLookupFunction(acceptable_raw_dtypes, magnitude_lookup_table)

def test_get_magnitude_lookup(setup_amplookup):
    magnitude_lookup_table = numpy.array([0.0]*256, numpy.float32)
    assert numpy.array_equal(setup_amplookup.magnitude_lookup_table, magnitude_lookup_table)

def test_set_magnitude_lookup(setup_amplookup):
    new_magnitude_lookup_table = numpy.array([1.0]*256, numpy.float32)
    setup_amplookup.set_magnitude_lookup(new_magnitude_lookup_table)
    assert numpy.array_equal(setup_amplookup.magnitude_lookup_table, new_magnitude_lookup_table)

def test_set_magnitude_lookup_failure_non_numpy_array(setup_amplookup):
    non_numpy_array = [1,2,3,4]
    with pytest.raises(ValueError, 
                       match = re.escape("requires a numpy.ndarray, got <class 'list'>")):
        setup_amplookup.set_magnitude_lookup(non_numpy_array)

def test_set_magnitude_lookup_failure_non_float_32(setup_amplookup):
    non_float32 = numpy.array([1.0]*256, numpy.uint8) 
    with pytest.raises(ValueError,
                       match = re.escape("requires a numpy.ndarray of float32 or 64 dtype, got uint8")):
        setup_amplookup.set_magnitude_lookup(non_float32)

def test_set_magnitude_lookup_non256(setup_amplookup):
    not_256_elements = numpy.array([0.0]*255, numpy.float32)
    with pytest.raises(ValueError, match = re.escape("Requires a one-dimensional numpy.ndarray with 256 elements, got shape (255,)")):
        setup_amplookup.set_magnitude_lookup(not_256_elements)


def test_set_magnitude_lookup_two_dimensional(setup_amplookup):
    two_dimensional = numpy.array([[1,2,3], [4,5,6]], numpy.float32)
    with pytest.raises(ValueError, 
                        match = re.escape("Requires a one-dimensional numpy.ndarray with 256 elements, got shape (2, 3)")):
        setup_amplookup.set_magnitude_lookup(two_dimensional)

# not sure how to test line 95

def test_forward_magnitude_theta(setup_amplookup):
    data = numpy.array([1, 1, 1], numpy.uint32) 
    out = numpy.empty_like(data)
    magnitude = setup_amplookup.magnitude_lookup_table
    theta = numpy.array([0, 1, 2], numpy.float32)
    subscript = (slice(0, 0), slice(1, 1), slice(2,2))
    setup_amplookup._forward_magnitude_theta(data, out, magnitude, theta, subscript)

    #  subscript = (0,1,2)
    # IndexError: arrays used as indices must be of integer (or boolean) type

# def test_reverse_magnitude_theta(setup_amplookup):
#     setup_amplookup._reverse_magnitude_theta()
