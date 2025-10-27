import numpy
import pytest
import re
import unittest

from sarpy.io.general.format_function import IdentityFunction, ComplexFormatFunction, SingleLUTFormatFunction


class TestIdentityFunction(unittest.TestCase):
    def test_reverse(self):
        base_data = numpy.reshape(numpy.arange(6, dtype='int32'), (3, 2))

        for axis in [(0, ), (1, ), (0, 1)]:
            with self.subTest(msg='axes[{}] reverse'.format(axis)):
                func_rev = IdentityFunction(
                    raw_shape=(3, 2), formatted_shape=(3, 2), reverse_axes=axis,
                    transpose_axes=None)
                out_data = func_rev(base_data, (slice(0, 3, 1), slice(0, 2, 1)))
                test_data = numpy.flip(base_data, axis=axis)
                self.assertTrue(numpy.all(test_data == out_data), msg='reverse failure')

    def test_transpose(self):
        base_data = numpy.reshape(numpy.arange(6, dtype='int32'), (3, 2))
        func_transpose = IdentityFunction(raw_shape=(3, 2), formatted_shape=(2, 3), transpose_axes=(1, 0))
        out_data = func_transpose(base_data, (slice(0, 3, 1), slice(0, 2, 1)))
        test_data = numpy.transpose(base_data, (1, 0))
        self.assertTrue(numpy.all(test_data == out_data), msg='transpose failure')

    def test_combined(self):
        base_data = numpy.reshape(numpy.arange(6, dtype='int32'), (3, 2))

        for axis in ((0, ), (1, ), (0, 1)):
            with self.subTest(msg='combined, reverse axes = `{}`'.format(axis)):
                func = IdentityFunction(
                    raw_shape=(3, 2), formatted_shape=(2, 3), reverse_axes=axis, transpose_axes=(1, 0))
                out_data = func(base_data, (slice(0, 3, 1), slice(0, 2, 1)))
                test_data = numpy.transpose(numpy.flip(base_data, axis=axis), (1, 0))
                self.assertTrue(numpy.all(test_data == out_data), msg='combined failure')


class TestComplexFunction(unittest.TestCase):
    def test_bad_typing(self):
        for order in ['IQ', 'QI']:
            with self.assertRaises(ValueError, msg='{} typing'.format(order)):
                _ = ComplexFormatFunction('uint8', order)
            with self.assertRaises(ValueError, msg='{} typing'.format(order)):
                _ = ComplexFormatFunction('uint16', order)
            with self.assertRaises(ValueError, msg='{} typing'.format(order)):
                _ = ComplexFormatFunction('uint32', order)

        for order in ['MP', 'PM']:
            with self.assertRaises(ValueError, msg='{} typing'.format(order)):
                _ = ComplexFormatFunction('int8', order)
            with self.assertRaises(ValueError, msg='{} typing'.format(order)):
                _ = ComplexFormatFunction('int16', order)
            with self.assertRaises(ValueError, msg='{} typing'.format(order)):
                _ = ComplexFormatFunction('int32', order)

    def test_float(self):
        base_data = numpy.reshape(numpy.arange(2, 14, dtype='float32'), (2, 3, 2))

        with self.subTest(msg='IQ float'):
            func = ComplexFormatFunction(
                'float32', 'IQ', raw_shape=(2, 3, 2), formatted_shape=(2, 3), band_dimension=2)
            out_data = func(base_data, (slice(0, 2, 1), slice(0, 3, 1), slice(0, 2, 1)))
            test_data = numpy.empty((2, 3), dtype='complex64')
            test_data.real = base_data[:, :, 0]
            test_data.imag = base_data[:, :, 1                                                                                                                                                                                                                                   ]
            self.assertTrue(numpy.all(out_data == test_data), msg='IQ forward')

            inv_data = func.inverse(out_data, (slice(0, 2, 1), slice(0, 3, 1)))
            self.assertTrue(numpy.all(numpy.abs(base_data - inv_data) < 1e-10), msg='IQ inverse')

        with self.subTest(msg='QI float'):
            func = ComplexFormatFunction(
                'float32', 'QI', raw_shape=(2, 3, 2), formatted_shape=(2, 3), band_dimension=2)
            out_data = func(base_data, (slice(0, 2, 1), slice(0, 3, 1), slice(0, 2, 1)))
            test_data = numpy.empty((2, 3), dtype='complex64')
            test_data.real = base_data[:, :, 1]
            test_data.imag = base_data[:, :, 0]
            self.assertTrue(numpy.all(out_data == test_data), msg='QI forward')
            inv_data = func.inverse(out_data, (slice(0, 2, 1), slice(0, 3, 1)))
            self.assertTrue(numpy.all(numpy.abs(base_data - inv_data) < 1e-10), msg='QI inverse')

        with self.subTest(msg='MP float'):
            func = ComplexFormatFunction(
                'float32', 'MP', raw_shape=(2, 3, 2), formatted_shape=(2, 3), band_dimension=2)
            out_data = func(base_data, (slice(0, 2, 1), slice(0, 3, 1), slice(0, 2, 1)))
            magnitude = base_data[:, :, 0]
            theta = base_data[:, :, 1]
            test_data = numpy.empty((2, 3), dtype='complex64')
            test_data.real = magnitude * numpy.cos(theta)
            test_data.imag = magnitude * numpy.sin(theta)
            self.assertTrue(numpy.all(out_data == test_data), msg='MP forward')

            inv_data = func.inverse(out_data, (slice(0, 2, 1), slice(0, 3, 1)))
            inv_array_check = (base_data - inv_data)
            self.assertTrue(numpy.all(numpy.abs(inv_array_check[:, :, 0]) < 1e-5), msg='M of MP inverse')
            self.assertTrue(numpy.all(numpy.abs(numpy.sin(inv_array_check[:, :, 1])) < 1e-5), msg='P of MP inverse')

        with self.subTest(msg='PM float'):
            func = ComplexFormatFunction(
                'float32', 'PM', raw_shape=(2, 3, 2), formatted_shape=(2, 3), band_dimension=2)
            out_data = func(base_data, (slice(0, 2, 1), slice(0, 3, 1), slice(0, 2, 1)))
            magnitude = base_data[:, :, 1]
            theta = base_data[:, :, 0]
            test_data = numpy.empty((2, 3), dtype='complex64')
            test_data.real = magnitude * numpy.cos(theta)
            test_data.imag = magnitude * numpy.sin(theta)
            self.assertTrue(numpy.all(out_data == test_data), msg='PM forward')

            inv_data = func.inverse(out_data, (slice(0, 2, 1), slice(0, 3, 1)))
            inv_array_check = (base_data - inv_data)
            self.assertTrue(numpy.all(numpy.abs(inv_array_check[:, :, 1]) < 1e-5), msg='M of PM inverse')
            self.assertTrue(numpy.all(numpy.abs(numpy.sin(inv_array_check[:, :, 0])) < 1e-5), msg='P of PM inverse')

    def test_int(self):
        for raw_type in ['int8', 'int16']:
            base_data = numpy.reshape(numpy.arange(2, 14, dtype=raw_type), (2, 3, 2))

            with self.subTest(msg='IQ {}'.format(raw_type)):
                func = ComplexFormatFunction(
                    'int16', 'IQ', raw_shape=(2, 3, 2), formatted_shape=(2, 3), band_dimension=2)
                out_data = func(base_data, (slice(0, 2, 1), slice(0, 3, 1), slice(0, 2, 1)))
                test_data = numpy.empty((2, 3), dtype='complex64')
                test_data.real = base_data[:, :, 0]
                test_data.imag = base_data[:, :, 1]
                self.assertTrue(numpy.all(out_data == test_data), msg='IQ {} forward'.format(raw_type))

                inv_data = func.inverse(out_data, (slice(0, 2, 1), slice(0, 3, 1)))
                self.assertTrue(numpy.all(base_data == inv_data), msg='IQ {} inverse'.format(raw_type))

            with self.subTest(msg='QI {}'.format(raw_type)):
                func = ComplexFormatFunction(
                    'int16', 'QI', raw_shape=(2, 3, 2), formatted_shape=(2, 3), band_dimension=2)
                out_data = func(base_data, (slice(0, 2, 1), slice(0, 3, 1), slice(0, 2, 1)))
                test_data = numpy.empty((2, 3), dtype='complex64')
                test_data.real = base_data[:, :, 1]
                test_data.imag = base_data[:, :, 0]
                self.assertTrue(numpy.all(out_data == test_data), msg='QI {} forward'.format(raw_type))
                inv_data = func.inverse(out_data, (slice(0, 2, 1), slice(0, 3, 1)))
                self.assertTrue(numpy.all(base_data == inv_data), msg='QI {} inverse'.format(raw_type))

    def test_uint(self):
        for bit_depth in [8, 16]:
            raw_type = 'uint{}'.format(bit_depth)
            base_data = numpy.arange(2, 14, dtype=raw_type)
            base_data[-1] = (1 << bit_depth) - 1
            base_data = numpy.reshape(base_data, (2, 3, 2))

            with self.subTest(msg='MP {}'.format(raw_type)):
                func = ComplexFormatFunction(
                    raw_type, 'MP', raw_shape=(2, 3, 2), formatted_shape=(2, 3), band_dimension=2)
                out_data = func(base_data, (slice(0, 2, 1), slice(0, 3, 1), slice(0, 2, 1)))
                magnitude = base_data[:, :, 0]
                theta = base_data[:, :, 1]*2.0*numpy.pi/(1 << bit_depth)
                test_data = numpy.empty((2, 3), dtype='complex64')
                test_data.real = magnitude * numpy.cos(theta)
                test_data.imag = magnitude * numpy.sin(theta)
                self.assertTrue(numpy.all(out_data == test_data), msg='MP {} forward'.format(raw_type))

                inv_data = func.inverse(out_data, (slice(0, 2, 1), slice(0, 3, 1)))
                self.assertTrue(numpy.all(base_data == inv_data), msg='MP {} inverse'.format(raw_type))

            with self.subTest(msg='PM {}'.format(raw_type)):
                func = ComplexFormatFunction(
                    raw_type, 'PM', raw_shape=(2, 3, 2), formatted_shape=(2, 3), band_dimension=2)
                out_data = func(base_data, (slice(0, 2, 1), slice(0, 3, 1), slice(0, 2, 1)))
                magnitude = base_data[:, :, 1]
                theta = base_data[:, :, 0]*2*numpy.pi/(1 << bit_depth)
                test_data = numpy.empty((2, 3), dtype='complex64')
                test_data.real = magnitude * numpy.cos(theta)
                test_data.imag = magnitude * numpy.sin(theta)
                self.assertTrue(numpy.all(out_data == test_data), msg='PM {} forward'.format(raw_type))

                inv_data = func.inverse(out_data, (slice(0, 2, 1), slice(0, 3, 1)))
                self.assertTrue(numpy.all(base_data == inv_data), msg='PM {} inverse'.format(raw_type))


class TestSingleLUTFormatFunction(unittest.TestCase):
    def test_forward(self):
        lut_sizes = ((24248,), (1<<16, 3))
        rng = numpy.random.default_rng()
        for lut_size in lut_sizes:
            with self.subTest(msg=f'LUT size:{lut_size}'):
                base_data = rng.integers(lut_size[0], size=(51, 49), dtype=numpy.uint16)
                lut = rng.integers(1<<8, size=lut_size, dtype=numpy.uint8)
                out_shape = base_data.shape if len(lut_size) == 1 else base_data.shape + (lut_size[1],)
                func = SingleLUTFormatFunction(lut, base_data.shape, out_shape)
                out_data = func(base_data, (slice(0, 51, 1), slice(0, 49, 1)))
                self.assertTrue(numpy.array_equal(out_data, lut[base_data]), msg='LUT forward')
    
    def test_single_lut_format_function_init(self):
        lookup_table = numpy.arange(256, dtype=numpy.uint8)
        raw_shape = (3, 3)
        formatted_shape = (3, 3)
        reverse_axes = None
        transpose_axes = None

        base = numpy.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ], dtype = numpy.uint8)
    
        function = SingleLUTFormatFunction(lookup_table, raw_shape, formatted_shape)
        result = function(base, slice(reverse_axes), slice(transpose_axes))

        # we expect result to equal base because we aren't applying any transformations to base (reverse axes and transpose axes are both None)
        assert numpy.array_equal(result, base) 
    
    def transform_raw_slice_no_transpose_2D(self):
        lut = numpy.arange(256, dtype = numpy.uint8)

        raw_shape = (100, 200)
        formatted_shape = (100, 200)

        transpose_axes = None

        function = SingleLUTFormatFunction(lut, raw_shape, formatted_shape, transpose_axes)

        subscript = [slice(10, 20, 1), slice(30, 50, 1)]

        result = function.transform_raw_slice(subscript)

        # since transpose is None, we should get the same value
        assert numpy.array_equal(result, subscript) 
    
    def transform_raw_slice_with_transpose_2D(self):
        lut = numpy.arange(256, dtype = numpy.uint8)

        raw_shape = (100, 200)
        formatted_shape = (200, 100)

        transpose_axes = (1, 0) # flips row/column

        function = SingleLUTFormatFunction(lut, raw_shape, formatted_shape, transpose_axes)

        subscript = [slice(10, 20, 1), slice(30, 40, 1)]

        expected = (slice(30, 40, 1), slice(10, 20, 1))

        result = function.transform_raw_slice(subscript)

        # since transpose is (1,0), we should swap the row and column values
        assert numpy.array_equal(result, expected) 

    def test_transform_raw_slice_mismatched_shapes(self):
        lut = numpy.arange(256, dtype = numpy.uint8)

        raw_shape = (1, 2)
        formatted_shape = (1, 2, 3)

        transpose_axes = (0, 1)

        function = SingleLUTFormatFunction(lut, raw_shape, formatted_shape, transpose_axes)

        subscript = [slice(0, 1, 1), slice(0, 1, 1)]

        with pytest.raises(IndexError, 
                    match = re.escape("list index out of range")):
            function.transform_raw_slice(subscript)


    def test_transform_raw_slice_matching_subscript_rawshape(self):
        lut = numpy.arange(256, dtype = numpy.uint8)

        raw_shape = (1, 2)
        formatted_shape = (1, 2)

        transpose_axes = (0, 1)

        function = SingleLUTFormatFunction(lut, raw_shape, formatted_shape, transpose_axes)

        subscript = [slice(0, 1, 1)]

        with pytest.raises(ValueError, 
                    match = re.escape("The length of subscript and raw_shape must match")):
            function.transform_raw_slice(subscript)

    def test_transform_raw_slice_unpopulated_step_value(self):
        lut = numpy.arange(256, dtype = numpy.uint8)

        raw_shape = (1, 2)
        formatted_shape = (1, 2, 3)

        transpose_axes = (0, 1)

        function = SingleLUTFormatFunction(lut, raw_shape, formatted_shape, transpose_axes)

        subscript = [slice(100, 10), slice(0, 1)]

        with pytest.raises(ValueError, 
                        match = re.escape("input slice has unpopulated step value")):
            function.transform_raw_slice(subscript)