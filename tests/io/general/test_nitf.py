#
# Copyright 2024 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import filecmp
import numpy as np
import pathlib

import sarpy.io.general.nitf
from sarpy.io.general.format_function import IdentityFunction, ComplexFormatFunction, SingleLUTFormatFunction
import tests

def test_iq_band_interleaved_by_block(tests_path, tmp_path):
    in_nitf = tests_path / "data/iq.nitf"
    with sarpy.io.general.nitf.NITFReader(str(in_nitf)) as reader:
        data = reader.read()
        data_raw = reader.read_raw()
        assert data_raw.ndim == data.ndim + 1
        assert data_raw.size == data.size * 2
        assert np.iscomplexobj(data)

        data_offset = reader.nitf_details.img_segment_offsets[0]
        manual_bytes = in_nitf.read_bytes()[data_offset : data_offset + data.nbytes]
        manual_data_raw = np.frombuffer(manual_bytes, dtype=data_raw.dtype).reshape(
            data_raw.shape
        )
        manual_data = manual_data_raw[0] + 1j * manual_data_raw[1]  # interleaved by block
        assert np.array_equal(data, manual_data)

        out_nitf = tmp_path / "out.nitf"
        writer_details = sarpy.io.general.nitf.NITFWritingDetails(
            reader.nitf_details.nitf_header,
            (sarpy.io.general.nitf.ImageSubheaderManager(reader.get_image_header(0)),),
            reader.image_segment_collections,
        )
        with sarpy.io.general.nitf.NITFWriter(
            str(out_nitf), writing_details=writer_details
        ) as writer:
            writer.write(data)
        assert filecmp.cmp(in_nitf, out_nitf, shallow=False)


def test_write_filehandle(tests_path, tmp_path):
    in_nitf = tests_path / "data/iq.nitf"
    with sarpy.io.general.nitf.NITFReader(str(in_nitf)) as reader:
        data = reader.read()
        writer_details = sarpy.io.general.nitf.NITFWritingDetails(
            reader.nitf_details.nitf_header,
            (sarpy.io.general.nitf.ImageSubheaderManager(reader.get_image_header(0)),),
            reader.image_segment_collections,
        )

    out_nitf = tmp_path / 'output.nitf'
    with out_nitf.open('wb') as fd:
        with sarpy.io.general.nitf.NITFWriter(
            fd, writing_details=writer_details
        ) as writer:
            writer.write(data)

        assert not fd.closed
    assert filecmp.cmp(in_nitf, out_nitf, shallow=False)

def test_in_memory_write(tests_path, tmp_path):
    in_nitf_mem = tests_path / "data/iq.nitf"
    with sarpy.io.general.nitf.NITFReader(str(in_nitf_mem)) as reader_mem:
        data_mem = reader_mem.read()
        writer_details_mem = sarpy.io.general.nitf.NITFWritingDetails(
            reader_mem.nitf_details.nitf_header,
            (sarpy.io.general.nitf.ImageSubheaderManager(reader_mem.get_image_header(0)),),
            reader_mem.image_segment_collections,
        )
    
    out_nitf_mem = tmp_path / 'output_memory.nitf'
    with out_nitf_mem.open('wb') as fd_mem:
        with sarpy.io.general.nitf.NITFWriter(
            fd_mem, writing_details=writer_details_mem, in_memory=True
        ) as writer_mem:
            writer_mem.write(data_mem)

        assert not fd_mem.closed
    assert filecmp.cmp(in_nitf_mem, out_nitf_mem, shallow=False)

test_data = tests.find_test_data_files(pathlib.Path(__file__).parent / "csi.json")
rgb_nitf = test_data["csi_files"][0] if test_data["csi_files"] else None

def test_read_rgb_nitf(tests_path, tmp_path):
    with sarpy.io.general.nitf.NITFReader(str(rgb_nitf)) as reader:
        data = reader.read()
        writer_details = sarpy.io.general.nitf.NITFWritingDetails(
            reader.nitf_details.nitf_header,
            (sarpy.io.general.nitf.ImageSubheaderManager(reader.get_image_header(0)),),
            reader.image_segment_collections,
        )
        bands = reader.nitf_details.img_headers[0].Bands
        assert len(bands) == 3 , f"This NITF has ({reader.image_segment_collections.count}) bands"

        band_ids = [band.IREPBAND for band in bands]
        assert band_ids == ["R", "G", "B"], f'The band ids are ({band_ids})'

    out_nitf = tmp_path / 'output.nitf'
    with out_nitf.open('wb') as fd:
        with sarpy.io.general.nitf.NITFWriter(
            fd, writing_details=writer_details
        ) as writer:
            writer.write(data) 
        assert not fd.closed

def test_open_rgb_nitf():
    with sarpy.io.general.nitf.NITFReader(str(rgb_nitf)) as reader:
        bands = reader.nitf_details.img_headers[0].Bands
        assert len(bands) == 3 , f"This NITF has ({reader.image_segment_collections.count}) bands"
        
        band_ids = [band.IREPBAND for band in bands]
        assert band_ids == ["R", "G", "B"], f'The band ids are ({band_ids})'
# "20200727f01p0034faradx0353_188_11113VH_000_132239_SL0006R_31N086W_001X___QVH_0101_SPY_CSI.nitf"