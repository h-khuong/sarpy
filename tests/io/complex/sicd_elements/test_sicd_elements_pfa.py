#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import numpy as np
import pytest

from sarpy.io.complex.sicd_elements import blocks
from sarpy.io.complex.sicd_elements import PFA

def test_STDeskewType_initialization_no_parameters():
    #checks if stdeskew  is an instance of STDeskew
    stdeskew = PFA.STDeskewType()
    assert isinstance(stdeskew, PFA.STDeskewType)

def test_STDeskewType_initialization(kwargs):
    made_up_poly = blocks.Poly2DType([[10, 5], [8, 3], [1, 0.5]])
    stdeskew = PFA.STDeskewType(False, made_up_poly, **kwargs)
    assert stdeskew.Applied is False
    assert stdeskew.STDSPhasePoly == made_up_poly
    assert stdeskew._xml_ns == kwargs['_xml_ns']
    assert stdeskew._xml_ns_key == kwargs['_xml_ns_key']

@pytest.fixture()
def setup_stdeskew(kwargs):
    made_up_poly = blocks.Poly2DType([[10, 5], [8, 3], [1, 0.5]])
    yield PFA.STDeskewType(False, made_up_poly, **kwargs)

def test_PFAType_nominal_initialization(sicd, kwargs, setup_stdeskew):
    # Nominal instantiation
    pfa_nom = PFA.PFAType(sicd.PFA.FPN,
                          sicd.PFA.IPN,
                          sicd.PFA.PolarAngRefTime, 
                          sicd.PFA.PolarAngPoly,
                          sicd.PFA.SpatialFreqSFPoly,
                          sicd.PFA.Krg1,
                          sicd.PFA.Krg2,
                          sicd.PFA.Kaz1,
                          sicd.PFA.Kaz2,
                          setup_stdeskew,
                          **kwargs)
    assert isinstance(pfa_nom, PFA.PFAType)
    assert pfa_nom._xml_ns == kwargs['_xml_ns']
    assert pfa_nom._xml_ns_key == kwargs['_xml_ns_key']
    assert pfa_nom._basic_validity_check()
    assert pfa_nom._check_polar_ang_ref()

def test_PFAType_no_polaranglepoly_path(sicd, kwargs, setup_stdeskew):
    # No PolarAnglePoly path
    pfa_no_pap = PFA.PFAType(sicd.PFA.FPN,
                             sicd.PFA.IPN,
                             sicd.PFA.PolarAngRefTime,
                             None,
                             sicd.PFA.SpatialFreqSFPoly,
                             sicd.PFA.Krg1,
                             sicd.PFA.Krg2,
                             sicd.PFA.Kaz1,
                             sicd.PFA.Kaz2,
                             setup_stdeskew,
                             **kwargs)
    assert pfa_no_pap._check_polar_ang_ref()

def test_empty_PFAType(sicd, tol):
    # Populate empty PFAType with sicd components after instantiation
    pfa_empty = PFA.PFAType()
    pfa_empty._derive_parameters(sicd.Grid, sicd.SCPCOA, sicd.GeoData, sicd.Position, sicd.Timeline)
    assert pfa_empty.PolarAngRefTime == pytest.approx(sicd.SCPCOA.SCPTime, abs=tol)
    assert isinstance(pfa_empty.IPN, blocks.XYZType)
    assert isinstance(pfa_empty.FPN, blocks.XYZType)
    assert isinstance(pfa_empty.PolarAngPoly, blocks.Poly1DType) # this test was failing because line 292 from PFA.py wasn't being executed so PolarAngPoly was None 
    assert isinstance(pfa_empty.SpatialFreqSFPoly, blocks.Poly1DType)
    assert isinstance(pfa_empty.Krg1, float)
    assert isinstance(pfa_empty.Krg2, float)
    assert isinstance(pfa_empty.Kaz1, float)
    assert isinstance(pfa_empty.Kaz2, float)

def test_empty_PFAType_no_geodata(sicd, tol):
    # Try it without GeoData
    pfa_empty_no_geo = PFA.PFAType()
    pfa_empty_no_geo._derive_parameters(sicd.Grid, sicd.SCPCOA, None, sicd.Position, sicd.Timeline)
    assert pfa_empty_no_geo.PolarAngRefTime == pytest.approx(sicd.SCPCOA.SCPTime, abs=tol)
    assert pfa_empty_no_geo.IPN is None
    assert pfa_empty_no_geo.FPN is None
    assert pfa_empty_no_geo.PolarAngPoly is None
    assert pfa_empty_no_geo.SpatialFreqSFPoly is None
    assert pfa_empty_no_geo.Krg1 is None
    assert pfa_empty_no_geo.Krg2 is None
    assert pfa_empty_no_geo.Kaz1 is None
    assert pfa_empty_no_geo.Kaz2 is None

def test_PFAType_no_FPN(sicd, setup_stdeskew):
    # Without FPN to test that path
    pfa_no_fpn = PFA.PFAType(None,
                             sicd.PFA.IPN,
                             sicd.PFA.PolarAngRefTime,
                             sicd.PFA.PolarAngPoly,
                             sicd.PFA.SpatialFreqSFPoly,
                             sicd.PFA.Krg1,
                             sicd.PFA.Krg2,
                             sicd.PFA.Kaz1,
                             sicd.PFA.Kaz2,
                             setup_stdeskew)
    assert pfa_no_fpn.pfa_polar_coords(sicd.Position,
                                       sicd.GeoData.SCP.ECF[:],
                                       0.0) == (None, None)
    assert pfa_no_fpn.pfa_polar_coords(sicd.Position,
                                       sicd.GeoData.SCP.ECF[:],
                                       np.array([6378137.0, 0])) == (None, None)
