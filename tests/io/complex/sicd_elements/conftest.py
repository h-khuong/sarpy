#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import pytest

from sarpy.io.complex.sicd_elements import SICD


@pytest.fixture()
def sicd(tests_path):
    xml_file = tests_path / 'data/example.sicd.xml'
    structure = SICD.SICDType().from_xml_file(xml_file)

    return structure


@pytest.fixture()
def rma_sicd(tests_path):
    xml_file = tests_path / 'data/example.sicd.rma.xml'
    structure = SICD.SICDType().from_xml_file(xml_file)

    return structure


@pytest.fixture()
def kwargs():
    return {'_xml_ns': 'ns', '_xml_ns_key': 'key'}


@pytest.fixture()
def tol():
    return 1e-8
