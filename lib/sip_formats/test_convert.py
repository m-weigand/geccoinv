#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Tests for converter functions

Run with

nosetests test_convert.py -s -v
"""
from nose.tools import *
import numpy as np
import sip_formats.convert as sip_convert


class test_converters():
    @classmethod
    def teardown(self):
        pass

    def precompute_values(self):
        rmag = np.array([49.999, 31.623, 24.992])
        rpha = np.array([-5.0000, -321.7506, -24.9948])
        Z = rmag * np.exp(1j * rpha / 1000)
        Y = 1 / Z
        rre = np.real(Z)
        rim = np.imag(Z)
        cre = np.real(Y)
        cim = np.imag(Y)
        cmag = np.abs(Y)
        cpha = np.arctan2(np.imag(Y), np.real(Y)) * 1000

        self.rmag = rmag
        self.rpha = rpha
        self.rre = rre
        self.rim = rim
        self.rmim = -rim

        self.cmag = cmag
        self.cpha = cpha
        self.cre = cre
        self.cim = cim
        self.cmim = -cim

    def setup(self):
        self.precompute_values()

    def check_from_function(self, func, input1, input2):
        input_data = np.hstack((input1, input2))
        out_cre, out_cim = func(input_data)
        output_data = np.hstack((out_cre, out_cim))
        true_result = np.hstack((self.cre, self.cim))
        diffs = np.abs(true_result - output_data.flatten())
        for term in diffs:
            # check to within 8 places
            assert_almost_equal(term, 0, 8)

    def check_to_function(self, func, output1, output2):
        output_data = func(self.cre, self.cim)
        true_result = np.hstack((output1, output2))
        diffs = np.abs(true_result - output_data.flatten())
        for term in diffs:
            # check to within 8 places
            assert_almost_equal(term, 0, 8)

    def test_from_cre_cim(self):
        self.check_from_function(sip_convert.from_cre_cim, self.cre, self.cim)

    def test_from_cre_cmim(self):
        self.check_from_function(sip_convert.from_cre_cmim, self.cre,
                                 self.cmim)

    def test_from_cmag_cpha(self):
        self.check_from_function(sip_convert.from_cmag_cpha, self.cmag,
                                 self.cpha)

    def test_from_rre_rim(self):
        self.check_from_function(sip_convert.from_rre_rim, self.rre, self.rim)

    def test_from_rre_rmim(self):
        self.check_from_function(sip_convert.from_rre_rmim, self.rre,
                                 self.rmim)

    def test_from_rmag_rpha(self):
        self.check_from_function(sip_convert.from_rmag_rpha, self.rmag,
                                 self.rpha)

    def test_to_cre_cim(self):
        self.check_to_function(sip_convert.to_cre_cim, self.cre, self.cim)

    def test_to_cre_mim(self):
        self.check_to_function(sip_convert.to_cre_cmim, self.cre, self.cmim)

    def test_to_cmag_cpha(self):
        self.check_to_function(sip_convert.to_cmag_cpha, self.cmag, self.cpha)

    def test_to_rre_rim(self):
        self.check_to_function(sip_convert.to_rre_rim, self.rre, self.rim)

    def test_convert(self):
        """
        We test by converting to all values, from all values
        """
        from_keys = sip_convert.from_converters.keys()

        # we know that the first key is rmag_rpha
        initial_values = np.hstack((self.rmag, self.rpha))
        start_values = initial_values

        for nr, from_key in enumerate(from_keys):
            to = (nr + 1) % len(from_keys)
            output_data = sip_convert.convert(from_keys[nr], from_keys[to],
                                              start_values)
            start_values = output_data

        diffs = start_values - initial_values

        for term in diffs.flatten():
            # check to within 3 places
            assert_almost_equal(term, 0, 3)
