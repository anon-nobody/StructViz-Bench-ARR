"""Unit tests for scripts/analysis/scoring.py (canonical_em)."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts", "analysis"))

from scoring import canonical_em, fold, object_fold_collisions, row_score, strip_index  # noqa: E402


# (a) existing normalisation -------------------------------------------------------------
@pytest.mark.parametrize("pred,key", [("Increasing", "increasing"), ("55.130", "55.13"),
                                      ("3.0", "3"), (" yes ", "yes")])
def test_a_existing_normalisation(pred, key):
    assert canonical_em(pred, None, key) == 1.0


@pytest.mark.parametrize("pred,key", [("48.36", "48.17"), ("10%", "10"), ("0.07", "-0.07"),
                                      ("1: 5", "5"), ("5 units", "5"), ("five", "5")])
def test_a_numeric_keys_unchanged(pred, key):
    assert canonical_em(pred, None, key) == 0.0


# (b) index prefix ----------------------------------------------------------------------
@pytest.mark.parametrize("pred,key", [("11: cloud", "cloud"), ("19.10B", "10B"),
                                      ("0:BET", "BET"), ("4. apparel", "apparel"),
                                      ("23: store_03", "store_03")])
def test_b_index_prefix(pred, key):
    assert canonical_em(pred, None, key) == 1.0


@pytest.mark.parametrize("pred,key", [("11: cloud", "rain"), ("11 cloud", "cloud"),
                                      ("cloud: 11", "cloud")])
def test_b_no_change(pred, key):
    assert canonical_em(pred, None, key) == 0.0


def test_strip_index():
    assert strip_index("12 : x") == "x"
    assert strip_index("x: 12") == "x: 12"


# (c) lexicon ----------------------------------------------------------------------------
@pytest.mark.parametrize("pred,key", [("positively", "positive"), ("Negatively.", "negative"),
                                      ("positively correlated", "positive"),
                                      ("rising", "increasing"), ("falling", "decreasing"),
                                      ("3: negatively", "negative")])
def test_c_lexicon(pred, key):
    assert canonical_em(pred, None, key) == 1.0


@pytest.mark.parametrize("pred,key", [("increasing", "positive"), ("rising", "higher"),
                                      ("positively", "negative"), ("falling", "lower"),
                                      ("rising", "decreasing"), ("increase", "increasing")])
def test_c_no_change(pred, key):
    assert canonical_em(pred, None, key) == 0.0


# (d) folding ----------------------------------------------------------------------------
@pytest.mark.parametrize("pred,key", [("second_half", "second half"), ("w. & h", "W. & H."),
                                      ("svm-w/o neg", "SVM-w/o neg."), ("'east'", "east"),
                                      ("store 03", "store_03"), ("First  Half", "first half")])
def test_d_folding(pred, key):
    assert canonical_em(pred, None, key) == 1.0


@pytest.mark.parametrize("pred,key", [("HDSA", "HDSA (+)"), ("en de", "en-de"),
                                      ("svmwo neg", "SVM-w/o neg."), ("", "none")])
def test_d_no_change(pred, key):
    assert canonical_em(pred, None, key) == 0.0


def test_fold():
    assert fold("  Second_Half. ") == "second half"
    assert fold("HDSA (+)") == "hdsa (+)"


# (e) raw fallback ------------------------------------------------------------------------
@pytest.mark.parametrize("pred,raw,key", [("10", "10B", "10B"), ("08", "CoNLL08", "CoNLL08"),
                                          ("4", "The answer is DCGCN4.", "DCGCN4"),
                                          ("18", "C18", "c18")])
def test_e_raw_fallback(pred, raw, key):
    assert canonical_em(pred, raw, key) == 1.0


@pytest.mark.parametrize("pred,raw,key", [("10", "10", "10B"), ("10", None, "10B"),
                                          ("10", "12B", "10B"), ("1.5", "1.5B", "1.5B"),
                                          ("10", "10B", "10C"), ("08", "CoNLL08", "CoNLL09")])
def test_e_no_change(pred, raw, key):
    assert canonical_em(pred, raw, key) == 0.0


# monotonicity -----------------------------------------------------------------------------
def test_stored_one_is_kept():
    assert canonical_em("anything", None, "other", stored=1.0) == 1.0


def test_row_score_uses_stored_and_raw():
    r = {"prediction": "10", "raw_response": "10B", "answer": "10B", "exact_match": "0.0"}
    assert row_score(r) == 1.0
    r2 = {"prediction": "10", "raw_response": "None", "answer": "10B", "exact_match": "0.0"}
    assert row_score(r2) == 0.0
    r3 = {"prediction": "x", "answer": "y", "exact_match": "1.0"}
    assert row_score(r3, raw_field=None) == 1.0


def test_object_fold_collisions():
    assert object_fold_collisions(["store_03", "store 03", "x"]) == [("store_03", "store 03")]
    assert object_fold_collisions(["HDSA", "HDSA (+)", "1.0", "1"]) == []
