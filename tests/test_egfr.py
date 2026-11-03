import pytest

from src.egfr import ckd_epi_2021, cockcroft_gault, stage


# Expected values worked out by hand from the CKD-EPI 2021 formula.
@pytest.mark.parametrize("scr,age,sex,expected", [
    (1.0, 50, "M", 91.7),
    (1.0, 60, "F", 64.5),
    (2.0, 70, "M", 35.2),
])
def test_ckd_epi_hand_calculated(scr, age, sex, expected):
    assert ckd_epi_2021(scr, age, sex) == pytest.approx(expected, abs=0.6)


def test_cockcroft_gault():
    assert cockcroft_gault(1.0, 40, 72, "M") == pytest.approx(100.0)
    assert cockcroft_gault(1.0, 40, 72, "F") == pytest.approx(85.0)


@pytest.mark.parametrize("egfr,expected", [(90, "G1"), (89.9, "G2"), (60, "G2"), (59.9, "G3a"), (45, "G3a"),
                                           (44.9, "G3b"), (30, "G3b"), (29.9, "G4"), (15, "G4"), (14.9, "G5")])
def test_stage_bands(egfr, expected):
    assert stage(egfr) == expected
