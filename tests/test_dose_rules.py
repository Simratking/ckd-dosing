import pytest

from src.dose_rules import check_contraindication, list_drugs, lookup_dose_rule


@pytest.mark.parametrize("drug,egfr,action", [
    ("metformin", 29, "avoid"), ("metformin", 30, "reduce"), ("metformin", 44, "reduce"), ("metformin", 45, "usual"),
    ("ibuprofen", 29, "avoid"), ("ibuprofen", 30, "caution"), ("ibuprofen", 59, "caution"), ("ibuprofen", 60, "usual"),
])
def test_egfr_band_edges(drug, egfr, action):
    assert lookup_dose_rule(drug, egfr)["action"] == action


@pytest.mark.parametrize("crcl,action", [(59, "reduce"), (60, "usual")])
def test_crcl_band_edges(crcl, action):
    assert lookup_dose_rule("gabapentin", 100, crcl)["action"] == action


def test_crcl_rule_uses_crcl_not_egfr():
    assert lookup_dose_rule("gabapentin", 100, 20)["dose"].startswith("200 to 700")


def test_crcl_rule_falls_back_to_egfr_and_says_so():
    r = lookup_dose_rule("gabapentin", 20)
    assert r["found"] and "weight missing" in r["basis"]


def test_unknown_drug_is_never_guessed():
    r = lookup_dose_rule("zzz-unknown", 50)
    assert r == {"drug": "zzz-unknown", "found": False, "message": "no rule on file"}


def test_aliases_and_spacing():
    assert lookup_dose_rule(" Naproxen ", 20)["action"] == "avoid"


def test_contraindication():
    assert "avoid" in check_contraindication("nitrofurantoin", 50, 20)
    assert check_contraindication("nitrofurantoin", 90, 90) is None


def test_every_rule_has_gapless_bands():
    from src.dose_rules import RULES
    for name, d in RULES.items():
        bands = sorted(d["bands"], key=lambda b: b["min"])
        assert bands[0]["min"] == 0 and bands[-1]["max"] is None, name
        for lo, hi in zip(bands, bands[1:]):
            assert lo["max"] == hi["min"], name
    assert len(list_drugs()) >= 8
