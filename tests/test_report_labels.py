"""Behavioural tests for the multilingual report-label extractor.

Each test pins down one rule from the extractor, so a change to the regexes that
breaks an earlier decision fails loudly instead of silently shifting the labels.
"""
import pytest

from knee_mri.report_labels import TARGETS, clauses, extract, normalize, unwrap

POS = 0.5   # scores above this count as a positive finding
NEG = 0.25  # explicit negatives score below the "not mentioned" default (0.28)
DEFAULT = 0.28


# --- output contract --------------------------------------------------------

def test_every_target_has_score_and_confidence():
    out = extract("Normal knee.")
    for t in TARGETS:
        assert 0.0 <= out[t] <= 1.0
        assert 0.0 <= out[t + "__conf"] <= 1.0
        assert t + "__npos" in out and t + "__nneg" in out


@pytest.mark.parametrize("report", ["", None, float("nan")])
def test_empty_or_missing_report_gives_uninformative_label(report):
    out = extract(report)
    assert out["ACL"] == pytest.approx(DEFAULT)
    assert out["ACL__conf"] < 0.1


# --- normalisation ----------------------------------------------------------

def test_normalize_folds_diacritics_and_turkish_i():
    assert normalize("Ön Çapraz BAĞ") == "on capraz bag"


def test_normalize_turns_hyphens_into_spaces():
    assert normalize("bucket-handle") == "bucket handle"


def test_unwrap_joins_hard_line_breaks_but_keeps_bullets():
    text = "There is a horizontal tear of the\nposterior horn.\n- Small effusion."
    lines = unwrap(text).split("\n")
    assert lines[0] == "There is a horizontal tear of the posterior horn."
    assert lines[1].startswith("- Small effusion")


def test_long_clauses_are_also_split_on_commas():
    long = ", ".join(["the medial meniscus shows a horizontal tear"] + ["word word word"] * 8) + "."
    assert len(clauses(long)) > 1


# --- polarity ---------------------------------------------------------------

def test_positive_acl_tear():
    assert extract("Complete tear of the anterior cruciate ligament.")["ACL"] > 0.9


@pytest.mark.parametrize("report", [
    "No tear of the anterior cruciate ligament.",              # English pre-negation
    "The anterior cruciate ligament is intact.",               # normality word
    "Ligamento cruzado anterior sin signos de rotura.",       # Spanish normal phrase
    "Ön çapraz bağda yırtık izlenmemektedir.",               # Turkish post-negation
])
def test_negated_or_normal_acl(report):
    assert extract(report)["ACL"] < NEG


@pytest.mark.parametrize("report", [
    "Ruptur des vorderen Kreuzbandes.",                       # German
    "Ön çapraz bağda tam kat yırtık izlendi.",               # Turkish
    "Rotura completa del ligamento cruzado anterior.",        # Spanish
])
def test_multilingual_positive_acl(report):
    assert extract(report)["ACL"] > 0.9


def test_uncertain_finding_lands_between_negative_and_positive():
    score = extract("Possible tear of the medial meniscus.")["Medial Meniscus"]
    certain = extract("Tear of the medial meniscus.")["Medial Meniscus"]
    assert POS < score < certain


def test_contrast_word_stops_negation():
    out = extract("No fracture but there is bone marrow edema of the lateral femoral condyle.")
    assert out["Fracture"] < NEG
    assert out["Contusion"] > POS


# --- severity ---------------------------------------------------------------

def test_small_effusion_scores_lower_than_marked_effusion():
    small = extract("Small joint effusion.")["Effusion"]
    large = extract("Marked joint effusion.")["Effusion"]
    assert POS < small < large


# --- anatomy and side -------------------------------------------------------

def test_medial_meniscus_tear_does_not_fire_lateral():
    out = extract("Horizontal tear of the posterior horn of the medial meniscus.")
    assert out["Medial Meniscus"] > 0.9
    assert out["Lateral Meniscus"] == pytest.approx(DEFAULT)


def test_both_menisci_phrase_hits_both_sides():
    out = extract("Degenerative tears of both menisci.")
    assert out["Medial Meniscus"] > POS and out["Lateral Meniscus"] > POS


# --- osteoarthritis compartments -------------------------------------------

def test_global_oa_is_inherited_by_all_compartments():
    out = extract("Tricompartmental osteoarthritis.")
    assert all(out[t] > POS for t in ("Medial OA", "Lateral OA", "PF OA"))


def test_sited_oa_stays_in_its_compartment():
    out = extract("Moderate osteoarthritis of the medial femorotibial compartment.")
    assert out["Medial OA"] > 0.9
    assert out["Lateral OA"] < POS
    assert out["PF OA"] < POS


# --- decoys -----------------------------------------------------------------

def test_microfracture_is_not_a_fracture():
    assert extract("Status post microfracture of the medial femoral condyle.")["Fracture"] < POS


def test_bakers_cyst():
    assert extract("Small Baker cyst.")["Baker's"] > POS


# --- synovitis back-off -----------------------------------------------------

def test_synovitis_backoff_follows_effusion_with_low_confidence():
    out = extract("Large joint effusion.")
    assert out["Synovitis__npos"] == 0
    assert out["Synovitis"] > extract("Normal knee.")["Synovitis"]
    assert out["Synovitis__conf"] < 0.2
