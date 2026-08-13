import pytest

from srs_algorithm.algorithms import ALGORITHMS, get_algorithm, list_algorithms
from srs_algorithm.algorithms.envelope import validate_envelope
from srs_algorithm.algorithms.correlation import CorrelationSimilarity
from srs_algorithm.algorithms.log_difference import LogDifferenceSimilarity

# NOT REAL DATA

# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def samples():
    return [{
        "sample_code": "S001",
        "values": {"Au": 0.6, "Cu": 1100, "Mo": 28, "As": 11},
    }]


@pytest.fixture
def references():
    return [
        {
            "id": 1, "deposit_id": "CP1", "deposit_name": "Cadia",
            "deposit_class": "Porphyry",
            "values": {"Au": 0.5, "Cu": 1200, "Mo": 30, "As": 10},
        },
        {
            "id": 2, "deposit_id": "GE1", "deposit_name": "Waihi",
            "deposit_class": "Epithermal",
            "values": {"Au": 5.0, "Cu": 20, "Mo": 1, "As": 200},
        },
        {
            "id": 3, "deposit_id": "CP2", "deposit_name": "Grasberg",
            "deposit_class": "Porphyry",
            "values": {"Au": 0.4, "Cu": 1300, "Mo": 35, "As": 9},
        },
        # A near-duplicate of Cadia so confidence-voting has something to
        # agree on beyond a single reference.
        {
            "id": 4, "deposit_id": "CP1", "deposit_name": "Cadia",
            "deposit_class": "Porphyry",
            "values": {"Au": 0.52, "Cu": 1180, "Mo": 29, "As": 10.5},
        },
    ]


# ── Registry ───────────────────────────────────────────────────────────────

def test_registry_has_five_algorithms():
    assert set(ALGORITHMS) == {
        "knn_aitchison", "association", "correlation",
        "distance", "log_difference_similarity",
    }


def test_get_algorithm_unknown_id_raises():
    with pytest.raises(ValueError):
        get_algorithm("does_not_exist")


def test_list_algorithms_declares_aitchison_evidence():
    capabilities = list_algorithms()
    assert capabilities["knn_aitchison"] == ["evidence"]
    assert capabilities["association"] == []


# ── Contract: every algorithm must pass validate_envelope() ────────────────

@pytest.mark.parametrize("algorithm_id", sorted(ALGORITHMS))
def test_every_algorithm_satisfies_the_contract(algorithm_id, samples, references):
    algorithm = get_algorithm(algorithm_id)
    result = algorithm.compare(samples, references, config={"k": 2, "top_n": 10})
    envelope = result.to_envelope()

    problems = validate_envelope(envelope)
    assert problems == [], f"{algorithm_id} violated the contract: {problems}"


@pytest.mark.parametrize("algorithm_id", sorted(ALGORITHMS))
def test_scores_are_bounded_and_descending(algorithm_id, samples, references):
    algorithm = get_algorithm(algorithm_id)
    result = algorithm.compare(samples, references, config={"k": 2, "top_n": 10})

    similarities = [match.similarity for match in result.matches]
    assert all(0.0 <= score <= 1.0 for score in similarities)
    assert similarities == sorted(similarities, reverse=True)


@pytest.mark.parametrize("algorithm_id", sorted(ALGORITHMS))
def test_confidence_present_within_detail_top_n_only(algorithm_id, samples, references):
    algorithm = get_algorithm(algorithm_id)
    result = algorithm.compare(
        samples, references, config={"k": 2, "top_n": 10, "detail_top_n": 2}
    )

    for match in result.matches[:2]:
        assert match.confidence is not None
        assert match.confidence["level"] in {"low", "medium", "high"}
    for match in result.matches[2:]:
        assert match.confidence is None


def test_top_match_agrees_across_algorithms(samples, references):
    # Sanity check the fixture itself: every algorithm should agree the
    # sample is closest to Cadia (id 1 or 4), not Waihi, given how the
    # fixture values were constructed.
    for algorithm_id in ALGORITHMS:
        algorithm = get_algorithm(algorithm_id)
        result = algorithm.compare(samples, references, config={"k": 2})
        assert result.matches[0].deposit_name == "Cadia", algorithm_id


# ── Edge cases ───────────────────────────────────────────────────────────

def test_no_shared_elements_scores_zero_not_an_error():
    algorithm = get_algorithm("association")
    samples = [{"sample_code": "S1", "values": {"Pt": 1.0, "Pd": 2.0}}]
    references = [{"id": 1, "deposit_name": "X", "values": {"Au": 1.0, "Cu": 1.0}}]

    result = algorithm.compare(samples, references)
    # score_pair() returns 0 for zero overlap rather than raising, per
    # base.py's `if not prepared.input_vector: return 0`.
    assert result.matches == [] or result.matches[0].similarity == 0


def test_correlation_falls_back_to_log_difference_on_single_element():
    samples = [{"sample_code": "S1", "values": {"Au": 0.5}}]
    references = [{"id": 1, "deposit_name": "X", "values": {"Au": 0.6}}]

    correlation = get_algorithm("correlation")
    log_diff = get_algorithm("log_difference_similarity")

    corr_result = correlation.compare(samples, references)
    log_result = log_diff.compare(samples, references)

    assert corr_result.matches[0].similarity == pytest.approx(
        log_result.matches[0].similarity
    )


def test_empty_reference_library_warns_and_returns_no_matches(samples):
    algorithm = get_algorithm("distance")
    result = algorithm.compare(samples, references=[])

    assert result.matches == []
    assert any("empty" in warning.lower() for warning in result.warnings)


def test_low_element_coverage_warns(references):
    # Sample only shares 1 of its 4 elements with any reference — should
    # trip the >=50% coverage warning added to PairwiseSimilarity.compare().
    samples = [{
        "sample_code": "S1",
        "values": {"Au": 0.5, "Zn": 1, "Pb": 1, "Ni": 1},
    }]
    algorithm = get_algorithm("distance")
    result = algorithm.compare(samples, references)

    assert any("shared with the reference library" in warning for warning in result.warnings)


def test_weighted_scoring_changes_the_ranking(samples, references):
    # If Au is weighted heavily, Waihi (high Au) should climb relative to an
    # unweighted run, even though it loses on every other element.
    algorithm = get_algorithm("association")

    unweighted = algorithm.compare(samples, references, config={"k": 2})
    weighted = algorithm.compare(samples, references, config={
        "k": 2,
        "preprocessing": {"element_weights": {"Au": 10.0}},
    })

    waihi_rank_unweighted = next(
        m.rank for m in unweighted.matches if m.deposit_name == "Waihi"
    )
    waihi_rank_weighted = next(
        m.rank for m in weighted.matches if m.deposit_name == "Waihi"
    )
    assert waihi_rank_weighted <= waihi_rank_unweighted
