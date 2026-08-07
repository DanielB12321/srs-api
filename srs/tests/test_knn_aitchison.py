"""
Tests for the compositional algorithm.

The envelope contract is already enforced for every registered algorithm by
test_algorithms.py, so these check what is specific to this one: that it really
behaves compositionally, that the two reported metrics measure different
things, and that the evidence decomposition adds up.

Compositional properties are asserted directly rather than by pinning numbers,
because "distance is unchanged when a sample is diluted" is the actual claim
being made. A pinned float would still pass if the property broke.
"""

from math import sqrt

from django.test import SimpleTestCase

from ..algorithms import get_algorithm, validate_envelope
from ..algorithms.knn_aitchison import (
    DEFAULT_K,
    KnnAitchisonSimilarity,
    _average_ranks,
)
from ..preprocessing import prepare_vectors, resolve_options

SAMPLE_VALUES = {"Cu": 100.0, "Zn": 10.0, "Au": 1.0}
ELEMENTS = set(SAMPLE_VALUES)


def prepared(input_values, reference_values, preprocessing=None, imputed=None):
    return prepare_vectors(
        input_values,
        reference_values,
        set(input_values) & set(reference_values),
        resolve_options(preprocessing),
        imputed,
    )


class CompositionalBehaviourTests(SimpleTestCase):
    """The properties that justify using this algorithm at all."""

    def setUp(self):
        self.algorithm = KnnAitchisonSimilarity()

    def score(self, reference_values, preprocessing=None):
        return self.algorithm.score_vectors(
            prepared(SAMPLE_VALUES, reference_values, preprocessing)
        )

    def test_an_identical_composition_scores_one(self):
        self.assertAlmostEqual(self.score(dict(SAMPLE_VALUES)), 1.0, places=12)

    def test_dilution_does_not_change_the_result(self):
        """
        The defining property. A sample at a tenth the concentration throughout
        is the same composition, so it must be indistinguishable from the
        original. This is what concentration-based methods get wrong.
        """
        diluted = {symbol: value / 10 for symbol, value in SAMPLE_VALUES.items()}
        concentrated = {symbol: value * 1000 for symbol, value in SAMPLE_VALUES.items()}

        self.assertAlmostEqual(self.score(diluted), 1.0, places=12)
        self.assertAlmostEqual(self.score(concentrated), 1.0, places=12)

    def test_a_changed_ratio_does_change_the_result(self):
        """Dilution invariance would be worthless if it ignored everything."""
        self.assertLess(self.score({"Cu": 1000.0, "Zn": 10.0, "Au": 1.0}), 1.0)

    def test_the_result_is_the_same_whatever_preprocessing_selected(self):
        """
        The user picks algorithm and preprocessing independently, so this
        applies the CLR itself when the request did not. All three routes must
        arrive at the same place.
        """
        reference = {"Cu": 1000.0, "Zn": 10.0, "Au": 1.0}
        raw = self.score(reference)
        logged = self.score(reference, {"log_transform": True})
        clr = self.score(reference, {"normalise": True})

        self.assertAlmostEqual(raw, logged, places=12)
        self.assertAlmostEqual(raw, clr, places=12)

    def test_similarity_falls_as_distance_grows(self):
        """The squashing is monotone, so ranking follows the distance."""
        closer = self.score({"Cu": 200.0, "Zn": 10.0, "Au": 1.0})
        further = self.score({"Cu": 10000.0, "Zn": 10.0, "Au": 1.0})

        self.assertGreater(closer, further)

    def test_similarity_is_exactly_the_documented_squashing_of_the_distance(self):
        """
        Ties the two reported numbers together. Without this, the similarity
        and the aitchison_distance in the same scores block could drift apart
        and every other test here would still pass, because monotonicity and
        the identical-composition case hold for any 1 / (1 + c * distance).
        """
        for reference in (
            {"Cu": 1000.0, "Zn": 10.0, "Au": 1.0},
            {"Cu": 10.0, "Zn": 100.0, "Au": 1.0},
            {"Cu": 1e6, "Zn": 10.0, "Au": 1e-6},
        ):
            with self.subTest(reference=reference):
                vectors = prepared(SAMPLE_VALUES, reference)
                distance = self.algorithm.raw_scores(vectors)["aitchison_distance"]

                self.assertAlmostEqual(
                    self.algorithm.score_vectors(vectors),
                    1 / (1 + distance),
                    places=12,
                )

    def test_similarity_stays_within_the_contract_range(self):
        for reference in (
            dict(SAMPLE_VALUES),
            {"Cu": 1e6, "Zn": 1e-6, "Au": 1.0},
            {"Cu": 1e-6, "Zn": 1e6, "Au": 1.0},
        ):
            with self.subTest(reference=reference):
                score = self.score(reference)
                self.assertGreater(score, 0.0)
                self.assertLessEqual(score, 1.0)


class RawMetricTests(SimpleTestCase):
    """Aitchison distance and Spearman rho answer different questions."""

    def setUp(self):
        self.algorithm = KnnAitchisonSimilarity()

    def metrics(self, reference_values):
        return self.algorithm.raw_scores(prepared(SAMPLE_VALUES, reference_values))

    def test_an_identical_composition_has_zero_distance_and_perfect_rho(self):
        metrics = self.metrics(dict(SAMPLE_VALUES))

        self.assertAlmostEqual(metrics["aitchison_distance"], 0.0, places=12)
        self.assertAlmostEqual(metrics["spearman_rho"], 1.0, places=12)

    def test_distance_matches_the_euclidean_definition_in_clr_space(self):
        """
        Cu ten times higher. In log space the input is 0, 2, 1 (mean 1) and the
        reference is 0, 3, 1 (mean 4/3), so the CLR difference is +1/3 on Au
        and Zn and -2/3 on Cu.
        """
        metrics = self.metrics({"Cu": 1000.0, "Zn": 10.0, "Au": 1.0})
        expected = sqrt((1 / 3) ** 2 + (2 / 3) ** 2 + (1 / 3) ** 2)

        self.assertAlmostEqual(metrics["aitchison_distance"], expected, places=12)

    def test_rho_detects_a_reversed_element_order(self):
        reversed_pattern = {"Cu": 1.0, "Zn": 10.0, "Au": 100.0}

        self.assertAlmostEqual(
            self.metrics(reversed_pattern)["spearman_rho"],
            -1.0,
            places=12,
        )

    def test_a_pair_can_be_distant_yet_perfectly_rank_correlated(self):
        """
        Why both metrics are worth reporting. This reference orders its
        elements exactly as the sample does, but sits far away in composition.
        """
        metrics = self.metrics({"Cu": 1e6, "Zn": 10.0, "Au": 1e-6})

        self.assertAlmostEqual(metrics["spearman_rho"], 1.0, places=12)
        self.assertGreater(metrics["aitchison_distance"], 5.0)

    def test_average_ranks_share_the_rank_across_ties(self):
        self.assertEqual(_average_ranks([10.0, 20.0, 30.0]), [1.0, 2.0, 3.0])
        self.assertEqual(_average_ranks([5.0, 5.0, 9.0]), [1.5, 1.5, 3.0])
        self.assertEqual(_average_ranks([7.0, 7.0, 7.0]), [2.0, 2.0, 2.0])


class EvidenceTests(SimpleTestCase):
    """The per-element decomposition of the distance."""

    def setUp(self):
        self.algorithm = KnnAitchisonSimilarity()

    def test_the_element_that_differs_is_reported_as_conflicting(self):
        supporting, conflicting = self.algorithm.evidence(
            prepared(SAMPLE_VALUES, {"Cu": 1000.0, "Zn": 10.0, "Au": 1.0})
        )

        self.assertEqual([entry.element for entry in conflicting], ["Cu"])
        self.assertEqual({entry.element for entry in supporting}, {"Au", "Zn"})

    def test_contributions_are_signed_by_which_side_they_fall_on(self):
        supporting, conflicting = self.algorithm.evidence(
            prepared(SAMPLE_VALUES, {"Cu": 1000.0, "Zn": 10.0, "Au": 1.0})
        )

        for entry in supporting:
            self.assertGreaterEqual(entry.contribution, 0.0)
        for entry in conflicting:
            self.assertLess(entry.contribution, 0.0)

    def test_contributions_are_scaled_to_a_total_of_one(self):
        supporting, conflicting = self.algorithm.evidence(
            prepared(SAMPLE_VALUES, {"Cu": 1000.0, "Zn": 5.0, "Au": 1.0})
        )
        total = sum(
            abs(entry.contribution)
            for entry in supporting + conflicting
        )

        self.assertAlmostEqual(total, 1.0, places=12)

    def test_an_identical_composition_produces_no_disagreement(self):
        supporting, conflicting = self.algorithm.evidence(
            prepared(SAMPLE_VALUES, dict(SAMPLE_VALUES))
        )

        self.assertEqual(conflicting, [])
        for entry in supporting:
            self.assertAlmostEqual(entry.contribution, 0.0, places=12)

    def test_the_imputed_flag_is_carried_through_from_the_mask(self):
        """Rule 3's imputed flag, from the preprocessing mask and nowhere else."""
        supporting, conflicting = self.algorithm.evidence(
            prepared(
                SAMPLE_VALUES,
                {"Cu": 1000.0, "Zn": 10.0, "Au": 1.0},
                imputed={"Cu"},
            )
        )
        by_element = {
            entry.element: entry
            for entry in supporting + conflicting
        }

        self.assertTrue(by_element["Cu"].imputed)
        self.assertFalse(by_element["Zn"].imputed)


class KnnConfidenceAndEnvelopeTests(SimpleTestCase):
    """The kNN part: judging a match by the company it keeps."""

    def references(self):
        # Three samples from one deposit and one from another, so the nearest
        # neighbours genuinely disagree about which deposit is winning.
        return [
            {"id": 1, "values": dict(SAMPLE_VALUES),
             "deposit_id": "WLN", "deposit_name": "Woodlawn", "deposit_class": "VHMS"},
            {"id": 2, "values": {"Cu": 110.0, "Zn": 11.0, "Au": 1.1},
             "deposit_id": "WLN", "deposit_name": "Woodlawn", "deposit_class": "VHMS"},
            {"id": 3, "values": {"Cu": 90.0, "Zn": 9.5, "Au": 0.95},
             "deposit_id": "WLN", "deposit_name": "Woodlawn", "deposit_class": "VHMS"},
            {"id": 4, "values": {"Cu": 1.0, "Zn": 10.0, "Au": 100.0},
             "deposit_id": "OTH", "deposit_name": "Other", "deposit_class": "Orogenic Au"},
        ]

    def envelope(self, config=None):
        return get_algorithm("knn_aitchison").compare(
            [{"sample_code": "S001", "values": SAMPLE_VALUES}],
            self.references(),
            config or {"top_n": 10},
        ).to_envelope()

    def test_the_envelope_satisfies_the_contract(self):
        self.assertEqual(validate_envelope(self.envelope()), [])

    def test_raw_metrics_travel_beside_the_normalised_similarity(self):
        scores = self.envelope()["matches"][0]["scores"]

        self.assertIn("similarity", scores)
        self.assertIn("aitchison_distance", scores)
        self.assertIn("spearman_rho", scores)

    def test_a_match_backed_by_its_neighbours_is_reported_as_confident(self):
        """
        Three of the four references come from Woodlawn and all sit close, so
        the top match should be well supported.
        """
        top = self.envelope()["matches"][0]

        self.assertEqual(top["deposit_id"], "WLN")
        self.assertEqual(top["confidence"]["n_reference_samples"], 3)
        self.assertAlmostEqual(top["confidence"]["consistency"], 0.75, places=12)
        self.assertEqual(top["confidence"]["level"], "high")

    def test_the_odd_deposit_out_is_reported_as_unconfident(self):
        matches = {
            match["deposit_id"]: match
            for match in self.envelope()["matches"]
        }

        self.assertEqual(matches["OTH"]["confidence"]["n_reference_samples"], 1)
        self.assertEqual(matches["OTH"]["confidence"]["level"], "low")

    def test_k_is_configurable_and_recorded_in_the_run(self):
        envelope = self.envelope({"top_n": 10, "k": 2})

        self.assertEqual(envelope["run"]["algorithm"]["params"]["k"], 2)
        # Only the two nearest vote now, and both are Woodlawn.
        self.assertAlmostEqual(
            envelope["matches"][0]["confidence"]["consistency"],
            1.0,
            places=12,
        )

    def test_k_defaults_to_the_documented_value(self):
        self.assertEqual(
            self.envelope()["run"]["algorithm"]["params"]["k"],
            DEFAULT_K,
        )

    def test_evidence_is_attached_to_the_leading_matches(self):
        matches = self.envelope()["matches"]
        evidence = matches[0]["evidence"]

        self.assertIn("supporting", evidence)
        self.assertIn("conflicting", evidence)
        for entry in evidence["supporting"] + evidence["conflicting"]:
            self.assertEqual(set(entry), {"element", "contribution", "imputed"})

    def test_detail_is_limited_to_the_requested_number_of_matches(self):
        """
        Evidence multiplies the size of a result set that already runs to
        hundreds of rows, so it is attached to the leading matches only.
        """
        envelope = self.envelope({"top_n": 10, "detail_top_n": 2})
        matches = envelope["matches"]

        self.assertEqual(len(matches), 4)
        self.assertIn("evidence", matches[0])
        self.assertIn("evidence", matches[1])
        self.assertNotIn("evidence", matches[2])
        self.assertNotIn("confidence", matches[2])

    def test_the_algorithm_declares_the_evidence_capability(self):
        """Consumers render optional blocks based on this declaration."""
        self.assertIn("evidence", get_algorithm("knn_aitchison").capabilities)

    def test_the_declared_capabilities_match_what_is_actually_produced(self):
        algorithm = get_algorithm("knn_aitchison")
        envelope = self.envelope()

        produces_evidence = any(
            "evidence" in match
            for match in envelope["matches"]
        )
        self.assertEqual(produces_evidence, "evidence" in algorithm.capabilities)
