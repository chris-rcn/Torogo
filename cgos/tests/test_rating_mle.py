# Tests for the anchored Bradley-Terry fit in app/rating.py (the python
# twin of elo-lib.js fitRatings, used for the server's periodic refit).
#
# Run from the repo root:
#   python3 -m unittest discover -s cgos/tests

import os
import sys
from unittest import TestCase

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server", "cgos"))

from app.rating import expectation, fitRatings  # noqa: E402


class TestFitRatings(TestCase):
    def test_recovers_true_ratings_from_exact_scores(self):
        truth = {"A": 1500.0, "B": 1700.0, "C": 1900.0}
        recs = []
        names = sorted(truth)
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                p = expectation(truth[x], truth[y])
                recs.append((x, y, 200 * p, 200 * (1 - p)))
        fit = fitRatings(recs, {"B": 1700.0}, regularize=0.0)
        self.assertAlmostEqual(fit["A"], 1500.0, delta=0.5)
        self.assertEqual(fit["B"], 1700.0)  # anchor untouched
        self.assertAlmostEqual(fit["C"], 1900.0, delta=0.5)

    def test_regularization_keeps_undefeated_finite(self):
        fit = fitRatings([("winner", "anchor", 10.0, 0.0)], {"anchor": 1000.0})
        self.assertTrue(1300.0 < fit["winner"] < 3000.0)

    def test_draws_count_half(self):
        # all draws -> equal ratings
        fit = fitRatings([("A", "B", 5.0, 5.0)], {"A": 1600.0}, regularize=0.0)
        self.assertAlmostEqual(fit["B"], 1600.0, delta=0.1)

    def test_anchorless_centres_on_default(self):
        fit = fitRatings([("A", "B", 60.0, 40.0)], {}, defaultRating=1600.0)
        self.assertAlmostEqual((fit["A"] + fit["B"]) / 2, 1600.0, delta=1e-6)
        self.assertGreater(fit["A"], fit["B"])

    def test_matches_elo_lib_js(self):
        # Same fixture as elo-lib.test.js: results must agree across the
        # two implementations (standings.js display vs server write-back).
        recs = [
            ("A", "B", 30.0, 70.0),
            ("B", "C", 40.0, 60.0),
            ("A", "C", 10.0, 90.0),
        ]
        fit = fitRatings(recs, {"B": 1700.0})
        import json
        import subprocess
        root = os.path.join(os.path.dirname(__file__), "..", "..")
        js = (
            "const {fitRatings} = require('./elo-lib.js');"
            "const fit = fitRatings("
            "[{a:'A',b:'B',wa:30,wb:70},{a:'B',b:'C',wa:40,wb:60},"
            "{a:'A',b:'C',wa:10,wb:90}],"
            "{anchors: new Map([['B',1700]])});"
            "console.log(JSON.stringify(Object.fromEntries(fit)))"
        )
        out = subprocess.run(
            ["node", "-e", js], cwd=root, capture_output=True, text=True
        )
        jsfit = json.loads(out.stdout.strip().splitlines()[-1])
        for name in fit:
            self.assertAlmostEqual(fit[name], jsfit[name], delta=0.1, msg=name)
