# Tests for the toroidal patch to the vendored CGOS GoGame.
#
# Run from the repo root:
#   python3 -m unittest discover -s cgos/tests

import os
import sys
from unittest import TestCase

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server", "cgos"))

from gogame.go import GoGame, KoRule, Rule  # noqa: E402


def game():
    return GoGame(9, Rule(KoRule.SIMPLE))


class TestTorusAdjacency(TestCase):
    def test_every_point_has_four_wrapped_neighbours(self):
        g = game()
        self.assertEqual(len(g.nbrs), 81)
        for ix, ns in g.nbrs.items():
            self.assertEqual(len(ns), 4, f"point {ix}")
            for p in ns:
                self.assertNotEqual(g.bd[p], 3, f"sentinel neighbour at {ix}")

    def test_adjacency_is_symmetric(self):
        g = game()
        for ix, ns in g.nbrs.items():
            for p in ns:
                self.assertIn(ix, g.nbrs[p])


class TestTorusRules(TestCase):
    def test_corner_stone_has_wrapped_liberties(self):
        # On a flat board B(b1) + B(a2) captures W(a1).  On a torus a1 keeps
        # liberties at j1 and a9.
        g = game()
        self.assertEqual(g.make("b1"), 0)   # B
        self.assertEqual(g.make("a1"), 0)   # W
        self.assertEqual(g.make("a2"), 0)   # B — no capture on a torus
        self.assertEqual(g.bd[g.mvToIndex("a1")] != 0, True)

    def test_capture_across_the_edge(self):
        g = game()
        for mv in ["b1", "a1", "a2", "pass", "j1", "pass"]:
            self.assertGreaterEqual(g.make(mv), 0, mv)
        self.assertEqual(g.make("a9"), 1)   # fourth wrapped liberty: capture

    def test_suicide_across_the_edge(self):
        g = game()
        for mv in ["b1", "pass", "j1", "pass", "a2", "pass", "a9"]:
            self.assertGreaterEqual(g.make(mv), 0, mv)
        self.assertEqual(g.make("a1"), -1)  # white fills its own last liberty

    def test_simple_ko_still_applies(self):
        # Build a ko in the middle of the board; the immediate recapture that
        # recreates the previous position must be rejected under SIMPLE ko.
        g = game()
        for mv in ["d5", "f6", "e6", "g5", "e4", "f4", "pass"]:
            self.assertGreaterEqual(g.make(mv), 0, mv)
        self.assertEqual(g.make("e5"), 0)    # white plays into the ko mouth
        self.assertEqual(g.make("f5"), 1)    # black captures e5 (ko)
        self.assertEqual(g.make("e5"), -2)   # immediate recapture: ko violation

    def test_tromp_taylor_score_whole_board(self):
        g = game()
        g.make("e5")
        self.assertEqual(g.ttScore(), 81)    # lone black stone owns the torus
