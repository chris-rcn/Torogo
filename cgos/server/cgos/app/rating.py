# The MIT License
#
# Copyright (C) 2009 Don Dailey and Jason House
# Copyright (c) 2022 Kensuke Matsuzaki
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


# routines to rate the games
# --------------------------
def expectation(me: float, you: float) -> float:
    x = (you - me) / 400.0
    d: float = 1.0 + pow(10.0, x)
    return 1.0 / d


def newrating(cur_rating: float, opp_rating: float, res: float, K: float) -> float:
    ex = expectation(cur_rating, opp_rating)
    nr = cur_rating + K * (res - ex)
    return nr


# Torogo: anchored Bradley-Terry maximum-likelihood fit (the python twin
# of elo-lib.js fitRatings).  The server refits every cfg.mleInterval
# finalized games and writes the result back, so matchmaking and the
# stored ratings use every game optimally instead of the path-limited
# incremental walk above.
#
# records:    iterable of (white, black, whiteWins, blackWins) — one entry
#             per game (fractional counts allowed; a draw is 0.5/0.5).
# anchors:    dict name -> rating, held fixed (defines the scale).
# regularize: virtual draws added per played pair (default 1) so an
#             undefeated engine still gets a finite rating.
# Returns dict name -> fitted rating for every player seen.
def fitRatings(records, anchors, regularize: float = 1.0,
               defaultRating: float = 1600.0) -> dict:
    import math
    C = math.log(10.0) / 400.0

    # aggregate per unordered pair
    pairs: dict = {}
    for w, b, ww, bw in records:
        key = (w, b) if w < b else (b, w)
        p = pairs.setdefault(key, [0.0, 0.0])
        if key[0] == w:
            p[0] += ww
            p[1] += bw
        else:
            p[0] += bw
            p[1] += ww

    players = set(anchors)
    for a, b in pairs:
        players.add(a)
        players.add(b)

    byp: dict = {p: [] for p in players}
    for (a, b), (wa, wb) in pairs.items():
        wa += regularize / 2.0
        wb += regularize / 2.0
        n = wa + wb
        byp[a].append((b, wa, n))
        byp[b].append((a, wb, n))

    base = sum(anchors.values()) / len(anchors) if anchors else defaultRating
    r = {p: anchors.get(p, base) for p in players}

    # coordinate-wise Newton ascent on the concave log-likelihood
    for _ in range(2000):
        maxd = 0.0
        for p in players:
            if p in anchors:
                continue
            g = 0.0
            h = 0.0
            for opp, w, n in byp[p]:
                e = expectation(r[p], r[opp])
                g += w - n * e
                h += n * e * (1.0 - e)
            if h < 1e-12:
                continue
            d = max(-350.0, min(350.0, g / (C * h)))
            r[p] += d
            maxd = max(maxd, abs(d))
        if maxd < 1e-4:
            break

    # no anchors: pin the scale by centring on defaultRating
    if not anchors and r:
        m = sum(r.values()) / len(r)
        for p in r:
            r[p] += defaultRating - m

    return r


# produce a printable rating given rating and K
# ---------------------------------------------
def strRate(elo: float, k: float) -> str:

    r = "%0.0f" % elo

    if elo < 0.0:
        r = "0"
    if k > 16.0:
        r += "?"

    return r
