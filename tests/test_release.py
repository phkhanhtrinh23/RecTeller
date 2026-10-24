"""Running example (Examples 1-4) and release-mode invariants."""
import numpy as np

import _path  # noqa: F401
from recteller.config import Protocol
from recteller.release import release


def test_running_example():
    proto = Protocol()
    assert 2 * proto.kp * proto.Delta / proto.eps_sel == 2.0      # b_sel
    assert proto.kp * proto.Delta / proto.eps_w == 3.0            # b_w
    cands = [1, 2, 3, 4, 5]
    s = np.array([2.0, 1.2, 0.8, -0.5, 1.5])
    public_order = [1, 2, 3, 5, 4]
    r = release(s, cands, public_order, proto, np.random.default_rng(0), kind="noisy",
                gumbel=np.array([0.4, 1.9, -0.3, 0.2, -0.6]), laplace=np.array([0.9, -0.4, -2.1]))
    assert r.items[:3] == [2, 1, 5]             # S = (i2, i1, i5)
    assert r.levels[:3] == [3, 2, 1]            # levels (3, 2, 1)
    assert r.items[3:] == [3, 4] and r.levels[3:] == [0, 0]   # fill i3, i4 at level 0
    ex = release(s, cands, public_order, proto, None, kind="exact_kp")
    assert ex.items[:3] == [1, 5, 2]            # noise-free (i1, i5, i2)


def test_modes_never_aliased():
    proto = Protocol()
    rng = np.random.default_rng(0)
    s = rng.normal(size=20) * 3
    cands = list(range(1, 21))
    a = release(s, cands, cands, proto.with_eps(float("inf")), rng, kind="noisy")
    b = release(s, cands, cands, proto, rng, kind="exact_kp")
    c = release(s, cands, cands, proto, rng, kind="topK_list")
    assert a.items == b.items and a.n_pers == 3          # eps = inf maps to exact_kp
    assert b.items != c.items and c.n_pers == 10         # never to the exact top-K list
    assert c.scores is None and b.scores is None


def test_clipping():
    from recteller.release import clip
    assert list(clip(np.array([100.0, -100.0, 0.0]), Protocol().clip)) == [5.0, -5.0, 0.0]


def test_random_replacement_is_per_item():
    """Random replacement replaces each item of the exact list independently with probability p."""
    proto = Protocol()
    rng = np.random.default_rng(0)
    s = np.linspace(3, -3, 20)
    cands = list(range(1, 21))
    n = [sum(i not in cands[:10] for i in release(s, cands, cands, proto, rng, kind="random_replace",
                                                   p_replace=0.3).items) for _ in range(2000)]
    assert abs(np.mean(n) - 3.0) < 0.15 and np.var(n) > 1.0   # Binomial(10, 0.3), not a fixed share


def test_empirical_dp_selection():
    """Monte-Carlo check of Prop. 1 for selection: for neighbours ||s-s'||_inf <= Delta, the
    probability of every ordered top-k' tuple changes by at most e^{eps_sel} (up to MC error)."""
    proto = Protocol(eps_w=float("inf"))
    rng = np.random.default_rng(1)
    s = np.array([1.0, 0.5, 0.0, -0.5, 0.2])
    s2 = s + np.array([-1, 1, 1, -1, 1]) * proto.Delta        # worst-case neighbour
    n = 200_000
    from collections import Counter
    from recteller.release import gumbel_topk
    b = 2 * proto.kp * proto.Delta / proto.eps_sel
    c1 = Counter(tuple(gumbel_topk(s, 3, b, rng)) for _ in range(n))
    c2 = Counter(tuple(gumbel_topk(s2, 3, b, rng)) for _ in range(n))
    for o in set(c1) | set(c2):
        p1, p2 = c1[o] / n, c2[o] / n
        if min(p1, p2) > 2e-3:                     # ignore MC-noisy rare tuples
            assert abs(np.log(p1 / p2)) <= proto.eps_sel + 0.15


def test_no_levels_spends_budget_on_selection():
    """eps_w = 0 omits the levels: no score magnitude is released, every selected slot has one level."""
    from dataclasses import replace
    proto = replace(Protocol(), eps_sel=4.0, eps_w=0.0)
    rng = np.random.default_rng(0)
    s = np.array([5.0, -5.0, 4.0, -4.0, 0.0, 3.0])
    for _ in range(50):
        r = release(s, list(range(1, 7)), list(range(1, 7)), proto, rng, kind="noisy")
        assert r.levels[: r.n_pers] == [1] * r.n_pers and set(r.levels[r.n_pers:]) <= {0}
