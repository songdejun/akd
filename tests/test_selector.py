from akd.selector import Decision, DecisionCache, pick_winner


def test_pick_winner_first_within_margin():
    # 1.02 is within 3% of 1.0: declared order wins (simpler variant preference)
    assert pick_winner([(0, 1.0), (1, 1.02), (2, 2.0)], 0.03) == 0


def test_pick_winner_clear_fastest():
    assert pick_winner([(0, 1.05), (1, 1.0)], 0.03) == 1


def test_pick_winner_margin_excluded():
    # 1.04 is outside the 3% margin around 1.0
    assert pick_winner([(0, 1.04), (1, 1.0)], 0.03) == 1


def test_decision_cache_put_get():
    cache = DecisionCache()
    d = Decision(winner=1, candidates=("a", "b"))
    cache.put("k1", d)
    assert cache.get("k1") is d
    assert len(cache) == 1


def test_decision_cache_reset_by_site_prefix():
    cache = DecisionCache()
    cache.put(("wrapper_a@f.py:0", "id", "dev", "args"), Decision(winner=0))
    cache.put(("wrapper_b@f.py:8", "id", "dev", "args"), Decision(winner=0))
    cache.reset("wrapper_a")
    assert len(cache) == 1
    assert cache.get(("wrapper_b@f.py:8", "id", "dev", "args")) is not None


def test_decision_cache_clear():
    cache = DecisionCache()
    cache.put(("a",), Decision(winner=0))
    cache.reset()
    assert len(cache) == 0
