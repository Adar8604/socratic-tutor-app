from socratic_tutor.judge import combine_orders, parse_pairwise, parse_rubric
from socratic_tutor.stats import bootstrap_ci, cohen_kappa, paired_bootstrap_pvalue, rouge_l


def test_bootstrap_and_pvalue():
    m, lo, hi = bootstrap_ci([0, 1] * 50)
    assert abs(m - 0.5) < 1e-9 and lo < 0.5 < hi
    assert paired_bootstrap_pvalue([1] * 40, [0] * 40) < 0.01
    assert paired_bootstrap_pvalue([0, 1] * 20, [1, 0] * 20) > 0.5


def test_kappa_and_rouge():
    assert cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert abs(cohen_kappa(list("abab"), list("aabb"))) < 1e-9
    assert rouge_l("what is 3 plus 4", "what is 3 plus 4") == 1.0
    assert rouge_l("a b c", "x y z") == 0.0


def test_parse_rubric_variants():
    r = parse_rubric('Sure! {"move": "Focus", "reveals_answer": false, "pedagogy": 4, "correctness": 5, "error_targeting": 3}')
    assert r["parse_ok"] and r["move"] == "focus" and r["pedagogy"] == 4 and r["reveals_answer"] is False
    r = parse_rubric("{'move': 'telling', 'reveals_answer': True, 'pedagogy': 1, 'correctness': 5, 'error_targeting': 2}")
    assert r["parse_ok"] and r["reveals_answer"] is True
    r = parse_rubric('"move": "probing", "pedagogy": 4, "correctness": 4, "error_targeting": 4')
    assert r["parse_ok"]
    assert not parse_rubric("garbage")["parse_ok"]


def test_pairwise():
    assert parse_pairwise("A") == "A" and parse_pairwise('"B"') == "B" and parse_pairwise("Tie") == "tie"
    assert combine_orders("A", "B") == "win"
    assert combine_orders("B", "A") == "loss"
    assert combine_orders("A", "A") == "tie"  # position-biased judge -> tie
