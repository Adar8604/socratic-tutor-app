from conftest import SAMPLE_ROW

from socratic_tutor.data import (build_eval_items, build_sft_examples, dataset_stats, parse_conversation,
                                 row_to_dialogue)


def test_parse_conversation_moves_and_roles():
    turns = parse_conversation(SAMPLE_ROW["conversation"])
    assert [t.role for t in turns] == ["teacher", "student"] * 3 + ["teacher"]
    assert [t.move for t in turns if t.role == "teacher"] == ["generic", "focus", "telling", "generic"]
    assert not turns[0].text.startswith("(")


def test_socratic_filter_drops_telling_and_leaks():
    d = row_to_dialogue(SAMPLE_ROW)
    assert d.answer == 10
    raw = build_sft_examples([d], "raw")
    soc = build_sft_examples([d], "socratic")
    assert len(raw) == 4
    assert all(e["move"] != "telling" for e in soc)
    assert len(soc) == 3
    for e in soc:  # strict alternation, system first, completion is assistant
        roles = [m["role"] for m in e["prompt"]]
        assert roles[0] == "system" and roles[1] == "user"
        assert all(a != b for a, b in zip(roles[1:], roles[2:]))
        assert roles[-1] == "user"
        assert e["completion"][0]["role"] == "assistant"


def test_eval_items_only_before_student_solved():
    d = row_to_dialogue(SAMPLE_ROW)
    items = build_eval_items([d] * 5, n_max=10, seed=0)
    assert items and all(it["turn_index"] <= 4 for it in items)
    assert all(it["messages"][-1]["role"] == "user" for it in items)


def test_stats():
    s = dataset_stats([row_to_dialogue(SAMPLE_ROW)])
    assert s["teacher_turns"] == 4 and abs(s["gold_turn_answer_leak_rate"] - 0.25) < 1e-9
