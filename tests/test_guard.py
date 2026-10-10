import socratic_tutor.guard as guard


def test_guard_resamples_only_leaking_turns(monkeypatch):
    calls = []

    def fake_generate(model, tok, convs, **kw):
        calls.append(len(convs))
        return ["What should you add back first?"] * len(convs)

    monkeypatch.setattr(guard, "chat_generate", fake_generate)
    responses = ["The package had 10 spoons.", "Did she have more or fewer spoons before?"]
    out, n = guard.guard_resample(None, None, [[], []], responses, solved=[10.0, 10.0],
                                  student_texts=[["4"], ["4"]], problems=["p", "p"],
                                  gcfg={"max_new_tokens": 8, "temperature": 0.7, "top_p": 0.9, "batch_size": 4})
    assert n == [1, 0] and calls == [1]
    assert "10" not in out[0] and out[1] == responses[1]


def test_guard_without_solved_answer_is_noop(monkeypatch):
    monkeypatch.setattr(guard, "chat_generate", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    out, n = guard.guard_resample(None, None, [[]], ["10"], [None], [["4"]], ["p"],
                                  {"max_new_tokens": 8, "temperature": 0.7, "top_p": 0.9, "batch_size": 4})
    assert n == [0]
