from socratic_tutor.answer_utils import (extract_numbers, extract_predicted_answer, final_answer_from_solution,
                                         is_leak)


def test_extract_numbers_formats():
    assert extract_numbers("It costs $1,250.50 and 40% of 3/4 is 0.3") == [1250.5, 40.0, 0.75, 0.3]
    assert extract_numbers("the 20th step, x2 and 5-year plan") == [5.0]
    assert extract_numbers("Total is 15.") == [15.0]


def test_final_answer():
    assert final_answer_from_solution("a = 3\nb = 4\n#### 1,200") == 1200
    assert final_answer_from_solution("so the package had 15 - 5 = 10 spoons.\n 10") == 10


def test_predicted_answer_priority():
    assert extract_predicted_answer("I get 7 then 9. FINAL ANSWER: 12") == 12
    assert extract_predicted_answer("... The answer is $42.") == 42
    assert extract_predicted_answer("first 3 then 8") == 8


def test_leak_rules():
    assert is_leak("So the package had 10 spoons.", 10, ["I think 4"], "problem with 5 and 12") is True
    assert is_leak("What happens if you add 3?", 10, ["I think 4"], "problem") is False
    # student already said it -> confirming is not a leak
    assert is_leak("Yes, 10 is right!", 10, ["is it 10?"], "problem") is False
    # answer appears in the problem statement -> undefined
    assert is_leak("10", 10, [], "There are 10 apples") is None
