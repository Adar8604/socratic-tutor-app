import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# A realistic MathDial-format row (format mirrors eth-nlped/mathdial).
SAMPLE_ROW = {
    "qid": 6000025,
    "scenario": 1,
    "question": "Julia bought a package of spoons. Her husband also bought 5 spoons. She used 3 spoons to taste stew and now has 12 spoons. How many spoons were in the package?",
    "ground_truth": "Julia had 12 + 3 = 15 spoons before tasting.\nSo the package had 15 - 5 = 10 spoons.\n 10",
    "student_incorrect_solution": "12 - 3 = 9 spoons, and 9 - 5 = 4 spoons. The answer is 4.",
    "student_profile": "Mia is a 7th grade student who sometimes mixes up operations.",
    "teacher_described_confusion": "subtracted instead of adding back",
    "self-correctness": "Yes",
    "self-typical-confusion": 4,
    "self-typical-interactions": 4,
    "conversation": (
        "Teacher: (generic)Hi Mia, can you walk me through your solution?|EOM|"
        "Mia: I subtracted the 3 spoons she used from 12, then subtracted the 5 from her husband.|EOM|"
        "Teacher: (focus)If she used 3 spoons and has 12 left, did she have more or fewer before tasting?|EOM|"
        "Mia: More, so 12 + 3 = 15.|EOM|"
        "Teacher: (telling)Right, and 15 - 5 = 10, so the package had 10 spoons.|EOM|"
        "Mia: Oh, so the answer is 10.|EOM|"
        "Teacher: (generic)Great job!"
    ),
}
