from ai1f_orchestrator import choose_action

def test_safety_precedes_question():
    assert choose_action(repair=None, next_question={"id":"x"}, risk_level="urgent") == "triage_interrupt"

def test_repair_precedes_question():
    assert choose_action(repair={"action":"repeat_question"}, next_question={"id":"x"}) == "repeat_question"

def test_question_action():
    assert choose_action(repair={"action":"accept_answer"}, next_question={"id":"x"}) == "ask_question"

def test_complete_action():
    assert choose_action(repair={"action":"accept_answer"}, next_question=None) == "complete_interview"
