def question_bank(index):
    questions = [
                "open mouth",
                "blink eyes",
                "turn face up",
                "turn face down",
                "turn face right",
                "turn face left"]
    return questions[index]

def challenge_result(question, det):
    if question == "turn face right":
        if det.right_total == 0:
            challenge = "fail"
        else:
            challenge = "pass"

    elif question == "turn face left":
        if det.left_total == 0:
            challenge = "fail"
        else:
            challenge = "pass"

    elif question == "turn face up":
        if det.up_total == 0:
            challenge = "fail"
        else:
            challenge = "pass"

    elif question == "turn face down":
        if det.down_total == 0:
            challenge = "fail"
        else:
            challenge = "pass"

    elif question == "blink eyes":
        if det.eye_total == 0:
            challenge = "fail"
        else:
            challenge = "pass"

    elif question == "open mouth":
        if det.mouth_total == 0:
            challenge = "fail"
        else:
            challenge = "pass"

    return challenge