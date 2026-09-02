from monitoring.feedback import load_feedback


def test_feedback_loader():
    records = load_feedback()

    assert isinstance(
        records,
        list,
    )
