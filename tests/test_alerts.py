from laptop_price.alerts import evaluate_drift_alert


def test_evaluate_drift_alert_ignores_small_samples() -> None:
    evaluation = evaluate_drift_alert({"sample_size": 3, "drift_detected": True})

    assert evaluation["should_alert"] is False
    assert evaluation["reason"] == "insufficient_sample_size"


def test_evaluate_drift_alert_fails_when_drift_detected_with_enough_samples() -> None:
    evaluation = evaluate_drift_alert({"sample_size": 50, "drift_detected": True})

    assert evaluation["should_alert"] is True
    assert evaluation["reason"] == "drift_detected"