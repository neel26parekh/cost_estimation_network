from laptop_price.ops import build_auth_headers, default_prediction_payload, run_preflight_checks


def test_build_auth_headers_omits_empty_key() -> None:
    assert build_auth_headers("") == {}
    assert build_auth_headers(None) == {}


def test_build_auth_headers_includes_api_key() -> None:
    assert build_auth_headers("secret") == {"x-api-key": "secret"}


def test_default_prediction_payload_matches_expected_contract() -> None:
    payload = default_prediction_payload()

    assert payload["company"] == "Dell"
    assert payload["screen_resolution"] == "1920x1080"
    assert payload["ssd"] == 256


def test_run_preflight_checks_returns_status_shape() -> None:
    checks = run_preflight_checks()

    assert "env_file_present" in checks
    assert "model_artifact_present" in checks
    assert "ready" in checks