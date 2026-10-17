from load_test.run_scenario import API_REQUEST_CAP, request_cap


def test_api_backend_is_capped_by_default():
    assert request_cap("api", 0) == API_REQUEST_CAP == 40


def test_explicit_cap_is_kept():
    assert request_cap("api", 10) == 10


def test_local_and_fallback_are_not_capped():
    assert request_cap("local", 0) == 0 and request_cap("fallback", 0) == 0
