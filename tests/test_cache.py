from bl_ranking.serving.cache import ResponseCache


def test_cache_key_changes_with_version():
    c = ResponseCache(ttl_s=60, max_items=10)
    assert c.key({"a": 1}, "7") != c.key({"a": 1}, "8")


def test_cache_key_ignores_dict_order():
    c = ResponseCache(ttl_s=60, max_items=10)
    assert c.key({"a": 1, "b": 2}, "7") == c.key({"b": 2, "a": 1}, "7")


def test_cache_put_get_and_ttl():
    c = ResponseCache(ttl_s=0.05, max_items=10)
    k = c.key({"a": 1}, "7")
    c.put(k, {"x": {"rank": 1.0}})
    assert c.get(k) == {"x": {"rank": 1.0}}
    import time

    time.sleep(0.1)
    assert c.get(k) is None


def test_cache_disabled_with_zero_items():
    c = ResponseCache(ttl_s=60, max_items=0)
    k = c.key({"a": 1}, "7")
    c.put(k, {"x": {}})
    assert c.get(k) is None


def test_request_hash_is_keyed_not_plain_sha256():
    """The hash covers name + phone: a plain SHA-256 could be matched by hashing guessed inputs."""
    import hashlib
    import json

    payload = {"fname": "Alex", "cellphone": 7135550100}
    a, b = ResponseCache(ttl_s=60, max_items=10), ResponseCache(ttl_s=60, max_items=10)
    plain = hashlib.sha256(f"7|{json.dumps(payload, sort_keys=True, separators=(',', ':'))}".encode()).hexdigest()
    assert a.key(payload, "7") != plain
    assert a.key(payload, "7") != b.key(payload, "7")  # per-process key
    assert a.key(payload, "7") == a.key(dict(payload), "7")  # still a stable cache key
