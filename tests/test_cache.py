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
