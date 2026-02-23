from dataset_eval.grammar import validate_by_format


def test_json_ok():
    r = validate_by_format('{"a": 1}', 'json')
    assert r.ok


def test_json_fail():
    r = validate_by_format('{"a": 1', 'json')
    assert not r.ok


def test_csv_ok():
    r = validate_by_format('a,b\n1,2\n3,4\n', 'csv')
    assert r.ok


def test_csv_width_mismatch():
    r = validate_by_format('a,b\n1,2,3\n', 'csv')
    assert not r.ok
