from src.core.agent.state import _add_or_reset_int, _add_or_reset_list


def test_add_or_reset_list_appends():
    assert _add_or_reset_list([{"a": 1}], [{"b": 2}]) == [{"a": 1}, {"b": 2}]


def test_add_or_reset_list_resets_on_none():
    # None is the sentinel a node uses to clear the parallel fan-in buffer.
    assert _add_or_reset_list([{"a": 1}], None) == []


def test_add_or_reset_int_accumulates():
    assert _add_or_reset_int(3, 4) == 7


def test_add_or_reset_int_resets_on_none():
    assert _add_or_reset_int(3, None) == 0
