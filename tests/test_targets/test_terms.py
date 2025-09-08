import pytest

from batdetect2.targets import terms


def test_tag_info_and_get_tag_from_info():
    tag_info = TagInfo(value="Myotis myotis", key="event")
    tag = terms.get_tag_from_info(tag_info)
    assert tag.value == "Myotis myotis"
    assert tag.term == terms.call_type


def test_get_tag_from_info_key_not_found():
    tag_info = TagInfo(value="test", key="non_existent_key")
    with pytest.raises(KeyError):
        terms.get_tag_from_info(tag_info)
