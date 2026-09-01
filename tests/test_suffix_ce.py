from sediment.trainer import _suffix_start


def test_suffix_starts_at_first_corrected_token():
    assert _suffix_start([7, 8, 9, 10, 11], [7, 8, 6, 12]) == 2


def test_suffix_accepts_chosen_extension_after_shared_prefix():
    assert _suffix_start([7, 8, 9], [7, 8]) == 2


def test_suffix_rejects_identical_or_shorter_chosen_continuation():
    assert _suffix_start([7, 8], [7, 8]) is None
    assert _suffix_start([7, 8], [7, 8, 9]) is None
