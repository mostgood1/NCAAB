from ncaab_model.data.team_normalize import canonical_slug

def test_alias_basic():
    assert canonical_slug("Saint Peter's Peacocks") == canonical_slug("St Peters Peacocks")


def test_alias_albany():
    # UAlbany vs Albany variant should collapse
    assert canonical_slug("UAlbany Great Danes") == canonical_slug("Albany Great Danes")


def test_alias_norfolk_state():
    assert canonical_slug("Norfolk St Spartans") == canonical_slug("Norfolk State Spartans")


def test_alias_san_jose_state():
    assert canonical_slug("San Jose St Spartans") == canonical_slug("San Jose State Spartans")


def test_alias_east_tennessee_state():
    assert canonical_slug("East Tennessee St Buccaneers") == canonical_slug("East Tennessee State Buccaneers")
