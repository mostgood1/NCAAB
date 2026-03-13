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


def test_alias_app_state_appalachian_state():
    assert canonical_slug("App State Mountaineers") == canonical_slug("Appalachian St Mountaineers")


def test_mount_st_marys_is_saint_not_state():
    assert canonical_slug("Mount St. Mary's Mountaineers") == canonical_slug("Mount St Marys")


def test_alias_loyola_chi_abbrev():
    assert canonical_slug("Loyola Chi") == canonical_slug("Loyola Chicago")


def test_alias_prairie_view_aandm_variant():
    # Odds feeds sometimes omit the A&M qualifier for Prairie View A&M.
    assert canonical_slug("Prairie View Panthers") == canonical_slug("Prairie View A&M Panthers")


def test_alias_florida_international_intl_variant():
    # Provider may use Intl / Int'l and include legacy 'Golden Panthers' nickname.
    assert canonical_slug("Florida Int'l Golden Panthers") == canonical_slug("Florida International Panthers")
    assert canonical_slug("Florida Intl Panthers") == canonical_slug("Florida International Panthers")
    assert canonical_slug("FIU") == canonical_slug("Florida International")


def test_alias_texas_am_cc_variant():
    # Provider abbreviates Texas A&M-Corpus Christi as Texas A&M-CC.
    assert canonical_slug("Texas A&M-CC Islanders") == canonical_slug("Texas A&M-Corpus Christi Islanders")


def test_alias_grambling_state_variant():
    # Some schedules list just 'Grambling'; provider often uses Grambling State/Grambling St.
    assert canonical_slug("Grambling Tigers") == canonical_slug("Grambling St Tigers")


def test_alias_george_washington_revolutionaries_variant():
    assert canonical_slug("George Washington Revolutionaries") == canonical_slug("GW Revolutionaries")


def test_alias_csu_northridge_variant():
    assert canonical_slug("Cal State Northridge Matadors") == canonical_slug("CSU Northridge Matadors")


def test_alias_sam_houston_state_variant():
    assert canonical_slug("Sam Houston Bearkats") == canonical_slug("Sam Houston St Bearkats")
