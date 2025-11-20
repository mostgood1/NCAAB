"""Verification utility for Division I team membership list.

Compares the current data/d1_conferences.csv against an embedded authoritative
2025-26 Division I men's basketball team list (including recent transitions).
Reports:
  - Missing teams (in authoritative list, absent from file)
  - Extra entries (present in file but not authoritative; including conference header rows)
  - Conference mismatches (same canonical team but different conference)
  - Duplicate canonical team rows

Usage:
    python -m ncaab_model.data.verify_d1_membership [--csv-out missing_report.csv]

Exit code 0 even if discrepancies found (informational); use output to patch file.
"""
from __future__ import annotations
import sys
from pathlib import Path
import argparse
import pandas as pd
from typing import Dict, List, Tuple

try:
    from .team_normalize import canonical_slug as _canon
except Exception:  # pragma: no cover
    def _canon(x: str) -> str:  # fallback passthrough
        return (x or '').strip().lower().replace(' ', '')

# Authoritative list (team, conference). Transitional teams included; obsolete former D1 (e.g., Hartford) excluded.
# NOTE: This is a curated minimal set. For full automation, replace with external authoritative feed integration.
_AUTHORITATIVE: List[Tuple[str, str]] = [
    # ACC
    ("Boston College Eagles", "ACC"), ("Clemson Tigers", "ACC"), ("Duke Blue Devils", "ACC"),
    ("Florida State Seminoles", "ACC"), ("Georgia Tech Yellow Jackets", "ACC"), ("Louisville Cardinals", "ACC"),
    ("Miami Hurricanes", "ACC"), ("North Carolina Tar Heels", "ACC"), ("NC State Wolfpack", "ACC"),
    ("Notre Dame Fighting Irish", "ACC"), ("Pittsburgh Panthers", "ACC"), ("Syracuse Orange", "ACC"),
    ("Virginia Cavaliers", "ACC"), ("Virginia Tech Hokies", "ACC"), ("Wake Forest Demon Deacons", "ACC"),
    # Big Ten (w/ adds UCLA, USC, Oregon, Washington)
    ("Illinois Fighting Illini", "Big Ten"),("Indiana Hoosiers", "Big Ten"),("Iowa Hawkeyes", "Big Ten"),
    ("Maryland Terrapins", "Big Ten"),("Michigan Wolverines", "Big Ten"),("Michigan State Spartans", "Big Ten"),
    ("Minnesota Golden Gophers", "Big Ten"),("Nebraska Cornhuskers", "Big Ten"),("Northwestern Wildcats", "Big Ten"),
    ("Ohio State Buckeyes", "Big Ten"),("Penn State Nittany Lions", "Big Ten"),("Purdue Boilermakers", "Big Ten"),
    ("Rutgers Scarlet Knights", "Big Ten"),("Wisconsin Badgers", "Big Ten"),("UCLA Bruins", "Big Ten"),
    ("USC Trojans", "Big Ten"),("Oregon Ducks", "Big Ten"),("Washington Huskies", "Big Ten"),
    # Big 12 (post realignment adds)
    ("Arizona Wildcats", "Big 12"),("Arizona State Sun Devils", "Big 12"),("Baylor Bears", "Big 12"),
    ("BYU Cougars", "Big 12"),("Cincinnati Bearcats", "Big 12"),("Colorado Buffaloes", "Big 12"),("Houston Cougars", "Big 12"),
    ("Iowa State Cyclones", "Big 12"),("Kansas Jayhawks", "Big 12"),("Kansas State Wildcats", "Big 12"),("Oklahoma State Cowboys", "Big 12"),
    ("TCU Horned Frogs", "Big 12"),("Texas Tech Red Raiders", "Big 12"),("UCF Knights", "Big 12"),("Utah Utes", "Big 12"),("West Virginia Mountaineers", "Big 12"),
    # SEC (Texas, Oklahoma moved here)
    ("Alabama Crimson Tide", "SEC"),("Arkansas Razorbacks", "SEC"),("Auburn Tigers", "SEC"),("Florida Gators", "SEC"),("Georgia Bulldogs", "SEC"),
    ("Kentucky Wildcats", "SEC"),("LSU Tigers", "SEC"),("Ole Miss Rebels", "SEC"),("Mississippi State Bulldogs", "SEC"),("Missouri Tigers", "SEC"),
    ("South Carolina Gamecocks", "SEC"),("Tennessee Volunteers", "SEC"),("Texas A&M Aggies", "SEC"),("Vanderbilt Commodores", "SEC"),("Texas Longhorns", "SEC"),("Oklahoma Sooners", "SEC"),
    # Big East
    ("UConn Huskies", "Big East"),("Villanova Wildcats", "Big East"),("Marquette Golden Eagles", "Big East"),("St. John's Red Storm", "Big East"),
    ("Georgetown Hoyas", "Big East"),("Providence Friars", "Big East"),("Butler Bulldogs", "Big East"),("Xavier Musketeers", "Big East"),("Seton Hall Pirates", "Big East"),("DePaul Blue Demons", "Big East"),("Creighton Bluejays", "Big East"),
    # A-10
    ("Dayton Flyers", "A-10"),("VCU Rams", "A-10"),("Saint Louis Billikens", "A-10"),("St. Bonaventure Bonnies", "A-10"),("Davidson Wildcats", "A-10"),("Richmond Spiders", "A-10"),("Saint Joseph's Hawks", "A-10"),("George Mason Patriots", "A-10"),("Duquesne Dukes", "A-10"),("Loyola Chicago Ramblers", "A-10"),
    # AAC realigned
    ("Memphis Tigers", "AAC"),("Florida Atlantic Owls", "AAC"),("UAB Blazers", "AAC"),("North Texas Mean Green", "AAC"),("Wichita State Shockers", "AAC"),("Tulane Green Wave", "AAC"),("Temple Owls", "AAC"),("South Florida Bulls", "AAC"),("East Carolina Pirates", "AAC"),("Rice Owls", "AAC"),("Charlotte 49ers", "AAC"),("UTSA Roadrunners", "AAC"),("Tulsa Golden Hurricane", "AAC"),("SMU Mustangs", "AAC"),
    # Mountain West
    ("San Diego State Aztecs", "Mountain West"),("New Mexico Lobos", "Mountain West"),("UNLV Rebels", "Mountain West"),("Nevada Wolf Pack", "Mountain West"),("Utah State Aggies", "Mountain West"),("Colorado State Rams", "Mountain West"),("Boise State Broncos", "Mountain West"),("Fresno State Bulldogs", "Mountain West"),("Wyoming Cowboys", "Mountain West"),("Air Force Falcons", "Mountain West"),("San José State Spartans", "Mountain West"),
    # WCC
    ("Gonzaga Bulldogs", "WCC"),("Saint Mary's Gaels", "WCC"),("San Francisco Dons", "WCC"),("Santa Clara Broncos", "WCC"),("Loyola Marymount Lions", "WCC"),("Pepperdine Waves", "WCC"),("San Diego Toreros", "WCC"),("Portland Pilots", "WCC"),("Pacific Tigers", "WCC"),
    # MAAC
    ("Iona Gaels", "MAAC"),("Rider Broncs", "MAAC"),("Fairfield Stags", "MAAC"),("Quinnipiac Bobcats", "MAAC"),("Manhattan Jaspers", "MAAC"),("Marist Red Foxes", "MAAC"),("Niagara Purple Eagles", "MAAC"),("Canisius Golden Griffins", "MAAC"),("Siena Saints", "MAAC"),("Saint Peter's Peacocks", "MAAC"),("Mount St. Mary's Mountaineers", "MAAC"),
    # MAC
    ("Akron Zips", "MAC"),("Ball State Cardinals", "MAC"),("Bowling Green Falcons", "MAC"),("Buffalo Bulls", "MAC"),("Central Michigan Chippewas", "MAC"),("Eastern Michigan Eagles", "MAC"),("Kent State Golden Flashes", "MAC"),("Miami (OH) RedHawks", "MAC"),("Northern Illinois Huskies", "MAC"),("Ohio Bobcats", "MAC"),("Toledo Rockets", "MAC"),("Western Michigan Broncos", "MAC"),
    # Sun Belt
    ("App State Mountaineers", "Sun Belt"),("Arkansas State Red Wolves", "Sun Belt"),("Coastal Carolina Chanticleers", "Sun Belt"),("Georgia Southern Eagles", "Sun Belt"),("Georgia State Panthers", "Sun Belt"),("James Madison Dukes", "Sun Belt"),("Louisiana Ragin' Cajuns", "Sun Belt"),("Marshall Thundering Herd", "Sun Belt"),("Old Dominion Monarchs", "Sun Belt"),("South Alabama Jaguars", "Sun Belt"),("Texas State Bobcats", "Sun Belt"),("Troy Trojans", "Sun Belt"),("ULM Warhawks", "Sun Belt"),("Southern Miss Golden Eagles", "Sun Belt"),
    # Pac-12 remnants (note realignment; may shrink in authoritative list but keep for season continuity if schedule uses)
    ("California Golden Bears", "Pac-12"),("Oregon State Beavers", "Pac-12"),("Stanford Cardinal", "Pac-12"),("Washington State Cougars", "Pac-12"),
    # SWAC
    ("Alabama A&M Bulldogs", "SWAC"),("Alabama State Hornets", "SWAC"),("Alcorn State Braves", "SWAC"),("Bethune-Cookman Wildcats", "SWAC"),("Florida A&M Rattlers", "SWAC"),("Grambling Tigers", "SWAC"),("Jackson State Tigers", "SWAC"),("Mississippi Valley State Delta Devils", "SWAC"),("Prairie View A&M Panthers", "SWAC"),("Southern Jaguars", "SWAC"),("Texas Southern Tigers", "SWAC"),
    # MEAC
    ("Howard Bison", "MEAC"),("Norfolk State Spartans", "MEAC"),("North Carolina Central Eagles", "MEAC"),("Coppin State Eagles", "MEAC"),("Delaware State Hornets", "MEAC"),("Maryland Eastern Shore Hawks", "MEAC"),("South Carolina State Bulldogs", "MEAC"),("Morgan State Bears", "MEAC"),
    # NEC
    ("Le Moyne Dolphins", "NEC"),("Fairleigh Dickinson Knights", "NEC"),("LIU Sharks", "NEC"),("Merrimack Warriors", "NEC"),("Sacred Heart Pioneers", "NEC"),("Saint Francis (PA) Red Flash", "NEC"),("Stonehill Skyhawks", "NEC"),("Wagner Seahawks", "NEC"),("Central Connecticut Blue Devils", "NEC"),
    # Ivy
    ("Brown Bears", "Ivy"),("Columbia Lions", "Ivy"),("Cornell Big Red", "Ivy"),("Dartmouth Big Green", "Ivy"),("Harvard Crimson", "Ivy"),("Penn Quakers", "Ivy"),("Princeton Tigers", "Ivy"),("Yale Bulldogs", "Ivy"),
    # Patriot
    ("American University Eagles", "Patriot"),("Army Black Knights", "Patriot"),("Boston University Terriers", "Patriot"),("Bucknell Bison", "Patriot"),("Colgate Raiders", "Patriot"),("Holy Cross Crusaders", "Patriot"),("Lafayette Leopards", "Patriot"),("Lehigh Mountain Hawks", "Patriot"),("Loyola Maryland Greyhounds", "Patriot"),("Navy Midshipmen", "Patriot"),
    # ASUN (includes transitions)
    ("Florida Gulf Coast Eagles", "ASUN"),("Jacksonville Dolphins", "ASUN"),("Lipscomb Bisons", "ASUN"),("North Alabama Lions", "ASUN"),("Queens Royals", "ASUN"),("Stetson Hatters", "ASUN"),("Austin Peay Governors", "ASUN"),("Bellarmine Knights", "ASUN"),("Central Arkansas Bears", "ASUN"),("Eastern Kentucky Colonels", "ASUN"),
    ("North Florida Ospreys", "ASUN"),
    # WAC
    ("UT Arlington Mavericks", "WAC"),("Abilene Christian Wildcats", "WAC"),("California Baptist Lancers", "WAC"),("Grand Canyon Lopes", "WAC"),("Stephen F. Austin Lumberjacks", "WAC"),("Tarleton Texans", "WAC"),("Southern Utah Thunderbirds", "WAC"),("Seattle Redhawks", "WAC"),("UTRGV Vaqueros", "WAC"),("Utah Tech Trailblazers", "WAC"),
    # Horizon
    ("Milwaukee Panthers", "Horizon"),("Purdue Fort Wayne Mastodons", "Horizon"),("Detroit Mercy Titans", "Horizon"),("Green Bay Phoenix", "Horizon"),("Oakland Golden Grizzlies", "Horizon"),("Wright State Raiders", "Horizon"),("Cleveland State Vikings", "Horizon"),("Robert Morris Colonials", "Horizon"),("Youngstown State Penguins", "Horizon"),("Northern Kentucky Norse", "Horizon"),
    # Missouri Valley
    ("Evansville Purple Aces", "Missouri Valley"),("Belmont Bruins", "Missouri Valley"),("Bradley Braves", "Missouri Valley"),("Drake Bulldogs", "Missouri Valley"),("UIC Flames", "Missouri Valley"),("Illinois State Redbirds", "Missouri Valley"),("Indiana State Sycamores", "Missouri Valley"),("Missouri State Bears", "Missouri Valley"),("Murray State Racers", "Missouri Valley"),("Northern Iowa Panthers", "Missouri Valley"),("Southern Illinois Salukis", "Missouri Valley"),("Valparaiso Beacons", "Missouri Valley"),("Western Illinois Leathernecks", "Missouri Valley"),
    # SoCon
    ("Chattanooga Mocs", "SoCon"),("The Citadel Bulldogs", "SoCon"),("ETSU Buccaneers", "SoCon"),("Furman Paladins", "SoCon"),("Mercer Bears", "SoCon"),("Samford Bulldogs", "SoCon"),("UNC Greensboro Spartans", "SoCon"),("VMI Keydets", "SoCon"),("Western Carolina Catamounts", "SoCon"),("Wofford Terriers", "SoCon"),
    # CAA
    ("Campbell Fighting Camels", "CAA"),("Charleston Cougars", "CAA"),("Delaware Fightin' Blue Hens", "CAA"),("Drexel Dragons", "CAA"),("Elon Phoenix", "CAA"),("Hampton Pirates", "CAA"),("Hofstra Pride", "CAA"),("Monmouth Hawks", "CAA"),("Northeastern Huskies", "CAA"),("NC A&T Aggies", "CAA"),("Stony Brook Seawolves", "CAA"),("Towson Tigers", "CAA"),("UNC Wilmington Seahawks", "CAA"),("William & Mary Tribe", "CAA"),
    # America East
    ("Albany Great Danes", "America East"),("Binghamton Bearcats", "America East"),("Bryant Bulldogs", "America East"),("Maine Black Bears", "America East"),("UMass Lowell River Hawks", "America East"),("New Hampshire Wildcats", "America East"),("NJIT Highlanders", "America East"),("Vermont Catamounts", "America East"),
    # Big South
    ("Gardner-Webb Runnin' Bulldogs", "Big South"),("High Point Panthers", "Big South"),("Longwood Lancers", "Big South"),("Presbyterian Blue Hose", "Big South"),("Radford Highlanders", "Big South"),("UNC Asheville Bulldogs", "Big South"),("USC Upstate Spartans", "Big South"),("Winthrop Eagles", "Big South"),
    # Ohio Valley (adds Southern Indiana)
    ("Eastern Illinois Panthers", "Ohio Valley"),("Little Rock Trojans", "Ohio Valley"),("Lindenwood Lions", "Ohio Valley"),("Morehead State Eagles", "Ohio Valley"),("SIU Edwardsville Cougars", "Ohio Valley"),("Southeast Missouri State Redhawks", "Ohio Valley"),("Tennessee State Tigers", "Ohio Valley"),("Tennessee Tech Golden Eagles", "Ohio Valley"),("UT Martin Skyhawks", "Ohio Valley"),("Southern Indiana Screaming Eagles", "Ohio Valley"),
    # Big Sky
    ("Portland State Vikings", "Big Sky"),("Eastern Washington Eagles", "Big Sky"),("Idaho Vandals", "Big Sky"),("Idaho State Bengals", "Big Sky"),("Montana Grizzlies", "Big Sky"),("Montana State Bobcats", "Big Sky"),("Northern Arizona Lumberjacks", "Big Sky"),("Northern Colorado Bears", "Big Sky"),("Sacramento State Hornets", "Big Sky"),("Weber State Wildcats", "Big Sky"),
    # Big West
    ("Cal Poly Mustangs", "Big West"),("Cal State Bakersfield Roadrunners", "Big West"),("Cal State Fullerton Titans", "Big West"),("Cal State Northridge Matadors", "Big West"),("Hawai'i Rainbow Warriors", "Big West"),("Long Beach State Beach", "Big West"),("UC Davis Aggies", "Big West"),("UC Irvine Anteaters", "Big West"),("UC Riverside Highlanders", "Big West"),("UC San Diego Tritons", "Big West"),("UC Santa Barbara Gauchos", "Big West"),
    # Conference USA (realigned set)
    ("FIU Panthers", "Conference USA"),("Jacksonville State Gamecocks", "Conference USA"),("Kennesaw State Owls", "Conference USA"),("Liberty Flames", "Conference USA"),("Louisiana Tech Bulldogs", "Conference USA"),("Middle Tennessee Blue Raiders", "Conference USA"),("New Mexico State Aggies", "Conference USA"),("Sam Houston Bearkats", "Conference USA"),("UTEP Miners", "Conference USA"),("Western Kentucky Hilltoppers", "Conference USA"),
    # Southland (includes Incarnate Word)
    ("Incarnate Word Cardinals", "Southland"),("Houston Christian Huskies", "Southland"),("Lamar Cardinals", "Southland"),("McNeese Cowboys", "Southland"),("New Orleans Privateers", "Southland"),("Nicholls Colonels", "Southland"),("Northwestern State Demons", "Southland"),("Southeastern Louisiana Lions", "Southland"),("Texas A&M-Corpus Christi Islanders", "Southland"),
    # Summit League
    ("Denver Pioneers", "Summit League"),("Kansas City Roos", "Summit League"),("North Dakota Fighting Hawks", "Summit League"),("North Dakota State Bison", "Summit League"),("Omaha Mavericks", "Summit League"),("Oral Roberts Golden Eagles", "Summit League"),("South Dakota Coyotes", "Summit League"),("South Dakota State Jackrabbits", "Summit League"),("St. Thomas (MN) Tommies", "Summit League"),
    # Colonial/Other transitional duplicates intentionally omitted if already listed
]

# Build authoritative mapping
_AUTH_MAP: Dict[str, Tuple[str, str]] = {}
for team, conf in _AUTHORITATIVE:
    slug = _canon(team)
    _AUTH_MAP[slug] = (team, conf)


def load_current(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect either columns team, conference or first two columns
    lc = {c.lower(): c for c in df.columns}
    team_col = lc.get('team') or list(df.columns)[0]
    conf_col = lc.get('conference') or (list(df.columns)[1] if len(df.columns) > 1 else None)
    out = df[[team_col] + ([conf_col] if conf_col else [])].copy()
    out.columns = ['team'] + (['conference'] if conf_col else [])
    # Drop comment/header lines starting with '#'
    out = out[~out['team'].astype(str).str.strip().str.startswith('#')]
    return out


def verify(d1_df: pd.DataFrame) -> Dict[str, List[str]]:
    report: Dict[str, List[str]] = {
        'missing': [],
        'extra': [],
        'conflicts': [],
        'duplicates': [],
        'header_like': [],
    }
    seen: Dict[str, List[str]] = {}
    for _, r in d1_df.iterrows():
        team = str(r.get('team') or '').strip()
        conf = str(r.get('conference') or '').strip()
        if not team:
            continue
        slug = _canon(team)
        seen.setdefault(slug, []).append(conf)
        # Header-like row (team equals conference or appears to be conference label)
        if team.lower() == conf.lower() or team.lower() in {c.lower() for _, c in _AUTHORITATIVE}:
            report['header_like'].append(team)
        if slug not in _AUTH_MAP:
            report['extra'].append(team)
        else:
            _, auth_conf = _AUTH_MAP[slug]
            if auth_conf != conf:
                report['conflicts'].append(f"{team} -> file '{conf}' vs auth '{auth_conf}'")
    # Missing teams
    for slug, (team, conf) in _AUTH_MAP.items():
        if slug not in seen:
            report['missing'].append(f"{team} ({conf})")
    # Duplicates (same slug multiple rows with differing or same conferences)
    for slug, confs in seen.items():
        if len(confs) > 1:
            uniq = sorted(set(confs))
            if slug in _AUTH_MAP:
                report['duplicates'].append(f"{_AUTH_MAP[slug][0]} -> {uniq}")
            else:
                report['duplicates'].append(f"{slug} (non-auth) -> {uniq}")
    return report


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv-out', help='Optional path to write a machine-readable missing/extras summary')
    ap.add_argument('--data-dir', help='Override data directory (default: project data/)')
    args = ap.parse_args(argv)

    base = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parents[3] / 'data'
    d1_path = base / 'd1_conferences.csv'
    if not d1_path.exists():
        print(f"ERROR: d1_conferences.csv not found at {d1_path}", file=sys.stderr)
        return 1
    d1_df = load_current(d1_path)
    report = verify(d1_df)

    def _section(title: str, items: List[str]):
        print(f"\n== {title} ({len(items)}) ==")
        for it in sorted(items):
            print(f" - {it}")

    print("Division I Membership Verification")
    print(f"Source file: {d1_path}")
    print(f"Total rows (raw): {len(d1_df)}")
    _section('Missing (add these)', report['missing'])
    _section('Extra (remove or confirm)', report['extra'])
    _section('Conference mismatches', report['conflicts'])
    _section('Duplicate canonical teams', report['duplicates'])
    _section('Header-like / placeholder rows', report['header_like'])

    if args.csv_out:
        rows = []
        for kind, items in report.items():
            for it in items:
                rows.append({'category': kind, 'value': it})
        out_df = pd.DataFrame(rows)
        out_df.to_csv(args.csv_out, index=False)
        print(f"\nWrote summary CSV to {args.csv_out}")

    # Non-zero exit could be used if strict, but keep 0 for now
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
