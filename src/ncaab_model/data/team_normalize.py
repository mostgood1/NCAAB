"""Team name normalization utilities.

Enhancements added:
 - generate_alias_candidates: rule-driven expansion of common NCAA naming variants
 - Support for directional abbreviations, saint/st variants, A&M forms, removal of leading 'the'
 - Token cleanup for hyphen/dash and punctuation variations

Existing functions (canonical_slug, normalize_slug, pair_key, ALIAS_MAP) are preserved.
"""

from __future__ import annotations
import re
import unicodedata
from typing import Set, Dict

ALIAS_MAP: Dict[str, str] = {
    # Directional and punctuation variants
    "texasa&m": "texasam",
    "texas a&m": "texasam",
    "texasam": "texasam",
    "stjohns": "stjohns",
    "saintjohns": "stjohns",
    "stbonaventure": "stbonaventure",
    "saintbonaventure": "stbonaventure",
    "stjosephs": "stjosephs",
    "saintjosephs": "stjosephs",
    "stmarys": "stmarys",
    "saintmarys": "stmarys",
    "saintmarysga": "stmarys",
    "stfrancispa": "stfrancispa",
    "saintfrancispa": "stfrancispa",
    "stfrancisny": "stfrancisny",
    "saintfrancisny": "stfrancisny",
    "unc": "northcarolina",
    "uconn": "connecticut",
    "utsa": "texassanantonio",
    "texassanantonio": "texassanantonio",
    "lsu": "louisianastate",
    "olemiss": "mississippiolemiss",
    "miami(fl)": "miamifl",
    "miamifl": "miamifl",
    "miami florida": "miamifl",
    "miamioh": "miamioh",
    "miami(oh)": "miamioh",
    # UC/Cal State style
    "calstatefullerton": "calstfullerton",
    "csufullerton": "calstfullerton",
    "calstfullerton": "calstfullerton",
    "calstatebakersfield": "calstbakersfield",
    "csubakersfield": "calstbakersfield",
    "calstbakersfield": "calstbakersfield",
    # Short names / acronyms
    "uab": "alabamabirmingham",
    "utep": "texaselpaso",
    "usc": "southerncalifornia",
    "unlv": "nevadalasvegas",
    "ucsd": "californiasandiego",
    "ucsb": "californiasantabarbara",
    "ucdavis": "californiadavis",
    "ucirvine": "californiairvine",
    "ucriverside": "californiariverside",
    "ucmerced": "californiamerced",
    # Directional/state variants
    "southernmississippi": "southernmiss",
    "southernmiss": "southernmiss",
    "alcorn": "alcornstate",
    "calbaptist": "californiabaptist",
    "williamandmary": "williammary",
    "williammary": "williammary",
    # Upstate / uncommon expansions
    "southcarolinaupstate": "uscupstate",
    "uscupstate": "uscupstate",
    # Saint vs St.
    "stpeters": "saintpeters",
    "saintpeters": "saintpeters",
    "stthomasmn": "stthomasmn",
    "saintthomasmn": "stthomasmn",
    # Mascot-inclusive variants (Albany, Norfolk St., San Jose St.)
    "albanygreatdanes": "ualbanygreatdanes",  # Odds feeds sometimes drop leading U
    "ualbanygreatdanes": "ualbanygreatdanes",
    "norfolkstspartans": "norfolkstatespartans",
    "norfolkstatespartans": "norfolkstatespartans",
    "sanjosestspartans": "sanjosestatespartans",
    "sanjosestatespartans": "sanjosestatespartans",
    # East Tennessee State abbreviation variants
    "easttennesseestbuccaneers": "easttennesseestatebuccaneers",
    "easttennesseestatebuccaneers": "easttennesseestatebuccaneers",
    # Cal State Fullerton mascot variants
    "csufullertontitans": "calstatefullertontitans",
    "calstatefullertontitans": "calstatefullertontitans",
    # High-confidence mappings from odds diagnostics (provider fused forms -> canonical institution+mascot or institution)
    "arkansasstredwolves": "arkansasstateredwolves",
    "washingtonstcougars": "washingtonstatecougars",
    "youngstownstpenguins": "youngstownstatepenguins",
    "moreheadsteagles": "moreheadstateeagles",
    "murraystracers": "murraystateracers",
    "oregonstbeavers": "oregonstatebeavers",
    "southcarolinastbulldogs": "southcarolinastatebulldogs",
    "fortwaynemastodons": "purduefortwaynemastodons",
    "grandcanyonantelopes": "grandcanyonlopes",
    "gardnerwebbbulldogs": "gardnerwebbrunninbulldogs",
    "georgiastpanthers": "georgiastatepanthers",
    "missvalleystdeltadevils": "mississippivalleystatedeltadevils",
    # Rebrand / abbreviation adjustments
    "gwrevolutionaries": "georgewashington",  # GWU rebrand, odds feed shortened
    "maineblack": "maine",  # truncated 'maineblackbears'
    "idahostatebengals": "idahostate",
    "texasaandmaggies": "texasam",  # fused variant
    "texasaandmccislanders": "texasamcorpuschristi",  # islanders
    "tarletonstatetexans": "tarletonstate",
    # Fused institution+mascot -> institution canonical (for schedule feeds concatenating both)
    "kansasjayhawks": "kansas",
    "northcarolinatarheels": "northcarolina",
    "louisianaragincajuns": "louisiana",
    "notredamefightingirish": "notredame",
    "wakeforestdemondeacons": "wakeforest",
    "wolfpack": "ncstate",  # generic leftover
    "tulsagoldenhurricane": "tulsa",
    "umasslowellriverhawks": "umasslowell",
    "gardnerwebbrunninbulldogs": "gardnerwebb",
    "pepperdinewaves": "pepperdine",
    "seelouisiana": "selouisiana",  # ensure variant mapping
    "bucknellbison": "bucknell",
    "rowanprofs": "rowan",  # non-D1 but keep mapping
    "mitengineers": "mit",  # exhibition mapping
        # From unmatched Nov-09 diagnostics (safe, high-confidence fusions/typos)
        "northcarolinacentraleagles": "northcarolinacentral",
        "dominicanildominican": "dominicanil",  # exhibition opponent form
        "oklahomastatecowboys": "oklahomastate",
        "illinoisstateredbirds": "illinoisstate",
        "southdakotastatejackrabbits": "southdakotastate",
        "northerniowapanthers": "northerniowa",
        "louisianatechbulldogs": "louisianatech",
        "montanastatebobcats": "montanastate",
        "arizonastatesundevils": "arizonastate",
        "evansvillepurpleaces": "evansville",
        "loyolachicagoramblers": "loyolachicago",
        "sandiegostateaztecs": "sandiegostate",
        "coloradostaterams": "coloradostate",
        "howardbison": "howard",
        "gramblingtigers": "gramblingstate",
        "oklahomastate": "oklahomastate",  # idempotent
        # Provider-specific fused mascot forms
        "georgiabulldogs": "georgia",
        "jamesmadisondukes": "jamesmadison",
        "pennsylvaniaquakers": "pennsylvania",
        "northtexasmeangreen": "northtexas",
        "idahostatebengals": "idahostate",
        # Known institution aliases
        "denverpioneers": "denver",
        "omahamavericks": "omaha",
        "oklahomacitystars": "oklahomacity",  # generic example
}

STOP_TOKENS = {
    # Removing these is generally safe for canonical slug
    "the": "",
    "university": "",
    "univ": "",
    # Keep 'state' and 'college' because they disambiguate
}

REPLACEMENTS = [
    (r"&", " and "),
    (r"a\s*&\s*m", "am"),
    (r"\bst\.\b", "st"),   # st. -> st (saint)
    (r"\bsaint\b", "st"),
    (r"[^a-z0-9]+", " "),
]

MULTISPACE = re.compile(r"\s+")

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_slug(name: str) -> str:
    if not isinstance(name, str) or not name:
        return ""
    s = _strip_accents(name).lower()
    for pat, repl in REPLACEMENTS:
        s = re.sub(pat, repl, s)
    # Remove stop tokens as whole words
    toks = [t for t in MULTISPACE.sub(" ", s).strip().split(" ") if t]
    # Contextual disambiguation for 'st' -> 'state' vs 'saint':
    # Heuristic: if 'st' occurs at the beginning of the name, interpret as 'saint';
    # otherwise (mid or end), interpret as 'state'. This matches common NCAA usage like
    # 'Arkansas St' (state) vs 'St John's' (saint).
    disamb = []
    for i, t in enumerate(toks):
        if t in {"st", "st."}:
            if i == 0:
                disamb.append("st")  # keep as saint marker
            else:
                disamb.append("state")
        else:
            disamb.append(t)
    toks = disamb
    # Strip trailing mascot tokens when the feed concatenates institution+mascot (e.g. 'airforcefalcons').
    # We remove only if more than one token and tail is in curated mascot set.
    mascot_tail = {
        "falcons","zips","crimsontide","eagles","mountaineers","sundevils","goldenlions","razorbacks","knights","tigers","governors","bears","bruins","wildcats","bearcats","broncos","braves","bulldogs","cougars","lancers","goldenbears","mustangs","titans","camels","griffins","chippewas","buccaneers","vikings","buffaloes","rams","roadrunners","matadors","biggreen","flyers","hornets","pioneers","dragons","bluedevils","pirates","hawks","stags","redbirds","hoosiers","sycamores","cyclones","jaguars","dolphins","dukes","panthers","leopards","explorers","mountainhawks","bisons","sharks","cardinals","ramblers","lions","greyhounds","blackbears","jaspers","redfoxes","goldeneagles","thunderingherd","minutemen","tigers","racers","cornhuskers","wolfpack","wildcats","lobos","privateers","purpleeagles","highlanders","fightinghawks","bison","norse","meangreen","demons","bobcats","cowboys","monarchs","mavericks","vaqueros","commodores","hokies","lakers","greatdanes","redwolves","redhawks","seawolves","penguins","lopes","texans","volunteers","aggies","islanders","jackrabbits","gaels","peacocks","friars","hornedfrogs","longhorns","greenwave","spiders","broncs","aztecs","dons","spartans","redstorm","billikens","nittanylions","vaqueros","eagles","wave","terriers","cowboys","musketeers","vaqueros","tide","rebels","gaels","deacons","deacons","gophers","huskies","hokies","spiders","warriors","texans","owls","hawks","wave","orange"
    }
    # Fused mascot stripping: if token is single fused and endswith a mascot substring, split
    if len(toks) == 1:
        single = toks[0]
        for masc in sorted(mascot_tail, key=len, reverse=True):
            if single.endswith(masc) and len(single) > len(masc) + 2:  # ensure institution part length
                inst = single[: -len(masc)]
                # Basic guard: avoid stripping if institution part too short (e.g., 'utahutes' -> 'utah')
                if len(inst) >= 3:
                    toks = [inst]
                break
    elif len(toks) > 1 and toks[-1] in mascot_tail:
        toks = toks[:-1]
    toks2 = [STOP_TOKENS.get(t, t) for t in toks]
    s2 = " ".join([t for t in toks2 if t])
    # Final collapse to slug without spaces
    slug = re.sub(r"[^a-z0-9]", "", s2)
    return slug

def canonical_slug(name: str) -> str:
    base = normalize_slug(name)
    if not base:
        return base
    return ALIAS_MAP.get(base, base)

def pair_key(a: str, b: str) -> str:
    ca = canonical_slug(a)
    cb = canonical_slug(b)
    if ca <= cb:
        return f"{ca}__{cb}"
    return f"{cb}__{ca}"

# Variant/expansion dictionaries used by candidate generation
_SAINT_VARIANTS = ["saint", "st", "st."]
_DIRECTIONAL_MAP = {
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "no": "north",
    "so": "south",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
}
_AMP_VARIANTS = ["a&m", "a & m", "aandm"]
_STOP_PREFIXES = ["the"]
_CAMPUS_QUALIFIERS = ["main", "campus", "university park"]
_ABBREV_MAP = {
    "utsa": "texas san antonio",
    "utep": "texas el paso",
    "ucla": "california los angeles",
    "uncg": "north carolina greensboro",
    "unlv": "nevada las vegas",
    "ucsb": "california santa barbara",
    "vcu": "virginia commonwealth",
    "lsu": "louisiana state",
    "tcu": "texas christian",
    "byu": "brigham young",
    "smu": "southern methodist",
    "fiu": "florida international",
    "fau": "florida atlantic",
    "uab": "alabama birmingham",
    "unc": "north carolina",
    "ncsu": "north carolina state",
    "ncstate": "north carolina state",
    "ucf": "central florida",
    "usf": "south florida",
    "ecu": "east carolina",
    "wvu": "west virginia",
    "uva": "virginia",
    "vt": "virginia tech",
}
_MASCOT_COMMON = {
    "bulldogs","tigers","wildcats","cougars","lions","bears","wolfpack","cardinals","raiders",
    "hawks","eagles","owls","broncos","spartans","gators","rams","badgers","crimson tide","orange",
    "rebels","aggies","aztecs","shockers","sun devils","demon deacons","blue devils","tar heels",
    "seminoles","longhorns","horned frogs","mountaineers","golden gophers","golden bears","golden eagles",
}

def _tokenize(name: str) -> list[str]:
    # Lowercase, replace punctuation/dashes with space, split
    n = re.sub(r"[\-\u2013\u2014]+", " ", name.lower())
    n = re.sub(r"[^a-z0-9& ]+", " ", n)
    return [t for t in n.split() if t]

def _expand_mount(tokens: list[str]) -> Set[str]:
    out = set()
    for i,t in enumerate(tokens):
        if t in {"mt","mt.","mnt"}:
            mod = tokens[:]
            mod[i] = "mount"
            out.add(" ".join(mod))
    return out

def _expand_saint(tokens: list[str]) -> Set[str]:
    out = set()
    if any(t in {"saint","st","st."} for t in tokens):
        for i, t in enumerate(tokens):
            if t in {"saint","st","st."}:
                for v in _SAINT_VARIANTS:
                    mod = tokens[:]
                    mod[i] = v
                    out.add(" ".join(mod))
    return out

def _expand_directional(tokens: list[str]) -> Set[str]:
    out = set()
    for i,t in enumerate(tokens):
        if t in _DIRECTIONAL_MAP:
            mod = tokens[:]
            mod[i] = _DIRECTIONAL_MAP[t]
            out.add(" ".join(mod))
    return out

def _expand_ampersand(tokens: list[str]) -> Set[str]:
    out = set()
    for i,t in enumerate(tokens):
        if t in {"a&m","a","aandm"} or t == "&":
            # Normalize cluster representing A&M forms
            for var in _AMP_VARIANTS:
                mod = [x for x in tokens if x not in {"&"}]  # remove standalone ampersand
                # Replace contiguous a m tokens if present
                joined = " ".join(mod)
                joined = re.sub(r"a\s*m", "a&m", joined)
                # Now swap a&m with variant
                out.add(joined.replace("a&m", var))
    return out

def _strip_prefixes(tokens: list[str]) -> Set[str]:
    out = set()
    if tokens and tokens[0] in _STOP_PREFIXES:
        out.add(" ".join(tokens[1:]))
    return out

def _strip_campus(tokens: list[str]) -> Set[str]:
    out = set()
    if tokens and tokens[-1] in _CAMPUS_QUALIFIERS:
        out.add(" ".join(tokens[:-1]))
    return out

def _expand_abbrev(tokens: list[str]) -> Set[str]:
    out = set()
    for i,t in enumerate(tokens):
        if t in _ABBREV_MAP:
            mod = tokens[:]
            mod[i] = _ABBREV_MAP[t]
            out.add(" ".join(mod))
    return out

def _strip_mascot(tokens: list[str]) -> Set[str]:
    out = set()
    # Remove trailing multi-token mascots first
    joined = " ".join(tokens)
    for masc in sorted(_MASCOT_COMMON, key=len, reverse=True):
        if joined.endswith(" "+masc) or joined == masc:
            core = joined[: -len(masc)].strip()
            if core:
                out.add(core)
    # Single token removal
    if tokens and tokens[-1] in _MASCOT_COMMON:
        out.add(" ".join(tokens[:-1]))
    return out

def generate_alias_candidates(name: str, max_candidates: int = 20) -> set[str]:
    """Generate rule-based alias candidates for a team name.

    Returns a set of raw name strings (not slugs). Caller should canonicalize each
    via canonical_slug. Does not include the original name.
    """
    if not name:
        return set()
    tokens = _tokenize(name)
    variants: Set[str] = set()
    generators = [
        _expand_saint,
        _expand_directional,
        _expand_mount,
        _expand_ampersand,
        _strip_prefixes,
        _strip_campus,
        _expand_abbrev,
        _strip_mascot,
    ]
    for fn in generators:
        try:
            variants |= fn(tokens)
        except Exception:
            pass
    # Also add simple hyphen join/split variants
    base = " ".join(tokens)
    collapsed = re.sub(r"\s+", " ", base)
    variants.add(collapsed.replace(" ", ""))  # compressed
    # Filter out the original name canonical and control length
    orig_slug = canonical_slug(name)
    out = set()
    for v in variants:
        if not v:
            continue
        if canonical_slug(v) != orig_slug:
            out.add(v)
        if len(out) >= max_candidates:
            break
    return out

# If odds coverage suggestion needs runtime candidate expansion, it can import generate_alias_candidates.
