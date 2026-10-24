"""Fixed Q&A about the Canada employee dataset, answered locally.

When a visitor asks something related to one of the questions below, ECHO
speaks that entry's answer VERBATIM -- no LLM, no budget slot, no invented
numbers. The figures are approximate by design; every answer already says
who to ask for exact numbers.

To change an answer, edit the `answer` text of the entry. To catch more ways
of asking, add words to its `aliases`. `locs` / `national` say which
locations an entry can speak for (see match_faq).

Matching is pure, offline keyword scoring (same idea as solutions.py) with
guards so near-neighbours don't swap: the question about Calgary headcount
never returns the Vancouver answer, "most common role" never returns "most
common skill", and anything that isn't clearly one of these questions falls
through to the normal chat path (match_faq returns None).
"""

from __future__ import annotations

import math
import re

# Every location a question can name. (A name can be spoken as a synonym:
# see _PHRASES / _SYN.)
_LOCATIONS = frozenset({"gta", "calgary", "vancouver", "montreal", "quebec",
                        "saskatoon"})
_ALL = _LOCATIONS
_MAJOR = frozenset({"gta", "calgary", "vancouver", "montreal"})

# fmt: off
FAQ = [
 dict(id=1, locs=(), national=True,
  question="How many employees are in the Canada employee dataset?",
  aliases="total headcount of employees in canada how many people work in canada workforce size overall number of staff canadian how many employees do we have in canada whole country entire country across canada overall size",
  answer="The dataset contains approximately 6,800 employee records across Canadian locations. For the exact and latest headcount, please connect with the HR team."),
 dict(id=2, locs=_ALL, national=True,
  question="Which locations in Canada do employees work from?",
  aliases="where do employees work offices cities sites location breakdown by location employees per location spread distribution subcon subcontractor canadian canada city cities sit based",
  answer="Employees are spread across GTA (around 3,600), Calgary (nearly 2,000), Vancouver (around 750), Subcon (around 230), Montreal (around 100), Others (around 90), Quebec (around 15) and Saskatoon (around 10). For exact figures on a specific location, please connect with the HR team or the location’s administration team."),
 dict(id=3, locs=("gta",), national=True,
  question="Which Canadian location has the most employees?",
  aliases="largest biggest location city office headquarters gta toronto how many employees in gta canada city sit",
  answer="GTA (Greater Toronto Area) has the most employees, with approximately 3,600 people, which is more than half of the total workforce of around 6,800. For exact figures, please connect with the HR team."),
 dict(id=4, locs=("calgary",), national=False,
  question="How many employees are based in Calgary?",
  aliases="calgary headcount people work in calgary calgary workforce how big size team office",
  answer="Calgary has nearly 2,000 employees, making it the second-largest location after GTA. For exact figures on the Calgary workforce, please connect with the HR team."),
 dict(id=5, locs=("vancouver",), national=False,
  question="How many employees work in Vancouver?",
  aliases="vancouver headcount people based in vancouver vancouver workforce how big size team office",
  answer="Vancouver has approximately 750 employees. For exact figures on the Vancouver workforce, please connect with the HR team."),
 dict(id=6, locs=("montreal", "quebec", "saskatoon"), national=False,
  question="How many employees are in Montreal, Quebec and Saskatoon?",
  aliases="montreal headcount quebec headcount saskatoon headcount people work in montreal quebec saskatoon how big size team office",
  answer="Montreal has around 100 employees, Quebec has around 15 and Saskatoon has around 10. For exact figures on these locations, please connect with the HR team."),
 dict(id=7, locs=(), national=True,
  question="What is the average experience of employees in Canada?",
  aliases="average years of experience tenure how experienced are employees overall mean experience canadian",
  answer="The average total experience across employees is approximately 13.5 years. For detailed experience reports, please connect with the HR team."),
 dict(id=8, locs=_ALL, national=True,
  question="What is the average experience by location?",
  aliases="average experience in each location per city experience in calgary gta vancouver montreal quebec saskatoon which location is most experienced how experienced are employees in calgary gta vancouver",
  answer="Average experience is around 14 years in GTA and Vancouver, around 13 years in Calgary and Montreal, around 13.5 years in Quebec and around 16 years in Saskatoon, which has the most experienced group on average. For detailed experience reports by location, please connect with the HR team."),
 dict(id=9, locs=(), national=True,
  question="How many employees have more than 20 years of experience?",
  aliases="over 20 years experience twenty years senior veterans long experience",
  answer="Approximately 660 employees have more than 20 years of total experience. For exact figures, please connect with the HR team."),
 dict(id=10, locs=(), national=True,
  question="What is the most common role in Canada?",
  aliases="top roles designation job title most employees hold technology lead senior consultant canadian",
  answer="Technology Lead is the most common role with around 2,550 employees, followed by Senior Consultant (around 860), Quality Engineering Lead (around 490), Project Manager (around 450) and Technology Analyst (around 350). For exact figures on roles, please connect with the HR team."),
 dict(id=11, locs=("calgary",), national=False,
  question="What is the most common role in Calgary?",
  aliases="calgary roles designation job title most employees hold in calgary",
  answer="In Calgary, the most common roles are Technology Lead (around 750), Senior Consultant (around 330) and Quality Engineering Lead (around 140). For exact figures on roles in Calgary, please connect with the HR team."),
 dict(id=12, locs=(), national=True,
  question="Which client account has the most employees in Canada?",
  aliases="biggest largest top client customer account project account most staff ms microsoft t-mobile citi pwc",
  answer="MS has the most employees with around 290, followed by T-Mobile USA, Inc. (around 210), CITI (around 200) and PricewaterhouseCoopers LLP (around 180). Another 225 or so employees are on Bench. For more details on any account, please connect with the concerned account or delivery team."),
 dict(id=13, locs=("gta",), national=False,
  question="Which is the biggest client account in GTA?",
  aliases="largest top client customer account in gta toronto most employees citi pricewaterhousecoopers",
  answer="In GTA, CITI is the largest client account with around 180 employees, followed by PricewaterhouseCoopers LLP with around 130. There are also around 140 employees mapped to Bench in GTA. For more details, please connect with the concerned account or delivery team."),
 dict(id=14, locs=("calgary",), national=False,
  question="Which is the biggest client account in Calgary?",
  aliases="largest top client customer account in calgary most employees southern california edison charles schwab",
  answer="In Calgary, Southern California Edison Company is the largest account with around 120 employees, followed by Charles Schwab & Co., Inc. (around 100) and MS (around 85). For more details, please connect with the concerned account or delivery team."),
 dict(id=15, locs=("vancouver",), national=False,
  question="Which is the biggest client account in Vancouver?",
  aliases="largest top client customer account in vancouver most employees boeing t-mobile",
  answer="In Vancouver, MS is the largest account with around 150 employees, followed by T-Mobile USA, Inc. (around 75) and The Boeing Company (around 60). For more details, please connect with the concerned account or delivery team."),
 dict(id=16, locs=("montreal",), national=False,
  question="Which clients do the Montreal employees mainly work for?",
  aliases="montreal client customer account who do montreal employees work for canadian national railway morgan stanley",
  answer="Most Montreal employees work for Canadian National Railway Company (around 40), followed by Morgan Stanley (around 25). For more details, please connect with the concerned account or delivery team."),
 dict(id=17, locs=("quebec", "saskatoon"), national=False,
  question="Which clients are served from Quebec and Saskatoon?",
  aliases="quebec client customer account saskatoon client customer account who do quebec saskatoon employees work for pfizer airbus bhp",
  answer="In Quebec, most employees work for Pfizer Global Pharmaceuticals (around 10), with a few on Airbus. In Saskatoon, most work for BHP Billiton Marketing Asia Pte Ltd (around 7). For more details, please connect with the concerned account or delivery team."),
 dict(id=18, locs=(), national=True,
  question="Which Canadian-headquartered clients have the most employees?",
  aliases="canada hq canadian client customer account toronto dominion td rogers canadian national railway scotia bank public works",
  answer="The largest Canada-headquartered client accounts are THE TORONTO-DOMINION BANK and ROGERS COMMUNICATIONS CANADA INC (around 85 each), followed by Canadian National Railway Company (around 80), Public Works and Government Services Canada (around 50) and Scotia Bank (around 35). For more details, please connect with the concerned account or delivery team."),
 dict(id=19, locs=(), national=True,
  question="Where are the headquarters of the clients that Canadian employees work for?",
  aliases="client hq country usa united states india japan hong kong switzerland which countries are the clients from belong country",
  answer="Most employees, approximately 5,000, serve clients headquartered in the USA. Others serve clients headquartered in Canada (around 600), India (around 200), Japan (around 190), Hong Kong SAR of China (around 130) and Switzerland (around 70). Around 225 employees are on Bench. For more details, please connect with the concerned account or delivery team."),
 dict(id=20, locs=(), national=True,
  question="What are the main business segments in Canada?",
  aliases="segment bfs cmt crl sure coremfg hc ins ls business units verticals largest segment belong",
  answer="The largest segments are BFS and CMT (around 1,300 employees each), CRL (around 1,200), SURE (around 1,100) and COREMFG (around 540), followed by HC (around 300), INS (around 300) and LS (around 200). For more details on any segment, please connect with the concerned segment or delivery team."),
 dict(id=21, locs=_MAJOR, national=True,
  question="Which segment is largest in each major location?",
  aliases="biggest segment by location per city in gta calgary vancouver montreal bfs sure cmt crl",
  answer="BFS is the largest segment in GTA (around 840), SURE is the largest in Calgary (around 440), CMT is the largest in Vancouver (around 310) and CRL is the largest in Montreal (around 50). For more details, please connect with the concerned segment or delivery team."),
 dict(id=22, locs=(), national=True,
  question="Which industries do Canadian employees support the most?",
  aliases="industry sectors domains verticals financial services telecommunications retail energy corporate services application code canada",
  answer="By application code, Financial Services is the largest industry (around 1,100 employees), followed by Telecommunications (around 690), Corporate Services (around 590), Retail (around 560) and Energy and Environmental (around 530). For more details, please connect with the concerned delivery team."),
 dict(id=23, locs=(), national=True,
  question="What is the most common billing type?",
  aliases="billing fixed price fp ctm time and material t&m commercial model contract type how many employees on fixed price ctm used",
  answer="Fixed Price (FP) is the most common billing type with around 3,450 employees, followed by CTM (around 1,450) and Time and Material, T&M (around 1,250). For more details on billing, please connect with the concerned finance or delivery team."),
 dict(id=24, locs=_MAJOR, national=True,
  question="How many employees are on bench in Canada?",
  aliases="bench benched unallocated resource management bench on notice how many on bench in gta calgary vancouver montreal how many are on bench in calgary",
  answer="Approximately 220 employees are on bench, with a few more on bench on-notice. By location, GTA has around 140 bench employees, Calgary around 50, Vancouver around 20 and Montreal only a couple. For the latest bench status, please connect with the Resource Management team or HR."),
 dict(id=25, locs=(), national=True,
  question="How many employees are in production projects?",
  aliases="production status project status allocation pre-production leave support project sales utilisation utilization deployed billable",
  answer="Approximately 6,100 employees are on Production projects. The rest are on bench (around 220), leave (around 170), project-sales (around 70), pre-production (around 65), support (around 40) and other categories. For more details, please connect with the Resource Management team or HR."),
 dict(id=26, locs=("gta", "calgary"), national=True,
  question="What is the split between Deputee and Base employees?",
  aliases="deputee deputees base employees employee type onsite offshore split ratio difference how many base employees are there in canada count",
  answer="There are approximately 5,100 Deputee employees and around 1,650 Base employees. GTA has around 2,550 Deputees and 1,050 Base employees, while Calgary has around 1,750 Deputees and 200 Base employees. For more details on employee type, please connect with the HR team."),
 dict(id=27, locs=(), national=True,
  question="What working modes do employees use?",
  aliases="work mode network infosys network client network vpn iconnect remote hybrid how do employees connect work from home employees work remotely work on which network",
  answer="Most employees work through the Infosys network (around 2,750) or the Client network (around 2,000). Others use Direct Client Network enabled (around 220), VPN (around 130) or VPN or Iconnect (around 95), and around 1,450 records are ‘Not Available’. For more details on working mode, please connect with the concerned delivery team or IT support."),
 dict(id=28, locs=(), national=True,
  question="What are the most common skills among Canadian employees?",
  aliases="top skills technology skillset expertise scrum java core java dotnet net springboot agile technical skills top skills canadian",
  answer="The most common skills are Infosys Global Agile methodology (around 780), Scrum (around 500), Java Application Support (around 420), Core Java and .Net Application Development (around 370 each) and Springboot (around 340). For more details on skills, please connect with the HR or Resource Management team."),
 dict(id=29, locs=(), national=True,
  question="What are the most common digital skill tags?",
  aliases="digital skill tags scrum master devops cloud professional sdet data professional ai builder certification tags",
  answer="The top digital skill tags are SCRUM MASTER (around 430), DEVOPS PROFESSIONAL (around 400), CLOUD PROFESSIONAL (around 280), SOFTWARE DEVELOPMENT ENGINEER IN TEST (SDET) (around 240) and DATA PROFESSIONAL (around 190). Around 110 employees are tagged as AI BUILDER. For more details, please connect with the HR or Resource Management team."),
 dict(id=30, locs=_MAJOR, national=True,
  question="How many employees have Generative AI skills and where are they located?",
  aliases="genai gen ai artificial intelligence ai skills llm how many people know ai ai expertise",
  answer="Approximately 200 employees list Generative AI skills: around 90 in GTA, 60 in Calgary, 40 in Vancouver, 15 in Montreal and one in Others. For more details, please connect with the HR or Resource Management team."),
]
# fmt: on

# The topic word(s) a question must contain before an entry may answer it.
# Each entry lists groups; every group needs at least one of its words. This
# stops "most common NAME in Canada" matching the most-common-ROLE answer.
_NEED = {
    1: [("employee",)],
    2: [("location", "city"), ("employee", "work", "count", "gta", "canada", "canadian")],
    # "largest location" needs a headcount-ish word, so "largest CITY in
    # Canada" (a geography question) is left to the normal chat.
    3: [("location", "gta", "city"), ("employee", "count", "gta", "location"),
        ("big", "gta", "count")],
    4: [("employee", "count", "big")], 5: [("employee", "count", "big")],
    6: [("employee", "count", "big")],
    7: [("experience",), ("employee", "canada", "canadian")],
    8: [("experience",), ("location", "city", "each", "per", "gta", "calgary",
                           "vancouver", "montreal", "quebec", "saskatoon")],
    9: [("experience",)],
    10: [("role",)], 11: [("role",)],
    12: [("client",)], 13: [("client",)], 14: [("client",)], 15: [("client",)],
    16: [("client",)], 17: [("client",)],
    18: [("client",), ("canada", "canadian", "hq")],
    19: [("client",), ("hq", "country", "location")],
    20: [("segment",)],
    21: [("segment",), ("each", "per", "location", "city", "gta", "calgary", "vancouver", "montreal")],
    22: [("industry",)],
    23: [("bill", "fp", "ctm", "tm")],
    24: [("bench",)], 25: [("production",)],
    26: [("deputee", "base")],
    27: [("workmode", "remote", "network", "vpn", "hybrid")],
    28: [("skill",)],
    29: [("tag", "digital", "scrum", "devops", "sdet", "builder")],
    30: [("genai",)],
}
# Words that make an entry the WRONG answer: "which country has the MOST
# employees" is not the plain headcount, and a list of locations is not "the
# biggest location".
_FORBID = {1: ("big", "common"), 2: ("big", "common")}

# Words that mean the visitor is asking about something else entirely.
_VETO = frozenset({"live", "resident", "population", "capital", "salary", "pay",
                   "paid", "weather", "temperature", "price", "cost", "restaurant",
                   "hotel", "flight", "visit", "tourist", "born", "prime",
                   "minister", "president", "gdp", "tax", "rent", "house"})
for _e in FAQ:
    _e["need"] = _NEED.get(_e["id"], [])
    _e["forbid"] = frozenset(_FORBID.get(_e["id"], ()))

# ---------------------------------------------------------------------------
# Normalisation: speech-to-text gives "How many people work in the G T A?",
# so questions and entries go through the same pipeline.
# ---------------------------------------------------------------------------

# Multi-word phrases -> one token. Applied to lower-cased text with
# punctuation already turned into spaces.
_PHRASES = [
    (r"greater toronto area|g t a|g\.t\.a", "gta"),
    (r"head ?count", "count employee"),
    (r"how many|how much|number of|count of|total number of", "count"),
    (r"most common|commonly|more common|commonest|most used|used most|most popular|most frequent(?:ly)?", "common"),
    (r"years? of experience", "experience"),
    (r"more than|greater than|exceeding|exceeds?|above|beyond", "over"),
    (r"gen a i|gen ai|generative a i|generative ai|gen\.ai", "genai"),
    (r"work(?:ing)? modes?|mode of work|work from home|wfh", "workmode"),
    (r"client accounts?|customer accounts?", "client"),
    (r"head ?quarter(?:s|ed)?|\bhq\b", "hq"),
    (r"job titles?|designations?", "role"),
    (r"time and material|time & material|t&m|t and m", "tm"),
    (r"fixed price", "fp"),
    (r"on bench|benched|bench", "bench"),
]
_PHRASES = [(re.compile(p), r) for p, r in _PHRASES]

_STOP = frozenset("""
a an the of in on at to for from with by and or but that this these those it
its is are was were be been am do does did has have had there here what whats
which who whom whose when why also please tell me you your yours our ours us
we i my can could would should will shall may might give show say know want
need like about any some something anything just than then so um uh okay ok
hey hi hello echo again kindly based located about into out up are use used using uses infosys infosy across compare comparison
""".split())

# Stem-level synonyms (applied after stemming).
_SYN = {
    "people": "employee", "person": "employee", "staff": "employee",
    "workforce": "employee", "folk": "employee", "member": "employee",
    "team": "employee", "resource": "employee", "associate": "employee",
    "toronto": "gta", "ontario": "gta",
    "montréal": "montreal", "québec": "quebec", "saskatchewan": "saskatoon",
    "where": "location", "office": "location", "site": "location",
    "place": "location", "branch": "location",
    "biggest": "big", "largest": "big", "top": "big", "major": "big",
    "main": "big", "primary": "big", "most": "big", "highest": "big",
    "popular": "common", "frequent": "common", "typical": "common",
    "avg": "average", "mean": "average",
    "tenure": "experience", "experienced": "experience", "seniority": "experience",
    "customer": "client", "account": "client",
    "position": "role", "job": "role", "title": "role",
    "skillset": "skill", "expertise": "skill", "technology": "tech",
    "twenty": "20",
    "remotely": "remote", "canadians": "canadian",
    "headquarter": "hq", "headquartered": "hq",
    "generative": "genai", "genai": "genai",
    "industrie": "industry", "industrial": "industry", "sector": "industry",
    "domain": "industry", "vertical": "industry",
    "ct": "ctm",
}


def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    if len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    else:
        return w
    if len(w) > 2 and w[-1] == w[-2] and w[-1] in "bdgmnprt":   # tagged -> tag
        w = w[:-1]
    return w


_WHO_WORKS_FOR = re.compile(r"\bwho\b.*\bwork(?:s|ing)? for\b")


def _tokens(text: str) -> set[str]:
    t = (text or "").lower().replace("&", " and ")
    t = re.sub(r"[^a-z0-9éèàç ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    for rx, rep in _PHRASES:
        t = rx.sub(f" {rep} ", t)
    out = set()
    if _WHO_WORKS_FOR.search(t):          # "who do the Montreal employees work for"
        out.add("client")
    for w in t.split():
        if w in _STOP or (len(w) < 2 and not w.isdigit()):
            continue
        w = _SYN.get(w, w)
        w = _SYN.get(_stem(w), _stem(w))
        if w and w not in _STOP:
            out.add(w)
    return out


# Entries: precompute vocab, locations and idf weights once. Words from the
# question itself count fully; words only in the aliases count a bit less.
_ALIAS_FACTOR = 0.6
for _e in FAQ:
    _qt = _tokens(_e["question"])
    _e["vocab"] = _qt | _tokens(_e["aliases"])
    _e["_f"] = {t: (1.0 if t in _qt else _ALIAS_FACTOR) for t in _e["vocab"]}
    _e["_locs"] = frozenset(_e["locs"])

_N = len(FAQ)
_DF: dict[str, int] = {}
for _e in FAQ:
    for _t in _e["vocab"]:
        _DF[_t] = _DF.get(_t, 0) + 1


def _w(t: str) -> float:
    """Rarer across the 30 entries = more telling. Unknown words count as
    the rarest, so stray content words pull confidence down."""
    df = _DF.get(t, 0)
    if df == 0:
        return _UNKNOWN_W
    return math.log((_N + 1) / (df + 0.5)) + 0.5


# Entry "size" for the focus penalty, filled in after the weights exist.
for _e in FAQ:
    _e["_size"] = sum(_w(t) * f for t, f in _e["_f"].items())

# Tuning knobs (see match_faq).
_UNKNOWN_W = 2.5        # weight of a question word no entry knows ("google")
_SIZE_EXP = 0.20        # how strongly long entries are penalised
MIN_COVERAGE = 0.60     # share of the question's weight an entry must explain
MIN_SCORE = 2.0         # and at least this much absolute weight
MIN_LEAD = 1.06         # best must beat the runner-up by 6% so near-ties fall through


def match_faq(question: str):
    """Return the FAQ entry that answers `question`, or None.

    Locations are guarded: a question naming a location only matches
    entries that can speak for it; a question naming none only matches
    entries with a national answer. Candidates are ranked by how well the
    question and the entry cover each other (so a short, focused entry beats
    a long one that merely contains the same common words). Anything not
    clearly one of the 30 questions returns None and takes the normal chat
    path."""
    q = _tokens(question)
    if not q or q & _VETO:
        return None
    qlocs = q & _LOCATIONS
    qtotal = sum(_w(t) for t in q)
    scored = []
    for e in FAQ:
        if qlocs:
            if not qlocs <= e["_locs"]:
                continue
        elif not e["national"]:
            continue
        if not all(q & set(g) for g in e["need"]) or q & e["forbid"]:
            continue
        hit = q & e["vocab"]
        if not hit:
            continue
        covered = sum(_w(t) for t in hit)                 # how much of the question
        matched = sum(_w(t) * e["_f"][t] for t in hit)     # how strongly
        rank = (covered / qtotal) * matched / (math.sqrt(qtotal) * e["_size"] ** _SIZE_EXP)
        scored.append((rank, covered, matched, e))
    if not scored:
        return None
    scored.sort(key=lambda s: s[0], reverse=True)
    rank, covered, matched, entry = scored[0]
    if matched < MIN_SCORE or covered / qtotal < MIN_COVERAGE:
        return None
    if len(scored) > 1 and rank < scored[1][0] * MIN_LEAD:
        return None
    return entry
