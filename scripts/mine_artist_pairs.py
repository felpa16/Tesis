#!/usr/bin/env python3
"""Mine same-artist style positives: two works by one performer, one era.

A style positive is two recordings by the same performer of two *different*
works (so they differ in content) that were released on the same album or
less than a year apart (so they are likely to share style). See experiments.md
section 26 for why, and scripts/listen_artist_pairs.py for the listening check
that has to pass before any of these pairs is trained on.

Performer. SHS credits are used as written, folded only for case and
whitespace: "Nat King Cole and His Trio" and "Nat King Cole with Orchestra
conducted by Gordon Jenkins" are different performers here, on purpose, since
a trio and an orchestra are different styles. --performer-key loose merges
them (accompaniment credits cut off) for comparison.

Release dates and albums. SHS100K has neither, and SecondHandSongs answers
scripted requests with a Cloudflare challenge, so they come from MusicBrainz
(CC0): one recording search per track, artist + title. A match needs a search
score >= --min-mb-score, the same normalized title, and an artist credit that
contains the performer's name. From the matches:
  * date  -- the earliest release date of any official release (singles
             included), kept as an interval: "1962" means 1962-01-01..12-31;
  * album -- the release group of type Album (or EP) without a Compilation,
             Live, Remix, DJ-mix or Demo tag whose earliest release is the
             earliest such; a song first released as a single still gets the
             album it appeared on.
The MusicBrainz date belongs to the recording MusicBrainz matched, which is not
necessarily the YouTube upload (a live take, a re-recording). The listening
check is what catches that.

Pair rule (the author's): same album, or released less than a year apart. With
dates known only to the year or month, "less than a year" must hold for every
day the intervals allow, so two years-only dates qualify only when equal.

MusicBrainz asks for one request per second and an identifying User-Agent.
Pass --user-agent "name/version ( contact )" or set MB_USER_AGENT; the default
is a generic name without contact details. Lookups are cached in
{out}/musicbrainz_{split}.jsonl and resume where they stopped, so the script
can be interrupted at any point. Pairs are rebuilt from the cache on every run.

Scale (train, downloaded, exact credits): 9,280 performers have >= 2 works,
31,376 recordings, ~9 h of lookups. --performers N looks up a random sample of
N eligible performers only, which is enough for the listening check.

Outputs in --out (default data/artist_pairs/):
  musicbrainz_{split}.jsonl   lookup cache, one line per track key
  {split}.jsonl               one line per style-positive pair
  {split}_summary.json        counts

The twelve cover factories of experiments.md section 25 (the work-disjoint
performer test) are excluded unless --keep-factories: training on their
performer identity would let that test measure memorised performers.

Examples:
    python scripts/mine_artist_pairs.py --performers 300 --seed 0
    python scripts/mine_artist_pairs.py                 # everything, ~9 h
    python scripts/mine_artist_pairs.py --no-fetch      # rebuild pairs from the cache
"""

from __future__ import annotations

import argparse
import calendar
import collections
import datetime as dt
import itertools
import json
import os
import random
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from shs100k_meta import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    Track,
    downloaded_csv,
    keys_from_csv,
    split_tracks,
)

MB_SEARCH = "https://musicbrainz.org/ws/2/recording/"
DEFAULT_USER_AGENT = "shs100k-style-pairs/0.1 ( research script )"

# experiments.md section 25, "(a) SHS100K's cover factories": the evaluation
# grid for performer identification, split by work. Kept out of training.
EVAL_FACTORIES = {
    "twinkle twinkle little rock star",
    "vitamin string quartet",
    "midnite string quartet",
    "8 bit arcade",
    "roma symphony orchestra",
    "yoga pop ups",
    "3 na bossa",
    "fausto papetti",
    "8-bit misfits",
    "acoustic heartstrings",
    "lullaby baby trio",
    "music box mania",
}

# Release-group secondary types whose dates and albums say nothing about when
# the performer recorded the song in that style.
EXCLUDED_SECONDARY = {"Compilation", "Live", "Remix", "DJ-mix", "Demo", "Mixtape/Street"}
ALBUM_TYPES = {"Album", "EP"}
# A MusicBrainz recording disambiguated like this is a different take.
ALTERNATE_TAKE = re.compile(r"\b(live|demo|remix|rehearsal|alternate|edit|mix)\b", re.I)

# Credits after these words name accompanists, not the performer.
_ACCOMPANIMENT = re.compile(
    r"\s+(with|feat\.?|featuring|ft\.?|cond\.? by|conducted by|"
    r"and (his|her|their)\b|& (his|her|their)\b|\(|\[|-\s)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------


def fold(text: str) -> str:
    """Lowercase ASCII, apostrophes dropped, other punctuation to single spaces."""
    text = re.sub(r"['’‘`´]", "", text).replace("&", " and ")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def primary_name(artist: str) -> str:
    """The performer's own name: the credit up to the first accompaniment word."""
    match = _ACCOMPANIMENT.search(artist)
    return (artist[: match.start()] if match else artist).strip() or artist.strip()


def query_name(artist: str) -> str:
    """primary_name without quote marks: 'Nat "King" Cole' -> 'Nat King Cole'."""
    return " ".join(re.sub(r"[\"“”]", " ", primary_name(artist)).split())


def contains_words(longer: str, shorter: str) -> bool:
    """True when `shorter` occurs in `longer` as whole words (both folded)."""
    return bool(shorter) and f" {shorter} " in f" {longer} "


def performer_key(artist: str, mode: str) -> str:
    if mode == "exact":
        return " ".join(artist.lower().split())
    name = fold(primary_name(artist))
    return re.sub(r"^the ", "", name)


def title_key(title: str) -> str:
    """Title without parenthesized or bracketed qualifiers ("(Remastered)")."""
    return fold(re.sub(r"[\(\[].*?[\)\]]", " ", title))


# --------------------------------------------------------------------------
# Dates: (lo, hi) intervals, so "1962" and "1962-05-01" compare honestly
# --------------------------------------------------------------------------


def date_interval(text: str | None) -> tuple[dt.date, dt.date] | None:
    if not text:
        return None
    parts = text.split("-")
    try:
        year = int(parts[0])
        if len(parts) == 1:
            return dt.date(year, 1, 1), dt.date(year, 12, 31)
        month = int(parts[1])
        if len(parts) == 2:
            last = calendar.monthrange(year, month)[1]
            return dt.date(year, month, 1), dt.date(year, month, last)
        day = dt.date(year, month, int(parts[2]))
        return day, day
    except (ValueError, IndexError):
        return None


def max_gap_days(a: str, b: str) -> int | None:
    """The largest gap, in days, the two (possibly partial) dates allow."""
    ia, ib = date_interval(a), date_interval(b)
    if ia is None or ib is None:
        return None
    return max((ia[1] - ib[0]).days, (ib[1] - ia[0]).days)


def _earlier(a: str | None, b: str | None) -> str | None:
    """The date whose interval starts first; ties go to the more precise one."""
    ia, ib = date_interval(a), date_interval(b)
    if ia is None:
        return b if ib is not None else None
    if ib is None:
        return a
    if ia[0] != ib[0]:
        return a if ia[0] < ib[0] else b
    return a if (ia[1] - ia[0]) <= (ib[1] - ib[0]) else b


# --------------------------------------------------------------------------
# MusicBrainz
# --------------------------------------------------------------------------


def lucene_phrase(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


class MusicBrainz:
    """Recording search at MusicBrainz's one-request-per-second limit."""

    def __init__(self, user_agent: str, interval: float = 1.1, retries: int = 6) -> None:
        self.user_agent = user_agent
        self.interval = interval
        self.retries = retries
        self._last = 0.0

    def _get(self, url: str) -> dict:
        delay = self.interval
        for attempt in range(self.retries):
            wait = self._last + self.interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            request = urllib.request.Request(
                url, headers={"User-Agent": self.user_agent, "Accept": "application/json"}
            )
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504):
                    raise
                retry_after = error.headers.get("Retry-After")
                delay = max(delay * 2, float(retry_after) if retry_after else 0.0)
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                delay *= 2
            time.sleep(min(delay, 60.0))
        raise RuntimeError(f"MusicBrainz kept failing after {self.retries} attempts")

    def search(self, artist: str, title: str, limit: int = 25) -> list[dict]:
        query = f"recording:{lucene_phrase(title)} AND artist:{lucene_phrase(artist)}"
        params = urllib.parse.urlencode({"query": query, "fmt": "json", "limit": limit})
        return self._get(f"{MB_SEARCH}?{params}").get("recordings", [])


def summarize(track: Track, recordings: list[dict], min_score: int) -> dict:
    """The lookup record for one track: matches, earliest date, original album."""
    name = fold(query_name(track.artist))
    wanted_title = title_key(track.title)
    matched = []
    for rec in recordings:
        if int(rec.get("score", 0)) < min_score:
            continue
        if title_key(rec.get("title", "")) != wanted_title:
            continue
        if ALTERNATE_TAKE.search(rec.get("disambiguation", "") or ""):
            continue
        credits = [fold(c.get("name", "")) for c in rec.get("artist-credit", [])]
        credits.append(fold(" ".join(c.get("name", "") + c.get("joinphrase", "")
                                     for c in rec.get("artist-credit", []))))
        if not any(contains_words(c, name) or contains_words(name, c) for c in credits):
            continue
        matched.append(rec)

    date: str | None = None
    groups: dict[str, dict] = {}
    artist_ids: set[str] = set()
    for rec in matched:
        for credit in rec.get("artist-credit", []):
            if credit.get("artist", {}).get("id"):
                artist_ids.add(credit["artist"]["id"])
        date = _earlier(date, rec.get("first-release-date"))
        for release in rec.get("releases", []):
            if release.get("status") not in (None, "Official"):
                continue
            group = release.get("release-group", {})
            secondary = set(group.get("secondary-types") or [])
            if secondary & EXCLUDED_SECONDARY:
                continue
            date = _earlier(date, release.get("date"))
            if group.get("primary-type") in ALBUM_TYPES and group.get("id"):
                entry = groups.setdefault(
                    group["id"], {"title": release.get("title", ""), "date": None}
                )
                entry["date"] = _earlier(entry["date"], release.get("date"))

    album = None
    dated = [(gid, g) for gid, g in groups.items() if date_interval(g["date"])]
    if dated:
        gid, g = min(dated, key=lambda item: date_interval(item[1]["date"])[0])
        album = {"id": gid, "title": g["title"], "date": g["date"]}
    return {
        "key": track.key,
        "status": "ok" if matched else "no_match",
        "query_artist": query_name(track.artist),
        "query_title": track.title,
        "n_results": len(recordings),
        "n_matched": len(matched),
        "best_score": max((int(r.get("score", 0)) for r in matched), default=None),
        "date": date,
        "album": album,
        "artist_ids": sorted(artist_ids),
    }


def read_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    cache[record["key"]] = record
    return cache


# --------------------------------------------------------------------------
# Pairs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StylePair:
    performer: str
    relation: str  # "same_album" | "within_year"
    a: Track
    b: Track
    gap_days: int | None
    album_a: str
    album_b: str
    date_a: str | None
    date_b: str | None

    def to_json(self) -> dict:
        return {
            "performer": self.performer,
            "relation": self.relation,
            "key_a": self.a.key,
            "key_b": self.b.key,
            "work_a": self.a.song_id,
            "work_b": self.b.song_id,
            "artist_a": self.a.artist,
            "artist_b": self.b.artist,
            "title_a": self.a.title,
            "title_b": self.b.title,
            "date_a": self.date_a,
            "date_b": self.date_b,
            "max_gap_days": self.gap_days,
            "album_a": self.album_a,
            "album_b": self.album_b,
        }


def relation(ra: dict, rb: dict, max_days: int, require_same_artist_id: bool) -> str | None:
    if ra.get("status") != "ok" or rb.get("status") != "ok":
        return None
    ids_a, ids_b = set(ra.get("artist_ids", [])), set(rb.get("artist_ids", []))
    if require_same_artist_id and ids_a and ids_b and not ids_a & ids_b:
        return None  # one lookup matched a namesake: neither date is trusted
    album_a, album_b = ra.get("album"), rb.get("album")
    if album_a and album_b and album_a["id"] == album_b["id"]:
        return "same_album"
    gap = max_gap_days(ra.get("date"), rb.get("date"))
    if gap is not None and gap < max_days:
        return "within_year"
    return None


def build_pairs(
    performers: dict[str, list[Track]],
    cache: dict[str, dict],
    max_days: int,
    require_same_artist_id: bool,
) -> list[StylePair]:
    pairs = []
    for performer, tracks in sorted(performers.items()):
        for a, b in itertools.combinations(tracks, 2):
            if a.song_id == b.song_id:
                continue  # the same work twice is a cover of itself, not a style pair
            if a.youtube_id == b.youtube_id:
                continue  # a medley listed under two works: the same audio twice
            ra, rb = cache.get(a.key), cache.get(b.key)
            if ra is None or rb is None:
                continue
            kind = relation(ra, rb, max_days, require_same_artist_id)
            if kind is None:
                continue
            album_a = (ra.get("album") or {}).get("title", "")
            album_b = (rb.get("album") or {}).get("title", "")
            pairs.append(StylePair(
                performer, kind, a, b, max_gap_days(ra.get("date"), rb.get("date")),
                album_a, album_b, ra.get("date"), rb.get("date"),
            ))
    return pairs


# --------------------------------------------------------------------------


def eligible_performers(
    split: str, data_root: Path, key_mode: str, keep_factories: bool
) -> dict[str, list[Track]]:
    """Performer -> downloaded tracks, for performers with >= 2 distinct works."""
    downloaded = keys_from_csv(downloaded_csv(data_root, split))
    if not downloaded:
        downloaded = keys_from_csv(downloaded_csv(DEFAULT_DATA_ROOT, split))
    if not downloaded:
        raise SystemExit(f"no downloaded-songs CSV for {split} under {data_root} or "
                         f"{DEFAULT_DATA_ROOT}")
    by_performer: dict[str, list[Track]] = collections.defaultdict(list)
    for track in split_tracks(split):
        if track.key in downloaded:
            by_performer[performer_key(track.artist, key_mode)].append(track)
    factories = {performer_key(f, mode) for f in EVAL_FACTORIES for mode in ("exact", "loose")}
    eligible = {}
    for performer, tracks in by_performer.items():
        if not keep_factories and (performer in factories
                                   or performer_key(performer, "loose") in factories):
            continue
        if len({t.song_id for t in tracks}) >= 2:
            eligible[performer] = tracks
    return eligible


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("DATA", DEFAULT_DATA_ROOT)),
                        help="where logs/{split}_downloaded_songs.csv lives "
                        "(falls back to the repo's data/)")
    parser.add_argument("--out", type=Path, default=DEFAULT_DATA_ROOT / "artist_pairs")
    parser.add_argument("--performer-key", choices=("exact", "loose"), default="exact")
    parser.add_argument("--performers", type=int, default=0,
                        help="look up a random sample of this many eligible performers "
                        "(0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-lookups", type=int, default=0,
                        help="stop after this many new lookups (0 = no cap)")
    parser.add_argument("--no-fetch", action="store_true",
                        help="only rebuild pairs from the cache")
    parser.add_argument("--min-mb-score", type=int, default=90)
    parser.add_argument("--max-days", type=int, default=365,
                        help="'less than a year apart', for every day the dates allow")
    parser.add_argument("--allow-namesakes", action="store_true",
                        help="keep pairs whose MusicBrainz artist ids do not overlap")
    parser.add_argument("--keep-factories", action="store_true",
                        help="do not exclude the evaluation cover factories")
    parser.add_argument("--user-agent",
                        default=os.environ.get("MB_USER_AGENT", DEFAULT_USER_AGENT))
    args = parser.parse_args()

    performers = eligible_performers(args.split, args.data_root, args.performer_key,
                                     args.keep_factories)
    n_tracks = sum(len(t) for t in performers.values())
    print(f"{len(performers)} performers with >= 2 downloaded works, {n_tracks} recordings")
    if args.performers > 0:
        rng = random.Random(args.seed)
        chosen = rng.sample(sorted(performers), min(args.performers, len(performers)))
        performers = {p: performers[p] for p in sorted(chosen)}
        print(f"sampled {len(performers)} performers "
              f"({sum(len(t) for t in performers.values())} recordings), seed {args.seed}")

    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / f"musicbrainz_{args.split}.jsonl"
    cache = read_cache(cache_path)
    todo = [t for tracks in performers.values() for t in tracks if t.key not in cache]
    if args.max_lookups > 0:
        todo = todo[: args.max_lookups]

    if todo and not args.no_fetch:
        print(f"{len(cache)} lookups cached; {len(todo)} to do "
              f"(~{len(todo) * 1.1 / 3600:.1f} h at one per second)")
        client = MusicBrainz(args.user_agent)
        started = time.monotonic()
        matched = 0
        with open(cache_path, "a", encoding="utf-8") as f:
            for i, track in enumerate(todo, start=1):
                try:
                    results = client.search(query_name(track.artist), track.title)
                    record = summarize(track, results, args.min_mb_score)
                except urllib.error.HTTPError as error:
                    if error.code != 400:
                        print(f"  {track.key}: lookup failed ({error}); skipped")
                        continue
                    # a query MusicBrainz cannot parse fails the same way every
                    # time, so it is cached rather than retried on every run
                    record = {"key": track.key, "status": "bad_query",
                              "query_artist": query_name(track.artist),
                              "query_title": track.title}
                except Exception as error:  # not cached, so retried on the next run
                    print(f"  {track.key}: lookup failed ({error}); skipped")
                    continue
                cache[track.key] = record
                matched += record["status"] == "ok"
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                if i % 50 == 0 or i == len(todo):
                    rate = i / (time.monotonic() - started)
                    print(f"  {i}/{len(todo)} looked up, {matched / i:.0%} matched, "
                          f"{(len(todo) - i) / rate / 60:.0f} min left")

    looked_up = [cache[t.key] for tracks in performers.values() for t in tracks
                 if t.key in cache]
    pairs = build_pairs(performers, cache, args.max_days, not args.allow_namesakes)
    by_relation = collections.Counter(p.relation for p in pairs)
    per_performer = collections.Counter(p.performer for p in pairs)
    summary = {
        "split": args.split,
        "performer_key": args.performer_key,
        "performers_considered": len(performers),
        "recordings_considered": sum(len(t) for t in performers.values()),
        "recordings_looked_up": len(looked_up),
        "recordings_matched": sum(r["status"] == "ok" for r in looked_up),
        "recordings_dated": sum(bool(r.get("date")) for r in looked_up),
        "recordings_with_album": sum(bool(r.get("album")) for r in looked_up),
        "pairs": len(pairs),
        "pairs_by_relation": dict(by_relation),
        "performers_with_pairs": len(per_performer),
        "pairs_per_performer_max": max(per_performer.values(), default=0),
        "max_days": args.max_days,
        "min_mb_score": args.min_mb_score,
    }
    with open(args.out / f"{args.split}.jsonl", "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_json(), ensure_ascii=False) + "\n")
    with open(args.out / f"{args.split}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out / f'{args.split}.jsonl'}")


if __name__ == "__main__":
    main()
