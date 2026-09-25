"""
Sounds of the Universe
A Streamlit application for open-ended astronomical search, scientific profiling,
and deterministic, provenance-aware sonification.

Initial deployment:
    streamlit run app.py

Recommended secrets:
    GEMINI_API_KEY = "..."
Optional model overrides:
    GEMINI_FLASH_MODEL = "gemini-3.6-flash"
    GEMINI_PRO_MODEL = "gemini-3.1-pro-preview"
    GEMINI_LITE_MODEL = "gemini-3.5-flash-lite"
Optional Search-and-Sonify web search keys (either is enough; if neither is
set, that pipeline falls back to this app's own astronomical resolvers
instead of fabricating "scraped" facts):
    TAVILY_API_KEY = "..."
    SERPER_API_KEY = "..."

The application deliberately does NOT claim that generated audio is the literal
sound of an astronomical object. It labels the basis as measured-data
sonification, physics-based model, or interpretive representation.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import re
import struct
import time
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import requests
import streamlit as st

try:
    from pydantic import BaseModel, Field
except Exception:
    BaseModel = None  # type: ignore
    Field = None  # type: ignore

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_NAME = "Sounds of the Universe"
APP_VERSION = "1.2.0"

# Requested Gemini model family. They can be overridden in Streamlit Secrets
# without changing application code.
GEMINI_FLASH_MODEL = "gemini-3.6-flash"
GEMINI_PRO_MODEL = "gemini-3.1-pro-preview"
GEMINI_LITE_MODEL = "gemini-3.5-flash-lite"

USER_AGENT = "SoundsOfTheUniverse/1.0 (+astronomical-sonification; Streamlit)"
HTTP_TIMEOUT = 12
MAX_SUGGESTIONS = 8
AUDIO_SAMPLE_RATE = 44_100
DEFAULT_DURATION = 12.0

SOURCE_SIMBAD = "SIMBAD / CDS"
SOURCE_SESAME = "CDS Sesame"
SOURCE_EXOPLANET = "NASA Exoplanet Archive"
SOURCE_JPL = "NASA/JPL Horizons"
SOURCE_GEMINI = "Gemini scientific synthesis"

OBJECT_TYPES = {
    "*": "Star",
    "V*": "Variable star",
    "Psr": "Pulsar",
    "G": "Galaxy",
    "GiC": "Globular cluster",
    "ClG": "Galaxy cluster",
    "PN": "Planetary nebula",
    "HII": "H II region",
    "SNR": "Supernova remnant",
    "Neb": "Nebula",
    "AGN": "Active galactic nucleus",
    "QSO": "Quasar",
    "BH": "Black-hole candidate",
    "Pl": "Planet",
    "Moo": "Moon",
    "As": "Asteroid",
    "Com": "Comet",
}


@dataclass
class SourceRecord:
    source: str
    field: str
    value: Any
    unit: str = ""
    status: str = "measured"
    uncertainty: Any = None
    retrieved_at: str = ""
    reference: str = ""


@dataclass
class ObjectProfile:
    query: str
    canonical_name: str
    object_type: str
    aliases: list[str] = field(default_factory=list)
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
    identifiers: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    sources: list[SourceRecord] = field(default_factory=list)
    source_notes: list[str] = field(default_factory=list)
    retrieval_errors: list[str] = field(default_factory=list)
    data_quality: str = "Limited evidence"
    audio_mode: str = "Interpretive representation"
    audio_basis: list[str] = field(default_factory=list)
    audio_explanation: str = ""
    limitations: list[str] = field(default_factory=list)
    retrieved_at: str = ""


@dataclass
class AudioResult:
    wav_bytes: bytes
    duration: float
    sample_rate: int
    seed: int
    mode: str
    mapping: list[dict[str, Any]]
    explanation: str
    cache_key: str


# ---------------------------------------------------------------------------
# Local astronomical index
# ---------------------------------------------------------------------------

# This is deliberately an accelerator, not the application's universe.
# Remote resolution remains available for arbitrary catalogue designations.
LOCAL_OBJECTS = [
    ("Mercury", "Planet · Solar System", "planet", ["Mercury"]),
    ("Venus", "Planet · Solar System", "planet", ["Venus"]),
    ("Earth", "Planet · Solar System", "planet", ["Earth"]),
    ("Mars", "Planet · Solar System", "planet", ["Mars"]),
    ("Jupiter", "Planet · Solar System", "planet", ["Jupiter"]),
    ("Saturn", "Planet · Solar System", "planet", ["Saturn"]),
    ("Uranus", "Planet · Solar System", "planet", ["Uranus"]),
    ("Neptune", "Planet · Solar System", "planet", ["Neptune"]),
    ("Moon", "Natural satellite · Earth", "moon", ["Moon", "Luna"]),
    ("Io", "Natural satellite · Jupiter", "moon", ["Io"]),
    ("Europa", "Natural satellite · Jupiter", "moon", ["Europa"]),
    ("Ganymede", "Natural satellite · Jupiter", "moon", ["Ganymede"]),
    ("Callisto", "Natural satellite · Jupiter", "moon", ["Callisto"]),
    ("Titan", "Natural satellite · Saturn", "moon", ["Titan"]),
    ("Enceladus", "Natural satellite · Saturn", "moon", ["Enceladus"]),
    ("Triton", "Natural satellite · Neptune", "moon", ["Triton"]),
    ("Pluto", "Dwarf planet · Kuiper belt", "dwarf planet", ["Pluto"]),
    ("Ceres", "Dwarf planet · Main belt", "dwarf planet", ["Ceres", "1 Ceres"]),
    ("Vesta", "Asteroid · Main belt", "asteroid", ["Vesta", "4 Vesta"]),
    ("Halley", "Comet · Solar System", "comet", ["Halley's Comet", "1P/Halley"]),
    ("Sirius", "Star · Alpha Canis Majoris", "star", ["Sirius", "Alpha Canis Majoris", "HD 48915"]),
    ("Vega", "Star · Alpha Lyrae", "star", ["Vega", "Alpha Lyrae", "HD 172167"]),
    ("Polaris", "Star · Alpha Ursae Minoris", "star", ["Polaris", "Alpha Ursae Minoris", "HD 8890"]),
    ("Betelgeuse", "Star · Alpha Orionis", "star", ["Betelgeuse", "Alpha Orionis", "HD 39801"]),
    ("Rigel", "Star · Beta Orionis", "star", ["Rigel", "Beta Orionis", "HD 34085"]),
    ("Antares", "Star · Alpha Scorpii", "star", ["Antares", "Alpha Scorpii", "HD 148478"]),
    ("Proxima Centauri", "Star · Alpha Centauri system", "star", ["Proxima Centauri", "HD 128621"]),
    ("Alpha Centauri", "Star system · Centaurus", "star system", ["Alpha Centauri", "Rigil Kentaurus"]),
    ("Barnard's Star", "Star · Red dwarf", "star", ["Barnard's Star", "HD 149084"]),
    ("Tau Ceti", "Star · G-type star", "star", ["Tau Ceti", "HD 10700"]),
    ("TRAPPIST-1", "Planetary system · Exoplanet host", "star", ["TRAPPIST-1", "2MASS J23062928-0502285"]),
    ("Kepler-22", "Planetary system · Exoplanet host", "star", ["Kepler-22"]),
    ("HD 209458", "Star · Exoplanet host", "star", ["HD 209458"]),
    ("WASP-12", "Star · Exoplanet host", "star", ["WASP-12"]),
    ("M31", "Andromeda Galaxy · Spiral galaxy", "galaxy", ["M31", "Messier 31", "NGC 224", "Andromeda Galaxy"]),
    ("M32", "Compact elliptical galaxy · Local Group", "galaxy", ["M32", "Messier 32"]),
    ("M33", "Triangulum Galaxy · Spiral galaxy", "galaxy", ["M33", "Messier 33", "NGC 598", "Triangulum Galaxy"]),
    ("M87", "Messier 87 · Giant elliptical galaxy", "galaxy", ["M87", "Messier 87", "Virgo A"]),
    ("NGC 1275", "Galaxy · Perseus cluster", "galaxy", ["NGC 1275", "Perseus A"]),
    ("NGC 224", "Galaxy · Andromeda", "galaxy", ["NGC 224", "M31"]),
    ("NGC 7000", "North America Nebula · Emission nebula", "nebula", ["NGC 7000"]),
    ("Crab Nebula", "Supernova remnant · M1", "supernova remnant", ["Crab Nebula", "M1", "Messier 1", "NGC 1952"]),
    ("Orion Nebula", "Emission nebula · M42", "nebula", ["Orion Nebula", "M42", "Messier 42", "NGC 1976"]),
    ("Ring Nebula", "Planetary nebula · M57", "planetary nebula", ["Ring Nebula", "M57", "Messier 57", "NGC 6720"]),
    ("Vela Supernova Remnant", "Supernova remnant · Vela", "supernova remnant", ["Vela SNR"]),
    ("Vela Pulsar", "Pulsar · PSR B0833−45", "pulsar", ["Vela Pulsar", "PSR B0833-45", "PSR J0835-4510"]),
    ("Crab Pulsar", "Pulsar · PSR B0531+21", "pulsar", ["Crab Pulsar", "PSR B0531+21", "PSR J0534+2200"]),
    ("PSR B1919+21", "Pulsar · first discovered radio pulsar", "pulsar", ["PSR B1919+21", "PSR J1921+2153"]),
    ("Sagittarius A*", "Supermassive black-hole candidate · Galactic Center", "black hole", ["Sagittarius A*", "Sgr A*", "Sgr A"]),
    ("Sagittarius B2", "Molecular cloud · Galactic Center", "molecular cloud", ["Sagittarius B2", "Sgr B2"]),
    ("Sagittarius Dwarf Spheroidal Galaxy", "Dwarf galaxy · Local Group", "galaxy", ["Sagittarius Dwarf Spheroidal Galaxy"]),
    ("TON 618", "Quasar · Extremely luminous AGN", "quasar", ["TON 618"]),
    ("3C 273", "Quasar · Active galactic nucleus", "quasar", ["3C 273"]),
    ("Cygnus X-1", "Black-hole binary candidate", "black-hole binary", ["Cygnus X-1", "Cyg X-1"]),
    ("M87*", "Supermassive black hole · M87 nucleus", "black hole", ["M87*", "M87 black hole"]),
    ("Kepler-22b", "Exoplanet · Kepler-22 system", "exoplanet", ["Kepler-22b"]),
    ("WASP-12b", "Exoplanet · Hot Jupiter", "exoplanet", ["WASP-12b"]),
    ("TRAPPIST-1 b", "Exoplanet · TRAPPIST-1 system", "exoplanet", ["TRAPPIST-1 b"]),
    ("TRAPPIST-1 c", "Exoplanet · TRAPPIST-1 system", "exoplanet", ["TRAPPIST-1 c"]),
    ("TRAPPIST-1 d", "Exoplanet · TRAPPIST-1 system", "exoplanet", ["TRAPPIST-1 d"]),
    ("TRAPPIST-1 e", "Exoplanet · TRAPPIST-1 system", "exoplanet", ["TRAPPIST-1 e"]),
    ("TRAPPIST-1 f", "Exoplanet · TRAPPIST-1 system", "exoplanet", ["TRAPPIST-1 f"]),
    ("TRAPPIST-1 g", "Exoplanet · TRAPPIST-1 system", "exoplanet", ["TRAPPIST-1 g"]),
    ("TRAPPIST-1 h", "Exoplanet · TRAPPIST-1 system", "exoplanet", ["TRAPPIST-1 h"]),
    ("M3", "Globular cluster · Messier 3", "globular cluster", ["M3", "Messier 3", "NGC 5272"]),
    ("M13", "Globular cluster · Messier 13", "globular cluster", ["M13", "Messier 13", "NGC 6205"]),
    ("M51", "Whirlpool Galaxy · interacting spiral galaxy", "galaxy", ["M51", "Messier 51", "NGC 5194"]),
    ("M81", "Bode's Galaxy · spiral galaxy", "galaxy", ["M81", "Messier 81", "NGC 3031"]),
    ("M82", "Cigar Galaxy · starburst galaxy", "galaxy", ["M82", "Messier 82", "NGC 3034"]),
    ("M104", "Sombrero Galaxy · spiral galaxy", "galaxy", ["M104", "Messier 104", "NGC 4594"]),
    ("M42", "Orion Nebula · emission nebula", "nebula", ["M42", "Messier 42", "NGC 1976"]),
    ("M57", "Ring Nebula · planetary nebula", "planetary nebula", ["M57", "Messier 57", "NGC 6720"]),
    ("M1", "Crab Nebula · supernova remnant", "supernova remnant", ["M1", "Messier 1", "NGC 1952"]),
]


def normalize_query(value: str) -> str:
    value = (value or "").strip().lower()
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = re.sub(r"\s+", " ", value)
    return value


def local_suggestions(query: str, limit: int = MAX_SUGGESTIONS) -> list[dict[str, Any]]:
    q = normalize_query(query)
    if not q:
        return []

    results: list[dict[str, Any]] = []
    for name, context, kind, aliases in LOCAL_OBJECTS:
        haystack = " ".join([name, context, *aliases]).lower()
        score = None
        if name.lower().startswith(q):
            score = 0
        elif any(a.lower().startswith(q) for a in aliases):
            score = 1
        elif q in haystack:
            score = 2
        if score is not None:
            results.append(
                {
                    "name": name,
                    "context": context,
                    "kind": kind,
                    "aliases": aliases,
                    "_score": score,
                }
            )

    results.sort(key=lambda x: (x["_score"], len(x["name"]), x["name"].lower()))
    for item in results:
        item.pop("_score", None)
    return results[:limit]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_get_json(url: str, params: Optional[dict[str, Any]] = None) -> Any:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json,text/plain,*/*"}
    response = requests.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    return response.json()


def http_get_text(url: str, params: Optional[dict[str, Any]] = None) -> str:
    headers = {"User-Agent": USER_AGENT, "Accept": "text/plain,*/*"}
    response = requests.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    return response.text


# ---------------------------------------------------------------------------
# CDS Sesame / SIMBAD resolution
# ---------------------------------------------------------------------------

def parse_sesame_text(text: str, query: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "query": query,
        "canonical_name": query,
        "object_type": "Astronomical object",
        "aliases": [],
        "ra_deg": None,
        "dec_deg": None,
        "sources": [],
        "raw": text,
    }

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("%J "):
            parts = line[3:].split()
            if len(parts) >= 2:
                try:
                    result["ra_deg"] = float(parts[0])
                    result["dec_deg"] = float(parts[1])
                except ValueError:
                    pass
        elif line.startswith("%I "):
            ident = line[3:].strip()
            if ident and ident not in result["aliases"]:
                result["aliases"].append(ident)
        elif line.startswith("%I.0 "):
            result["canonical_name"] = line[5:].strip()
        elif line.startswith("%C."):
            value = line.split(None, 1)[-1].strip()
            result["object_type_code"] = value
            result["object_type"] = OBJECT_TYPES.get(value, value or "Astronomical object")
        elif line.startswith("%S "):
            value = line[3:].strip()
            if value:
                result["spectral_type"] = value.split()[0]

    result["aliases"] = result["aliases"][:80]
    return result


@st.cache_data(ttl=900, show_spinner=False)
def sesame_resolve(query: str) -> Optional[dict[str, Any]]:
    query = query.strip()
    if not query:
        return None

    # All-resolver text format is intentionally used here because it is compact,
    # stable, and can expose SIMBAD/NED/VizieR-derived identifiers.
    url = "https://cds.unistra.fr/cgi-bin/nph-sesame/-oI/NSV"
    try:
        text = http_get_text(url, params={"": query})
        parsed = parse_sesame_text(text, query)
        if parsed.get("aliases") or parsed.get("ra_deg") is not None:
            return parsed
    except Exception:
        pass

    # Fallback endpoint form.
    try:
        url = "https://cds.unistra.fr/cgi-bin/nph-sesame/-oI/NSV"
        response = requests.get(
            f"{url}?{requests.utils.quote(query)}",
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
        )
        response.raise_for_status()
        parsed = parse_sesame_text(response.text, query)
        if parsed.get("aliases") or parsed.get("ra_deg") is not None:
            return parsed
    except Exception:
        return None
    return None


@st.cache_data(ttl=900, show_spinner=False)
def simbad_prefix_search(prefix: str, limit: int = MAX_SUGGESTIONS) -> list[dict[str, Any]]:
    """
    Best-effort remote autocomplete for catalogue-style prefixes.

    SIMBAD supports wildcard identifier queries, but a one-character query can
    be enormous. We therefore use the local accelerator for the first character
    and query SIMBAD remotely for more specific catalogue prefixes.
    """
    q = prefix.strip()
    if len(q) < 2:
        return []

    upper = q.upper()
    patterns = []
    if re.match(r"^(M|NGC|IC|HD|HIP|PSR|2MASS|TYC|BD|GJ|WISE|TOI|KIC|KOI)\b", upper):
        patterns.append(q + "*")
    elif re.match(r"^(M|NGC|IC|HD|HIP|PSR|KIC|KOI|TOI)[0-9A-Z-]*$", upper):
        patterns.append(q + "*")

    if not patterns:
        return []

    script_url = "https://simbad.cds.unistra.fr/simbad/sim-script"
    # The script service accepts a SIMBAD script. We ask for identifiers only.
    pattern = patterns[0].replace("'", "''")
    script = (
        "format object form1 \"%IDLIST[1] | %OTYPELIST | %MAIN_ID\"\\n"
        f"query id {pattern}\\n"
    )
    try:
        response = requests.post(
            script_url,
            data={"submit": "submit script", "script": script},
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
        )
        response.raise_for_status()
        text = response.text
        suggestions = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("::") or line.startswith("Script"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                name = parts[-1]
                if name and name.lower() not in {x["name"].lower() for x in suggestions}:
                    suggestions.append(
                        {
                            "name": name,
                            "context": parts[-2] or "Astronomical object",
                            "kind": "catalogue object",
                            "aliases": [parts[0]] if parts[0] else [],
                        }
                    )
            if len(suggestions) >= limit:
                break
        return suggestions[:limit]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# NASA Exoplanet Archive
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def exoplanet_lookup(query: str) -> list[dict[str, Any]]:
    q = query.strip().replace("'", "''")
    if not q:
        return []

    # Use a compact set of fields. The archive's TAP service is public and
    # supports ADQL. We try exact planet name, then host name.
    columns = (
        "pl_name,hostname,discoverymethod,disc_year,pl_orbper,pl_rade,pl_radj,"
        "pl_bmasse,pl_bmassj,pl_eqt,pl_insol,pl_orbsmax,pl_orbeccen,"
        "st_spectype,st_teff,st_rad,st_mass,ra,dec"
    )
    queries = [
        f"select top 5 {columns} from pscomppars where lower(pl_name)=lower('{q}')",
        f"select top 5 {columns} from pscomppars where lower(hostname)=lower('{q}')",
    ]
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    for adql in queries:
        try:
            data = http_get_json(url, params={"query": adql, "format": "json"})
            if isinstance(data, list) and data:
                return data
        except Exception:
            continue
    return []


# ---------------------------------------------------------------------------
# NASA/JPL Horizons
# ---------------------------------------------------------------------------

HORIZONS_COMMANDS = {
    "Mercury": "199",
    "Venus": "299",
    "Earth": "399",
    "Mars": "499",
    "Jupiter": "599",
    "Saturn": "699",
    "Uranus": "799",
    "Neptune": "899",
    "Pluto": "999",
    "Moon": "301",
    "Io": "501",
    "Europa": "502",
    "Ganymede": "503",
    "Callisto": "504",
    "Titan": "606",
    "Enceladus": "602",
    "Triton": "801",
    "Ceres": "1",
    "Vesta": "4",
}


@st.cache_data(ttl=3600, show_spinner=False)
def horizons_body_data(name: str) -> Optional[dict[str, Any]]:
    command = HORIZONS_COMMANDS.get(name)
    if not command:
        return None

    url = "https://ssd.jpl.nasa.gov/api/horizons.api"
    params = {
        "format": "json",
        "COMMAND": f"'{command}'",
        "OBJ_DATA": "YES",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "OBSERVER",
        "CENTER": "500@399",
        "START_TIME": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "STOP_TIME": (datetime.now(timezone.utc)).strftime("%Y-%m-%d"),
        "STEP_SIZE": "1 d",
        "QUANTITIES": "9,20,23,24,29",
    }
    try:
        data = http_get_json(url, params=params)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Scientific profile construction
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_source(
    profile: ObjectProfile,
    source: str,
    field_name: str,
    value: Any,
    unit: str = "",
    status: str = "measured",
    uncertainty: Any = None,
    reference: str = "",
) -> None:
    if value is None or value == "":
        return
    profile.sources.append(
        SourceRecord(
            source=source,
            field=field_name,
            value=value,
            unit=unit,
            status=status,
            uncertainty=uncertainty,
            retrieved_at=profile.retrieved_at,
            reference=reference,
        )
    )
    profile.properties[field_name] = value


def infer_audio_mode(profile: ObjectProfile) -> str:
    kind = profile.object_type.lower()
    if "pulsar" in kind and any(s.field in {"period_s", "pulse_period"} for s in profile.sources):
        return "Measured-data sonification"
    if profile.sources:
        return "Physics-based auditory model"
    return "Interpretive representation"


def build_profile(query: str) -> ObjectProfile:
    retrieved_at = now_iso()
    profile = ObjectProfile(
        query=query,
        canonical_name=query.strip(),
        object_type="Astronomical object",
        retrieved_at=retrieved_at,
    )

    local = local_suggestions(query, 1)
    if local and normalize_query(local[0]["name"]) == normalize_query(query):
        profile.canonical_name = local[0]["name"]
        profile.object_type = local[0]["context"].split("·")[0].strip()
        profile.aliases = local[0]["aliases"][:]

    sesame = sesame_resolve(query)
    if sesame:
        profile.canonical_name = sesame.get("canonical_name") or profile.canonical_name
        profile.object_type = sesame.get("object_type") or profile.object_type
        profile.aliases = list(dict.fromkeys((profile.aliases or []) + sesame.get("aliases", [])))
        profile.ra_deg = sesame.get("ra_deg")
        profile.dec_deg = sesame.get("dec_deg")
        if profile.ra_deg is not None:
            add_source(profile, SOURCE_SESAME, "right_ascension_deg", profile.ra_deg, "deg")
        if profile.dec_deg is not None:
            add_source(profile, SOURCE_SESAME, "declination_deg", profile.dec_deg, "deg")
        if sesame.get("spectral_type"):
            add_source(profile, SOURCE_SIMBAD, "spectral_type", sesame["spectral_type"], "classification")

    # Exoplanet enrichment.
    exo_rows = exoplanet_lookup(query)
    if not exo_rows:
        # Also try the resolved canonical name and exact aliases.
        for candidate in [profile.canonical_name, *profile.aliases[:5]]:
            exo_rows = exoplanet_lookup(candidate)
            if exo_rows:
                break

    if exo_rows:
        row = exo_rows[0]
        profile.object_type = "Exoplanet" if row.get("pl_name") else profile.object_type
        profile.canonical_name = row.get("pl_name") or profile.canonical_name
        for key, unit in [
            ("pl_orbper", "days"),
            ("pl_rade", "Earth radii"),
            ("pl_radj", "Jupiter radii"),
            ("pl_bmasse", "Earth masses"),
            ("pl_bmassj", "Jupiter masses"),
            ("pl_eqt", "K"),
            ("pl_insol", "Earth flux"),
            ("pl_orbsmax", "AU"),
            ("pl_orbeccen", ""),
            ("st_teff", "K"),
            ("st_rad", "Solar radii"),
            ("st_mass", "Solar masses"),
            ("ra", "deg"),
            ("dec", "deg"),
        ]:
            value = row.get(key)
            if value is not None:
                add_source(profile, SOURCE_EXOPLANET, key, value, unit, "measured_or_catalogued")
        if row.get("discoverymethod"):
            add_source(profile, SOURCE_EXOPLANET, "discovery_method", row["discoverymethod"], "classification")
        if row.get("st_spectype"):
            add_source(profile, SOURCE_EXOPLANET, "host_spectral_type", row["st_spectype"], "classification")

    # Solar-system enrichment via JPL Horizons.
    horizon = horizons_body_data(profile.canonical_name)
    if horizon:
        profile.source_notes.append("JPL Horizons returned current Solar System ephemeris/object data.")
        # Horizons' main body metadata is returned as text inside result. We keep
        # it as provenance rather than pretending to parse every possible field.
        result_text = str(horizon.get("result", ""))
        if result_text:
            profile.properties["jpl_horizons_available"] = True
            add_source(
                profile,
                SOURCE_JPL,
                "horizons_object_record",
                result_text[:3000],
                "text",
                "catalogue/ephemeris",
            )

    if not profile.sources:
        profile.retrieval_errors.append(
            "No structured scientific values were returned by the current resolvers."
        )

    profile.identifiers = list(dict.fromkeys(profile.aliases[:50]))
    profile.audio_mode = infer_audio_mode(profile)

    if len(profile.sources) >= 8:
        profile.data_quality = "High confidence"
    elif len(profile.sources) >= 3:
        profile.data_quality = "Moderate confidence"
    else:
        profile.data_quality = "Limited evidence"

    profile.audio_basis = choose_audio_basis(profile)
    profile.audio_explanation = build_rule_based_audio_explanation(profile)
    return profile


def choose_audio_basis(profile: ObjectProfile) -> list[str]:
    kind = profile.object_type.lower()
    props = profile.properties
    basis: list[str] = []

    if "pulsar" in kind:
        if any(k in props for k in ("period_s", "pulse_period")):
            basis.append("Observed periodic timing")
        basis.append("Pulse timing preserved within an audible time scale")
    elif "planet" in kind or "moon" in kind:
        if "pl_eqt" in props:
            basis.append("Catalogued equilibrium temperature")
        if "pl_orbper" in props:
            basis.append("Catalogued orbital period")
        if "horizons_object_record" in props:
            basis.append("NASA/JPL Horizons object/ephemeris context")
        basis.append("Object-class-specific physical parameter mapping")
    elif "star" in kind:
        if "spectral_type" in props:
            basis.append("Stellar spectral classification")
        if "st_teff" in props:
            basis.append("Catalogued effective temperature")
        basis.append("Stellar-parameter auditory mapping")
    elif "galaxy" in kind or "quasar" in kind or "active" in kind:
        basis.append("Resolved extragalactic object classification")
        if profile.ra_deg is not None:
            basis.append("Sky-position identity")
        basis.append("Extragalactic parameter mapping where measurements are available")
    elif "nebula" in kind or "supernova" in kind:
        basis.append("Resolved nebular/supernova-remnant classification")
        basis.append("Measured parameters where available")
    elif "black hole" in kind:
        basis.append("Resolved black-hole identity and available observational/model context")
        basis.append("No claim of literal black-hole sound")
    else:
        basis.append("Resolved astronomical identity")
        basis.append("Available measured/catalogued parameters")

    return basis


# ---------------------------------------------------------------------------
# Gemini structured synthesis
# ---------------------------------------------------------------------------

if BaseModel is not None:
    class GeminiAudioPlan(BaseModel):
        primary_phenomenon: str = Field(description="The most defensible scientific feature to emphasize.")
        secondary_phenomena: list[str] = Field(default_factory=list)
        data_basis: str = Field(description="Measured, catalogued, modeled, or interpretive basis.")
        recommended_mapping: str = Field(description="Human-readable mapping recommendation; do not invent numerical facts.")
        confidence: str = Field(description="High confidence, Moderate confidence, or Limited evidence.")
        limitations: str = Field(description="Important scientific limitations.")
        listener_focus: str = Field(description="What the listener should notice.")
else:
    GeminiAudioPlan = None  # type: ignore


def get_gemini_client():
    if genai is None:
        return None
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None
    if not key:
        import os
        key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    try:
        return genai.Client(api_key=key)
    except Exception:
        return None


def model_secret(name: str, default: str) -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def gemini_plan(profile: ObjectProfile) -> Optional[GeminiAudioPlan]:
    client = get_gemini_client()
    if client is None or GeminiAudioPlan is None:
        return None

    payload = {
        "canonical_name": profile.canonical_name,
        "object_type": profile.object_type,
        "coordinates": {"ra_deg": profile.ra_deg, "dec_deg": profile.dec_deg},
        "properties": profile.properties,
        "sources": [asdict(s) for s in profile.sources],
    }
    prompt = f"""
You are the scientific interpretation layer for an astronomical sonification
application.

Use ONLY the supplied structured data. Never invent a numerical measurement,
atmosphere, magnetic field, composition, temperature, mass, rotation rate,
period, or observational result that is absent from the data.

The audio engine is deterministic and Python-controlled. You are NOT generating
audio and must not request arbitrary effects.

Classify the strongest defensible scientific basis for an auditory
representation, suggest relationships between already-present data and audio,
and state limitations. Distinguish measured/catalogued information from model
or interpretation. Never call generated audio "the actual sound of" an object.

OBJECT DATA:
{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}
"""
    try:
        response = client.models.generate_content(
            model=st.session_state.get("gemini_model", model_secret("GEMINI_FLASH_MODEL", GEMINI_FLASH_MODEL)),
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GeminiAudioPlan,
                temperature=0.0,
                max_output_tokens=900,
            ),
        )
        if getattr(response, "parsed", None):
            return response.parsed
        text = getattr(response, "text", "") or ""
        if text:
            return GeminiAudioPlan.model_validate_json(text)
    except Exception:
        return None
    return None


def gemini_explanation(profile: ObjectProfile) -> str:
    client = get_gemini_client()
    if client is None:
        return profile.audio_explanation

    payload = {
        "object": profile.canonical_name,
        "type": profile.object_type,
        "properties": profile.properties,
        "sources": [asdict(s) for s in profile.sources],
        "audio_mode": profile.audio_mode,
        "audio_basis": profile.audio_basis,
    }
    prompt = f"""
Explain this astronomical sonification to a general audience, especially a
listener who may rely primarily on hearing.

Strict rules:
- Use only supplied data.
- Do not invent measurements.
- Do not describe the result as literal sound from the object.
- Explicitly identify measured/catalogued vs modeled vs interpretive elements.
- Explain any frequency shifting or time scaling in plain language.
- Keep it scientifically cautious and accessible.

DATA:
{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}
"""
    try:
        response = client.models.generate_content(
            model=st.session_state.get("gemini_model", model_secret("GEMINI_FLASH_MODEL", GEMINI_FLASH_MODEL)),
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1000,
            ),
        )
        text = getattr(response, "text", "") or ""
        return text.strip() or profile.audio_explanation
    except Exception:
        return profile.audio_explanation


# ---------------------------------------------------------------------------
# Search-and-Sonify: live web data ingestion layer (BVI tactile pipeline)
# ---------------------------------------------------------------------------

_FACT_PATTERNS = {
    "estimated_mass": re.compile(
        r"mass[^.\n]{0,40}?([\d][\d.,]*\s?(?:solar masses|earth masses|jupiter masses|kg|kilograms|tons))",
        re.IGNORECASE,
    ),
    "radius": re.compile(
        r"radius[^.\n]{0,40}?([\d][\d.,]*\s?(?:km|kilometers|solar radii|earth radii|light[- ]years|au|miles))",
        re.IGNORECASE,
    ),
    "temperature": re.compile(
        r"temperature[^.\n]{0,40}?([\d][\d.,]*\s?(?:k\b|kelvin|°c|celsius|°f|fahrenheit))",
        re.IGNORECASE,
    ),
    "explosion_velocity": re.compile(
        r"(?:explosion|ejecta|outflow|expansion)[^.\n]{0,25}?velocity[^.\n]{0,40}?"
        r"([\d][\d.,]*\s?(?:km/s|kilometers per second|m/s|mph))",
        re.IGNORECASE,
    ),
}

_OBJECT_TYPE_KEYWORDS = [
    "black hole", "neutron star", "pulsar", "supernova", "nebula", "exoplanet",
    "quasar", "galaxy", "star cluster", "white dwarf", "red giant", "protostar", "star",
]


def _extract_physical_facts(text: str) -> dict[str, Optional[str]]:
    facts: dict[str, Optional[str]] = {
        "estimated_mass": None, "radius": None, "temperature": None,
        "explosion_velocity": None, "object_type": None,
    }
    if not text:
        return facts
    for key, pattern in _FACT_PATTERNS.items():
        match = pattern.search(text)
        if match:
            facts[key] = match.group(1).strip()
    lowered = text.lower()
    for keyword in _OBJECT_TYPE_KEYWORDS:
        if keyword in lowered:
            facts["object_type"] = keyword.title()
            break
    return facts


def _search_api_key(name: str) -> Optional[str]:
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    if not value:
        import os
        value = os.environ.get(name)
    return value or None


@st.cache_data(ttl=1800, show_spinner=False)
def _tavily_search(query: str, api_key: str) -> list[dict[str, str]]:
    response = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": api_key, "query": f"{query} astronomy mass radius temperature", "max_results": 5},
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    return [
        {"title": r.get("title", ""), "snippet": r.get("content", ""), "url": r.get("url", "")}
        for r in data.get("results", [])
    ]


@st.cache_data(ttl=1800, show_spinner=False)
def _serper_search(query: str, api_key: str) -> list[dict[str, str]]:
    response = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": f"{query} astronomy mass radius temperature"},
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    return [
        {"title": r.get("title", ""), "snippet": r.get("snippet", ""), "url": r.get("link", "")}
        for r in data.get("organic", [])
    ]


def fetch_live_web_data(query: str) -> dict[str, Any]:
    """
    Data ingestion / scraping layer for the Search-and-Sonify pipeline.

    Tries a real search API first (Tavily, then Serper) if a key is present in
    secrets/environment, and extracts candidate physical facts (mass, radius,
    temperature, explosion velocity, object type) from the returned snippets.

    If no search key is configured, or a search returns nothing, this falls
    back to the application's own astronomical resolvers (SIMBAD/Sesame, NASA
    Exoplanet Archive, JPL Horizons) rather than simulating fake web content,
    so the pipeline degrades safely instead of inventing numbers.
    """
    query = (query or "").strip()
    result: dict[str, Any] = {
        "query": query, "found": False, "source": None, "object_type": None,
        "estimated_mass": None, "radius": None, "temperature": None,
        "explosion_velocity": None, "raw_snippets": [],
    }
    if not query:
        return result

    snippets: list[dict[str, str]] = []
    source = None
    try:
        tavily_key = _search_api_key("TAVILY_API_KEY")
        serper_key = _search_api_key("SERPER_API_KEY")
        if tavily_key:
            snippets = _tavily_search(query, tavily_key)
            source = "tavily"
        elif serper_key:
            snippets = _serper_search(query, serper_key)
            source = "serper"
    except Exception:
        snippets = []

    if snippets:
        combined = " ".join(f"{s.get('title', '')}. {s.get('snippet', '')}" for s in snippets)
        result.update(_extract_physical_facts(combined))
        result["raw_snippets"] = snippets[:5]
        result["source"] = source
        result["found"] = True
        return result

    # Structural fallback: reuse the app's own real astronomical resolvers
    # instead of simulating fake web content.
    try:
        profile = build_profile(query)
    except Exception:
        profile = None
    if profile and profile.sources:
        props = profile.properties
        result["object_type"] = profile.object_type
        result["estimated_mass"] = props.get("pl_bmasse") or props.get("st_mass")
        result["radius"] = props.get("pl_rade") or props.get("st_rad")
        result["temperature"] = props.get("pl_eqt") or props.get("st_teff")
        result["raw_snippets"] = [
            {"title": s.source, "snippet": f"{s.field}: {s.value} {s.unit}".strip(), "url": ""}
            for s in profile.sources[:6]
        ]
        result["source"] = "internal_astronomy_resolvers"
        result["found"] = True
    return result


# ---------------------------------------------------------------------------
# Search-and-Sonify: Gemini tactile sound-design layer
# ---------------------------------------------------------------------------

TACTILE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "object_name": {"type": "string"},
        "psychological_concept": {"type": "string"},
        "audio_mapping_parameters": {
            "type": "object",
            "properties": {
                "sub_bass_frequency_hz": {"type": "number"},
                "time_dilation_lfo_hz": {"type": "number"},
                "granular_density": {"type": "number"},
                "granular_grain_size_ms": {"type": "number"},
                "acoustic_grit_factor": {"type": "number"},
                "ambient_noise_smoothing": {"type": "number"},
                "atmospheric_dampening_coefficient": {"type": "number"},
                "fm_modulation_index": {"type": "number"},
            },
            "required": [
                "sub_bass_frequency_hz", "time_dilation_lfo_hz", "granular_density",
                "granular_grain_size_ms", "acoustic_grit_factor", "ambient_noise_smoothing",
                "atmospheric_dampening_coefficient", "fm_modulation_index",
            ],
        },
    },
    "required": ["object_name", "psychological_concept", "audio_mapping_parameters"],
}

TACTILE_SYSTEM_INSTRUCTION = """
You are a Cognitive Somatic Sound Designer for a real-time accessibility
application used primarily by blind and visually impaired listeners.

Given best-effort web/catalogue facts about an astronomical object, design a
tactile, embodied audio mapping: an interpretive sound-design translation of
the object's known character into the audio engine's control parameters. This
is a creative sound-design choice, not a new scientific measurement, so ground
the psychological_concept and parameter choices in whatever real facts were
supplied (object type, mass, radius, temperature, velocity) without asserting
invented physical measurements of your own.

Respond ONLY with a raw JSON object matching the required schema. Do not wrap
the JSON in Markdown code fences and do not include any other text.
"""


def gemini_tactile_plan(query: str, live_data: dict[str, Any], model: str) -> Optional[dict[str, Any]]:
    client = get_gemini_client()
    if client is None or types is None:
        return None

    prompt = f"""
OBJECT SEARCHED: {query}

SCRAPED / RESOLVED DATA (source: {live_data.get('source') or 'none'}):
{json.dumps(live_data, ensure_ascii=False, indent=2, default=str)}
"""
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=TACTILE_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=TACTILE_JSON_SCHEMA,
                temperature=0.85,
                max_output_tokens=600,
            ),
        )
        text = (getattr(response, "text", "") or "").strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        if text:
            return json.loads(text)
    except Exception:
        return None
    return None


def fallback_tactile_plan(query: str, live_data: dict[str, Any]) -> dict[str, Any]:
    """Deterministic fallback used when Gemini is unavailable or fails, so the
    Search-and-Sonify pipeline degrades safely instead of breaking."""
    seed = int(hashlib.sha256((query or "unknown").encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    kind = (live_data.get("object_type") or "").lower()

    if "black hole" in kind:
        concept = "The crushing weight and inescapable pull of a gravitational abyss."
        base = dict(sub_bass_frequency_hz=28, time_dilation_lfo_hz=0.12, granular_density=0.2,
                    granular_grain_size_ms=40, acoustic_grit_factor=0.25, ambient_noise_smoothing=260,
                    atmospheric_dampening_coefficient=0.75, fm_modulation_index=3.0)
    elif "supernova" in kind:
        concept = "A violent, expanding burst of kinetic debris tearing outward."
        base = dict(sub_bass_frequency_hz=40, time_dilation_lfo_hz=0.6, granular_density=0.9,
                    granular_grain_size_ms=12, acoustic_grit_factor=0.85, ambient_noise_smoothing=60,
                    atmospheric_dampening_coefficient=0.1, fm_modulation_index=9.0)
    elif "nebula" in kind:
        concept = "A slow, cold drift through a vast and diffuse cloud."
        base = dict(sub_bass_frequency_hz=24, time_dilation_lfo_hz=0.2, granular_density=0.35,
                    granular_grain_size_ms=60, acoustic_grit_factor=0.2, ambient_noise_smoothing=320,
                    atmospheric_dampening_coefficient=0.5, fm_modulation_index=2.0)
    else:
        concept = f"A steady, controlled presence representing {query.strip() or 'this object'}."
        base = dict(sub_bass_frequency_hz=32, time_dilation_lfo_hz=0.3, granular_density=0.4,
                    granular_grain_size_ms=25, acoustic_grit_factor=0.35, ambient_noise_smoothing=180,
                    atmospheric_dampening_coefficient=0.3, fm_modulation_index=5.0)

    params = {}
    for key, value in base.items():
        params[key] = float(value * (1.0 + rng.uniform(-0.08, 0.08)))

    return {
        "object_name": query.strip() or "Unknown object",
        "psychological_concept": concept,
        "audio_mapping_parameters": params,
    }


def run_search_and_sonify(query: str, model: str, duration: float) -> dict[str, Any]:
    """The full 3-step Search-and-Sonify pipeline: web/data ingestion -> Gemini
    structured tactile plan -> real-time psychoacoustic rendering."""
    live_data = fetch_live_web_data(query)
    plan = gemini_tactile_plan(query, live_data, model)
    used_fallback = plan is None
    if plan is None:
        plan = fallback_tactile_plan(query, live_data)
    audio = synthesize_tactile_audio(plan, duration)
    return {"query": query, "live_data": live_data, "plan": plan, "audio": audio, "used_fallback": used_fallback}


# ---------------------------------------------------------------------------
# Deterministic sonification engine
# ---------------------------------------------------------------------------

def stable_seed(profile: ObjectProfile) -> int:
    canonical = json.dumps(
        {
            "name": profile.canonical_name,
            "type": profile.object_type,
            "properties": profile.properties,
            "sources": [asdict(s) for s in profile.sources],
            "app_version": APP_VERSION,
        },
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0xFFFFFFFF


def clamp(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


def normalized_number(value: Any, low: float, high: float, default: float = 0.5) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    if high == low:
        return default
    # Log scale is more useful for wide astronomical ranges.
    if low > 0 and high > low and x > 0:
        lx, llo, lhi = math.log10(x), math.log10(low), math.log10(high)
        return clamp((lx - llo) / (lhi - llo), 0.0, 1.0)
    return clamp((x - low) / (high - low), 0.0, 1.0)


def sine_layer(t: np.ndarray, freq: float, amp: float, phase: float = 0.0) -> np.ndarray:
    return amp * np.sin(2.0 * np.pi * freq * t + phase)


def band_limited_noise(rng: np.random.Generator, n: int, smooth: int = 120) -> np.ndarray:
    raw = rng.normal(0.0, 1.0, n)
    if smooth <= 1:
        return raw
    kernel = np.ones(smooth, dtype=np.float64) / smooth
    smooth_noise = np.convolve(raw, kernel, mode="same")
    smooth_noise /= max(np.max(np.abs(smooth_noise)), 1e-9)
    return smooth_noise


def make_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    samples = np.asarray(samples, dtype=np.float64)
    samples = np.nan_to_num(samples)
    peak = float(np.max(np.abs(samples))) if samples.size else 1.0
    if peak > 0:
        samples = samples / peak
    # Conservative peak level: -3 dBFS.
    samples = samples * 0.707
    pcm = np.int16(np.clip(samples, -1.0, 1.0) * 32767)
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return out.getvalue()


def synthesize_audio(
    profile: ObjectProfile,
    duration: float = DEFAULT_DURATION,
    intensity: float = 0.65,
) -> AudioResult:
    duration = clamp(float(duration), 4.0, 30.0)
    intensity = clamp(float(intensity), 0.15, 1.0)
    sr = AUDIO_SAMPLE_RATE
    n = int(duration * sr)
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(stable_seed(profile))
    props = profile.properties
    kind = profile.object_type.lower()

    audio = np.zeros(n, dtype=np.float64)
    mapping: list[dict[str, Any]] = []

    # Stable envelope avoids clicks and keeps playback controlled.
    attack = np.minimum(1.0, t / 0.12)
    release = np.minimum(1.0, (duration - t) / 0.25)
    envelope = np.clip(np.minimum(attack, release), 0.0, 1.0)

    # Pulsars: preserve a documented periodicity if available. Since the
    # current public profile may not expose a period, do not invent one.
    if "pulsar" in kind:
        period = props.get("period_s") or props.get("pulse_period")
        if period is not None:
            p = float(period)
            audible_period = clamp(0.25 + math.log10(max(p, 1e-6)) + 1.0, 0.25, 2.0)
            phase = (t % audible_period) / audible_period
            pulse = np.exp(-0.5 * ((phase - 0.12) / 0.035) ** 2)
            carrier = sine_layer(t, 220.0, 0.55) + sine_layer(t, 440.0, 0.16)
            audio += pulse * carrier
            mapping.append(
                {
                    "layer": "Periodic pulse",
                    "source_parameter": "period_s / pulse_period",
                    "relationship": "Original pulse timing compressed/shifted into an audible rhythmic interval.",
                    "status": "Measured-data sonification",
                }
            )
        else:
            # No invented pulsar period. Use a restrained deterministic marker
            # based only on the object's resolved identity hash.
            base = 180.0 + (stable_seed(profile) % 120)
            audio += sine_layer(t, base, 0.34)
            mapping.append(
                {
                    "layer": "Identity carrier",
                    "source_parameter": "Resolved object identity",
                    "relationship": "No measured pulse timing was available in the current profile; no pulse rate is fabricated.",
                    "status": "Interpretive representation",
                }
            )

    elif "planet" in kind or "moon" in kind or "dwarf planet" in kind:
        eqt = props.get("pl_eqt")
        orbper = props.get("pl_orbper")
        radius = props.get("pl_rade") or props.get("pl_radj")
        base = 90.0
        if eqt is not None:
            temp_norm = normalized_number(eqt, 50, 3000, 0.45)
            base = 70.0 + 220.0 * temp_norm
            audio += sine_layer(t, base, 0.36)
            mapping.append(
                {
                    "layer": "Primary physical carrier",
                    "source_parameter": "pl_eqt",
                    "relationship": "Equilibrium temperature controls carrier frequency within an audible range; it is a mapping, not a literal acoustic frequency.",
                    "status": "Catalogued-data mapping",
                }
            )
        else:
            audio += sine_layer(t, base, 0.25)
            mapping.append(
                {
                    "layer": "Physical-profile carrier",
                    "source_parameter": "Object class",
                    "relationship": "No temperature measurement was available; no temperature value is fabricated.",
                    "status": "Physics-informed fallback",
                }
            )

        if orbper is not None:
            period_norm = normalized_number(orbper, 0.1, 1000, 0.5)
            mod_rate = 0.08 + 0.35 * period_norm
            mod = 0.5 * (1.0 + np.sin(2 * np.pi * mod_rate * t))
            audio += mod * sine_layer(t, base * 2.0, 0.18)
            mapping.append(
                {
                    "layer": "Orbital modulation",
                    "source_parameter": "pl_orbper",
                    "relationship": "Catalogued orbital period is compressed into a slow audible modulation.",
                    "status": "Catalogued-data mapping",
                }
            )

        if radius is not None:
            rnorm = normalized_number(radius, 0.1, 20, 0.5)
            texture = band_limited_noise(rng, n, int(90 - 60 * rnorm))
            audio += texture * (0.045 + 0.07 * rnorm)
            mapping.append(
                {
                    "layer": "Low-level texture",
                    "source_parameter": "planetary radius",
                    "relationship": "Size influences texture amplitude only; it is not presented as a natural sound property.",
                    "status": "Interpretive mapping",
                }
            )

    elif "star" in kind:
        teff = props.get("st_teff")
        if teff is not None:
            temp_norm = normalized_number(teff, 2500, 50000, 0.5)
            fundamental = 140.0 + 300.0 * temp_norm
            audio += sine_layer(t, fundamental, 0.32)
            audio += sine_layer(t, fundamental * 2.0, 0.12)
            mapping.append(
                {
                    "layer": "Spectral-temperature carrier",
                    "source_parameter": "st_teff",
                    "relationship": "Effective temperature is mapped to a controlled audible carrier; it is not the star's acoustic frequency.",
                    "status": "Catalogued-data mapping",
                }
            )
        else:
            spectral = str(props.get("spectral_type", ""))
            fundamental = 180.0 + (sum(ord(c) for c in spectral) % 160)
            audio += sine_layer(t, fundamental, 0.30)
            mapping.append(
                {
                    "layer": "Spectral-class carrier",
                    "source_parameter": "spectral_type",
                    "relationship": "Spectral classification selects a reproducible carrier family.",
                    "status": "Classification-based model",
                }
            )

        # Gentle variability texture. It is not claimed to be measured
        # variability unless an actual time-series value exists.
        texture = band_limited_noise(rng, n, 180)
        audio += texture * 0.045
        mapping.append(
            {
                "layer": "Stellar texture",
                "source_parameter": "Deterministic model texture",
                "relationship": "Low-level texture prevents a sterile tone; it is explicitly modeled rather than observed.",
                "status": "Physics-informed model",
            }
        )

    elif "galaxy" in kind or "quasar" in kind or "active" in kind:
        base = 110.0 + (stable_seed(profile) % 90)
        audio += sine_layer(t, base, 0.30)
        if profile.ra_deg is not None and profile.dec_deg is not None:
            position_factor = (math.sin(math.radians(profile.dec_deg)) + 1.0) / 2.0
            audio += sine_layer(t, 220.0 + 160.0 * position_factor, 0.12)
            mapping.append(
                {
                    "layer": "Resolved sky-position layer",
                    "source_parameter": "RA / Dec",
                    "relationship": "Coordinates control a reproducible harmonic relationship to preserve object identity.",
                    "status": "Interpretive identity mapping",
                }
            )
        mapping.append(
            {
                "layer": "Extragalactic carrier",
                "source_parameter": "Resolved object class",
                "relationship": "The sound architecture differs from planetary and stellar classes; absent spectral/time-series measurements are not fabricated.",
                "status": "Physics-informed model",
            }
        )

    elif "nebula" in kind or "supernova" in kind:
        noise = band_limited_noise(rng, n, 65)
        audio += noise * 0.10
        audio += sine_layer(t, 150.0 + stable_seed(profile) % 120, 0.24)
        mapping.append(
            {
                "layer": "Nebular dynamic texture",
                "source_parameter": "Object class",
                "relationship": "Controlled texture represents a model of complex spatial/physical structure; it is not an acoustic recording.",
                "status": "Physics-informed model",
            }
        )

    elif "black hole" in kind:
        audio += sine_layer(t, 70.0, 0.32)
        sweep = 100.0 + 380.0 * (t / duration) ** 1.5
        audio += 0.17 * np.sin(2 * np.pi * sweep * t)
        mapping.append(
            {
                "layer": "Gravitational-scale model",
                "source_parameter": "Black-hole object class",
                "relationship": "A controlled frequency trajectory represents changing dynamical scale; it is not a measured acoustic signal.",
                "status": "Physics-based auditory model",
            }
        )
        mapping.append(
            {
                "layer": "Scientific limitation",
                "source_parameter": "No gravitational-wave series supplied",
                "relationship": "No gravitational-wave measurement is fabricated.",
                "status": "Explicit limitation",
            }
        )

    else:
        base = 120.0 + stable_seed(profile) % 220
        audio += sine_layer(t, base, 0.26)
        texture = band_limited_noise(rng, n, 140)
        audio += texture * 0.035
        mapping.append(
            {
                "layer": "General resolved-object model",
                "source_parameter": "Available scientific profile",
                "relationship": "Deterministic representation based on available identity/profile information.",
                "status": "Interpretive representation",
            }
        )

    # Add a very low-amplitude dynamic layer only when there is enough data.
    if len(profile.sources) >= 3:
        modulation = 0.75 + 0.25 * np.sin(2 * np.pi * 0.07 * t)
        audio *= modulation
        mapping.append(
            {
                "layer": "Dynamic envelope",
                "source_parameter": "Scientific profile completeness",
                "relationship": "Controls slow dynamics without introducing an unobserved physical measurement.",
                "status": "Accessibility-oriented model",
            }
        )

    audio = np.tanh(audio * (0.72 + intensity * 0.58))
    audio *= envelope

    mode = profile.audio_mode
    explanation = build_rule_based_audio_explanation(profile)
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "profile": asdict(profile),
                "duration": duration,
                "intensity": intensity,
                "seed": stable_seed(profile),
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()

    return AudioResult(
        wav_bytes=make_wav(audio, sr),
        duration=duration,
        sample_rate=sr,
        seed=stable_seed(profile),
        mode=mode,
        mapping=mapping,
        explanation=explanation,
        cache_key=cache_key,
    )


# ---------------------------------------------------------------------------
# Search-and-Sonify: psychoacoustic real-time audio engine
# ---------------------------------------------------------------------------

def synthesize_tactile_audio(plan: dict[str, Any], duration: float = 14.0) -> dict[str, Any]:
    """
    Renders the JSON audio-mapping parameters from the Gemini tactile plan (or
    its deterministic fallback) directly into a NumPy oscillator/noise/FM
    stack. This is the Search-and-Sonify engine; it is separate from
    synthesize_audio() above, which remains the catalogue-driven deterministic
    engine used by the Object / Audio Lab / Export pages.
    """
    params = dict(plan.get("audio_mapping_parameters") or {})
    sub_bass_hz = clamp(float(params.get("sub_bass_frequency_hz", 32.0)), 20.0, 50.0)
    time_dilation_hz = clamp(float(params.get("time_dilation_lfo_hz", 0.3)), 0.02, 3.0)
    granular_density = clamp(float(params.get("granular_density", 0.4)), 0.0, 1.0)
    grain_ms = clamp(float(params.get("granular_grain_size_ms", 25.0)), 2.0, 120.0)
    grit = clamp(float(params.get("acoustic_grit_factor", 0.3)), 0.0, 1.0)
    noise_smoothing = int(clamp(float(params.get("ambient_noise_smoothing", 180)), 1, 400))
    dampening = clamp(float(params.get("atmospheric_dampening_coefficient", 0.2)), 0.0, 1.0)
    fm_index = clamp(float(params.get("fm_modulation_index", 5.0)), 0.0, 20.0)

    sr = AUDIO_SAMPLE_RATE
    duration = clamp(float(duration), 4.0, 30.0)
    n = int(duration * sr)
    t = np.arange(n, dtype=np.float64) / sr
    seed = int(hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    # Crushing weight / gravity: deep sub-bass sine for physical chest/
    # headphone vibration.
    sub_bass = np.sin(2.0 * np.pi * sub_bass_hz * t)

    # Time dilation: a crawling LFO that stretches gain over a slow cycle, to
    # feel like space/time stretching.
    time_lfo = 0.5 + 0.5 * np.sin(2.0 * np.pi * time_dilation_hz * t)
    sub_bass *= 0.55 + 0.45 * time_lfo

    # Kinetic debris: granular micro-bursts mixed with high-frequency acoustic
    # grit, simulating a violent, expanding explosion.
    grain_samples = max(1, int(sr * grain_ms / 1000.0))
    n_slots = max(1, n // grain_samples)
    grain_gate = np.zeros(n)
    active_slots = rng.random(n_slots) < granular_density
    for i, active in enumerate(active_slots):
        if active:
            start = i * grain_samples
            end = min(n, start + grain_samples)
            grain_gate[start:end] = 1.0
    grit_noise = band_limited_noise(rng, n, max(1, int(6 - 5 * grit)))
    kinetic_debris = grain_gate * grit_noise * (0.25 + 0.5 * grit)

    # Cold immensity: heavily smoothed ambient noise sweeps for freezing drift
    # in an endless void.
    ambient = band_limited_noise(rng, n, noise_smoothing)
    ambient_sweep = 0.5 + 0.5 * np.sin(2.0 * np.pi * 0.03 * t + 1.7)
    ambient_layer = ambient * ambient_sweep * 0.3

    # Suffocating atmosphere vs. an electric, stormy field: FM synthesis
    # loop with a rolling-average digital low-pass filter standing in for
    # atmospheric dampening.
    fm_carrier_hz = sub_bass_hz * 4.0
    fm_mod_hz = max(0.5, sub_bass_hz * 0.5)
    fm_signal = np.sin(2.0 * np.pi * fm_carrier_hz * t + fm_index * np.sin(2.0 * np.pi * fm_mod_hz * t))
    lowpass_window = max(1, int(3 + dampening * 250))
    if lowpass_window > 1:
        kernel = np.ones(lowpass_window) / lowpass_window
        fm_signal = np.convolve(fm_signal, kernel, mode="same")
    fm_layer = fm_signal * (0.18 + 0.22 * (1.0 - dampening))

    mix = 0.55 * sub_bass + 0.9 * kinetic_debris + 0.6 * ambient_layer + 0.5 * fm_layer

    attack = np.minimum(1.0, t / 0.15)
    release = np.minimum(1.0, (duration - t) / 0.3)
    envelope = np.clip(np.minimum(attack, release), 0.0, 1.0)
    mix *= envelope

    # Soft-knee hyperbolic tangent saturation limiter: prevents clipping and
    # maximizes somatic low-end impact.
    mix = np.tanh(mix * 1.4)

    return {"wav_bytes": make_wav(mix, sr), "duration": duration, "sample_rate": sr}


def prepare_object(query: str) -> ObjectProfile:
    """Resolve, enrich, explain, and sonify one selected object in one workflow."""
    profile = build_profile(query)
    # Gemini is only an interpretation layer. The deterministic Python audio
    # engine remains the authority for the actual waveform.
    plan = gemini_plan(profile)
    if plan:
        profile.audio_mode = plan.data_basis
        profile.audio_explanation = (
            f"{plan.primary_phenomenon}. {plan.recommended_mapping} "
            f"Listener focus: {plan.listener_focus} Limitations: {plan.limitations}"
        )
        profile.limitations = [plan.limitations]
    st.session_state.profile = profile
    st.session_state.audio = synthesize_audio(profile)
    st.session_state.selected_query = query
    if query not in st.session_state.recent:
        st.session_state.recent.append(query)
    return profile


def build_object_description(profile: ObjectProfile) -> str:
    kind = profile.object_type or "astronomical object"
    evidence = "measured or catalogue data" if profile.sources else "limited resolved information"
    return (
        f"{profile.canonical_name} is resolved as {kind}. The profile below is built from "
        f"{evidence} retrieved by the application's astronomy services. The auditory representation "
        "uses explicit mappings from the available evidence; it is not presented as a literal recording "
        "from the object."
    )


def build_object_facts(profile: ObjectProfile) -> list[str]:
    facts = []
    if profile.ra_deg is not None and profile.dec_deg is not None:
        facts.append(f"Sky position: RA {profile.ra_deg:.6f}°, Dec {profile.dec_deg:.6f}°.")
    if profile.object_type:
        facts.append(f"Resolved object class: {profile.object_type}.")
    for source in profile.sources[:8]:
        if source.field in {"right_ascension_deg", "declination_deg"}:
            continue
        value = source.value
        if isinstance(value, float):
            value = f"{value:.6g}"
        unit = f" {source.unit}" if source.unit else ""
        facts.append(f"{source.field}: {value}{unit} ({source.status}).")
    return facts


def build_rule_based_audio_explanation(profile: ObjectProfile) -> str:
    name = profile.canonical_name
    kind = profile.object_type
    if "pulsar" in kind.lower():
        return (
            f"This is a scientific auditory representation of {name}. "
            "Where measured pulse timing is available it is preserved in an audible "
            "time scale. Where it is not available in the current retrieved profile, "
            "the application does not invent a pulse rate."
        )
    if "black hole" in kind.lower():
        return (
            f"This is a physics-based auditory representation of {name}, not the "
            "literal sound of a black hole. The current profile is used to construct "
            "a controlled model, and no gravitational-wave measurement is implied "
            "unless such data is explicitly present."
        )
    return (
        f"This is a {profile.audio_mode.lower()} of {name}. "
        "The audio engine uses available scientific or catalogue parameters, "
        "maps them into controlled audible ranges, and labels modeled or "
        "interpretive relationships rather than presenting them as literal recordings."
    )


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def profile_json(profile: ObjectProfile, audio: Optional[AudioResult] = None) -> bytes:
    payload = asdict(profile)
    payload["application"] = APP_NAME
    payload["application_version"] = APP_VERSION
    if audio:
        payload["audio_metadata"] = {
            "duration_seconds": audio.duration,
            "sample_rate": audio.sample_rate,
            "seed": audio.seed,
            "mode": audio.mode,
            "cache_key": audio.cache_key,
            "mapping": audio.mapping,
        }
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")


def profile_csv(profile: ObjectProfile) -> bytes:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["object", "object_type", "field", "value", "unit", "status", "uncertainty", "source", "retrieved_at"])
    for source in profile.sources:
        writer.writerow(
            [
                profile.canonical_name,
                profile.object_type,
                source.field,
                source.value,
                source.unit,
                source.status,
                source.uncertainty,
                source.source,
                source.retrieved_at,
            ]
        )
    return out.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=APP_NAME,
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');

:root {
  --ink: #101216;
  --muted: #646a73;
  --line: #dfe2e6;
  --paper: #f7f7f5;
  --white: #ffffff;
  --blue: #315fce;
  --blue-soft: #eaf0ff;
  --dark: #0c0e12;
}

html, body, [class*="css"] {
  font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
  background: var(--paper);
  color: var(--ink);
}

.block-container {
  max-width: 1440px;
  padding-top: 1.5rem;
  padding-bottom: 5rem;
}

header[data-testid="stHeader"] {
  background: transparent;
}

section[data-testid="stSidebar"] { display: none !important; }
.su-topbar { position: sticky; top: 0; z-index: 999; background: rgba(247,247,245,.98); border: 1px solid var(--line); padding: .7rem .8rem .55rem; backdrop-filter: blur(12px); }
.su-brand { display:flex; flex-direction:column; gap:.18rem; min-width:0; padding:.1rem 0 .25rem; }
.su-brand strong { font-size:clamp(.86rem, 1.45vw, 1.08rem); letter-spacing:.045em; line-height:1.15; white-space:normal; overflow-wrap:anywhere; }
.su-brand span { font-family:"DM Mono",monospace; font-size:clamp(.48rem, .7vw, .60rem); letter-spacing:.09em; color:var(--muted); line-height:1.25; white-space:normal; }
.su-header-nav { margin-top:.3rem; }
.su-header-rule { height:.45rem; }
.su-topbar button { min-height:2.45rem; height:auto; font-size:clamp(.60rem, .78vw, .74rem); line-height:1.15; white-space:normal !important; overflow-wrap:anywhere; padding:.45rem .25rem !important; }
.su-topbar [data-testid="stSelectbox"] label { display:none; }
.su-topbar [data-baseweb="select"] > div { min-height:2.45rem; height:auto; border-radius:0; border-color:var(--line); background:#fff; color:#101216; }
.su-topbar [data-baseweb="select"] * { color:#101216 !important; }
@media (max-width: 900px) {
  .su-topbar { padding:.62rem .55rem .5rem; }
  .su-topbar button { font-size:.62rem; min-height:2.6rem; }
}
@media (max-width: 600px) {
  .su-brand strong { font-size:.88rem; letter-spacing:.025em; }
  .su-brand span { font-size:.48rem; letter-spacing:.065em; }
  .su-topbar button { font-size:.59rem; min-height:2.55rem; }
}

.su-kicker {
  font-family: "DM Mono", monospace;
  font-size: 0.72rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 1.1rem;
}

.su-hero {
  background: var(--dark);
  color: white;
  padding: clamp(2.2rem, 5vw, 5rem);
  border-radius: 0;
  margin-bottom: 1.5rem;
}

.su-hero h1 {
  font-size: clamp(3rem, 7vw, 7.4rem);
  line-height: 0.92;
  letter-spacing: -0.065em;
  margin: 0;
  font-weight: 700;
  max-width: 980px;
}

.su-hero p {
  max-width: 820px;
  color: #c7cbd2;
  font-size: 1.08rem;
  line-height: 1.65;
  margin-top: 1.5rem;
}

.su-rule {
  border-top: 1px solid var(--line);
  margin: 2.3rem 0 1.3rem;
}

.su-section {
  font-family: "DM Mono", monospace;
  font-size: 0.72rem;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 2.5rem 0 0.8rem;
}

.su-object-title {
  font-size: clamp(2.2rem, 5vw, 5.2rem);
  line-height: 0.95;
  letter-spacing: -0.055em;
  margin: 0;
}

.su-object-meta {
  font-family: "DM Mono", monospace;
  font-size: 0.78rem;
  color: var(--muted);
  margin-top: 0.8rem;
}

.su-panel {
  border-top: 1px solid var(--line);
  padding-top: 1.25rem;
  margin-top: 1.25rem;
}

.su-label {
  font-family: "DM Mono", monospace;
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--muted);
}

.su-value {
  font-size: 1.02rem;
  margin-top: 0.35rem;
  line-height: 1.55;
}

.su-note {
  background: var(--blue-soft);
  border-left: 3px solid var(--blue);
  padding: 1rem 1.1rem;
  line-height: 1.55;
}

.su-dark {
  background: var(--dark);
  color: white;
  padding: 1.7rem;
}

.su-dark .su-label {
  color: #aeb4bf;
}

div[data-testid="stTextInput"] input { border:1px solid #bfc4cb; border-radius:0; min-height:3.8rem; font-size:1.1rem; background:#fff !important; color:#101216 !important; -webkit-text-fill-color:#101216 !important; caret-color:#315fce !important; opacity:1 !important; }
div[data-testid="stTextInput"] input::placeholder { color:#6a7079 !important; -webkit-text-fill-color:#6a7079 !important; opacity:1 !important; }

div[data-testid="stTextInput"] input:focus {
  border-color: var(--blue);
  box-shadow: 0 0 0 1px var(--blue);
}

button[kind="secondary"] {
  border-radius: 0;
  border-color: var(--line);
}

button[kind="primary"] {
  border-radius: 0;
  background: var(--blue);
  border-color: var(--blue);
}

.stDownloadButton button {
  border-radius: 0;
  width: 100%;
}

[data-testid="stAudio"] {
  margin-top: 0.7rem;
}

@media (max-width: 760px) {
  .su-hero { padding: 2rem 1.2rem; }
  .su-hero h1 { font-size: 3.4rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


def go_home() -> None:
    st.session_state["selected_query"] = None
    st.session_state["profile"] = None
    st.session_state["audio"] = None
    st.session_state["search"] = ""


if "selected_query" not in st.session_state:
    st.session_state.selected_query = None
if "profile" not in st.session_state:
    st.session_state.profile = None
if "audio" not in st.session_state:
    st.session_state.audio = None
if "recent" not in st.session_state:
    st.session_state.recent = []
if "favorites" not in st.session_state:
    st.session_state.favorites = []
if "search" not in st.session_state:
    st.session_state.search = ""
if "tactile_query" not in st.session_state:
    st.session_state.tactile_query = ""
if "tactile_result" not in st.session_state:
    st.session_state.tactile_result = None


# Persistent header navigation
PAGES = ["Home", "Object", "Tactile Sonify", "Audio Lab", "Favorites", "Data / Export", "Scientific Method", "About"]
MODEL_OPTIONS = {GEMINI_FLASH_MODEL: "Gemini 3.6 Flash", GEMINI_PRO_MODEL: "Gemini 3.1 Pro Preview", GEMINI_LITE_MODEL: "Gemini 3.5 Flash Lite"}
if "page" not in st.session_state: st.session_state.page = "Home"
if "gemini_model" not in st.session_state: st.session_state.gemini_model = model_secret("GEMINI_FLASH_MODEL", GEMINI_FLASH_MODEL)

# The navigation deliberately uses two compact rows rather than seven narrow
# columns. This keeps every label readable in portrait, landscape, tablet, and
# desktop layouts instead of relying on horizontal clipping.
st.markdown('<div class="su-topbar">', unsafe_allow_html=True)
brand, model_col = st.columns([3.1, 1.4], vertical_alignment="center")
with brand:
    st.markdown(f'<div class="su-brand"><strong>{APP_NAME}</strong><span>ASTRONOMICAL AUDIO EXPLORATION</span></div>', unsafe_allow_html=True)
with model_col:
    selected_model = st.selectbox(
        "Gemini model", list(MODEL_OPTIONS),
        index=list(MODEL_OPTIONS).index(st.session_state.gemini_model) if st.session_state.gemini_model in MODEL_OPTIONS else 0,
        format_func=lambda x: MODEL_OPTIONS[x],
        label_visibility="collapsed",
        key="header_gemini_model",
    )
    st.session_state.gemini_model = selected_model

st.markdown('<div class="su-header-nav">', unsafe_allow_html=True)
nav_rows = [PAGES[:4], PAGES[4:]]
for row_index, row_pages in enumerate(nav_rows):
    cols = st.columns(len(row_pages), gap="small")
    for col, page_name in zip(cols, row_pages):
        with col:
            if st.button(
                page_name, key=f"topnav_{page_name}", use_container_width=True,
                type="primary" if st.session_state.page == page_name else "secondary"
            ):
                st.session_state.page = page_name
                st.rerun()
st.markdown('</div></div><div class="su-header-rule"></div>', unsafe_allow_html=True)
page = st.session_state.page

# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

if page == "Home":
    st.markdown(
        """
<div class="su-hero">
  <div class="su-kicker" style="color:#9da4af;">Astronomical audio exploration</div>
  <h1>Hear the physical universe.</h1>
  <p>
    Search astronomical objects, resolve them through scientific services,
    inspect their physical data, and experience deterministic auditory
    representations built from measured or explicitly modeled information.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="su-section">01 — Search the universe</div>', unsafe_allow_html=True)
    search = st.text_input(
        "Search the universe",
        value=st.session_state.search,
        placeholder="Venus, M31, NGC 1275, PSR B1919+21, TRAPPIST-1 b…",
        label_visibility="visible",
        key="home_search",
    )
    st.session_state.search = search

    suggestions = local_suggestions(search)
    # Local suggestions respond immediately from the first meaningful character.
    # Remote SIMBAD prefix expansion begins at two characters because a one-letter
    # catalogue prefix is too broad to query responsibly.
    if len(search.strip()) >= 2 and len(suggestions) < MAX_SUGGESTIONS:
        remote = simbad_prefix_search(search, MAX_SUGGESTIONS - len(suggestions))
        seen = {normalize_query(x["name"]) for x in suggestions}
        for item in remote:
            if normalize_query(item["name"]) not in seen:
                suggestions.append(item)
                seen.add(normalize_query(item["name"]))
            if len(suggestions) >= MAX_SUGGESTIONS:
                break

    if search.strip():
        if suggestions:
            st.markdown('<div class="su-section">Live results</div>', unsafe_allow_html=True)
            for i, item in enumerate(suggestions):
                label = f'{item["name"]}  ·  {item["context"]}'
                if st.button(label, key=f"suggest_{i}_{item['name']}", use_container_width=True):
                    selected = item["name"]
                    st.session_state.search = selected
                    with st.spinner("Resolving scientific data and building the auditory representation…"):
                        prepare_object(selected)
                    st.session_state.page = "Object"
                    st.rerun()

        if st.button(f'Search astronomical databases for "{search.strip()}"', type="primary", use_container_width=True):
            selected = search.strip()
            with st.spinner("Resolving scientific data and building the auditory representation…"):
                prepare_object(selected)
            st.session_state.page = "Object"
            st.rerun()
    else:
        st.markdown(
            """
<div class="su-note">
<strong>Open-ended discovery.</strong><br>
The local index accelerates common searches, but it is not the universe's
object limit. Unknown catalogue names can be resolved through astronomical
services when available.
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="su-section">02 — What this system does</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### Resolve")
        st.write("Identify objects and aliases through astronomical name-resolution services.")
    with c2:
        st.markdown("### Validate")
        st.write("Preserve units, provenance, retrieval time, and the distinction between measured and modeled information.")
    with c3:
        st.markdown("### Sonify")
        st.write("Convert defensible data into a deterministic auditory representation and explain every mapping.")

# ---------------------------------------------------------------------------
# Object page
# ---------------------------------------------------------------------------

elif page == "Object":
    if not st.session_state.profile:
        st.markdown('<div class="su-section">03 — Object</div>', unsafe_allow_html=True)
        st.info("Search for an astronomical object from the Home page.")
    else:
        profile: ObjectProfile = st.session_state.profile
        st.markdown('<div class="su-section">03 — Object profile</div>', unsafe_allow_html=True)
        st.markdown(f'<h1 class="su-object-title">{profile.canonical_name}</h1>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="su-object-meta">{profile.object_type.upper()} · RETRIEVED {profile.retrieved_at}</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="su-rule"></div>', unsafe_allow_html=True)
        st.markdown('<div class="su-section">Description</div>', unsafe_allow_html=True)
        st.write(build_object_description(profile))

        st.markdown('<div class="su-section">Key facts</div>', unsafe_allow_html=True)
        facts = build_object_facts(profile)
        if facts:
            for fact in facts:
                st.markdown(f"- {fact}")
        else:
            st.write("No structured facts were returned by the current resolvers.")

        st.markdown('<div class="su-section">Physical profile</div>', unsafe_allow_html=True)

        if profile.ra_deg is not None or profile.dec_deg is not None:
            a, b = st.columns(2)
            with a:
                st.markdown('<div class="su-label">Right ascension</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="su-value">{profile.ra_deg if profile.ra_deg is not None else "Unavailable"}°</div>', unsafe_allow_html=True)
            with b:
                st.markdown('<div class="su-label">Declination</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="su-value">{profile.dec_deg if profile.dec_deg is not None else "Unavailable"}°</div>', unsafe_allow_html=True)

        if profile.properties:
            rows = []
            for key, value in profile.properties.items():
                if key == "horizons_object_record":
                    continue
                rows.append({"Parameter": key, "Value": str(value)})
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)

        st.markdown('<div class="su-section">Scientific basis</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="su-note"><strong>{profile.data_quality}</strong> — this label describes the basis of the current sonification mapping, not whether the astronomical object itself exists.</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="su-section">Audio identity</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
<div class="su-dark">
  <div class="su-label">Mode</div>
  <div style="font-size:1.5rem;margin-top:.3rem">{profile.audio_mode}</div>
  <div style="margin-top:1rem;line-height:1.6">
    {profile.audio_explanation}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        if st.session_state.audio is None:
            with st.spinner("Building deterministic scientific audio…"):
                st.session_state.audio = synthesize_audio(profile)

        st.markdown('<div class="su-section">Scientific audio</div>', unsafe_allow_html=True)
        st.audio(st.session_state.audio.wav_bytes, format="audio/wav")
        st.write(st.session_state.audio.explanation)
        st.caption(
            f"{st.session_state.audio.mode} · {st.session_state.audio.sample_rate:,} Hz · "
            f"{st.session_state.audio.duration:.0f} s · deterministic seed {st.session_state.audio.seed}"
        )
        if st.button("Regenerate auditory representation", use_container_width=True):
            with st.spinner("Rebuilding from the same scientific profile…"):
                st.session_state.audio = synthesize_audio(profile)
            st.rerun()

        st.markdown('<div class="su-section">Scientific provenance</div>', unsafe_allow_html=True)
        if profile.sources:
            provenance_rows = [asdict(s) for s in profile.sources]
            st.dataframe(provenance_rows, use_container_width=True, hide_index=True)
        else:
            st.warning("No structured scientific values were returned by the current resolvers.")

        if profile.aliases:
            st.markdown('<div class="su-label">Known identifiers / aliases</div>', unsafe_allow_html=True)
            st.write(", ".join(profile.aliases[:50]))

        if profile.retrieval_errors:
            for err in profile.retrieval_errors:
                st.warning(err)

# ---------------------------------------------------------------------------
# Tactile Sonify (Search-and-Sonify, BVI pipeline)
# ---------------------------------------------------------------------------

elif page == "Tactile Sonify":
    st.markdown('<div class="su-section">Search-and-Sonify — tactile pipeline</div>', unsafe_allow_html=True)
    st.write(
        "Search any astronomical object. This pipeline pulls live web data, sends it to the "
        "selected Gemini model to design a tactile audio mapping, and renders it immediately as "
        "a real-time psychoacoustic soundscape — built for listeners who rely primarily on hearing."
    )

    tactile_query = st.text_input(
        "Search any astronomical object",
        value=st.session_state.tactile_query,
        placeholder="Sagittarius A*, TON 618, Crab Nebula, Betelgeuse…",
        key="tactile_search_input",
    )
    st.session_state.tactile_query = tactile_query
    tactile_duration = st.slider("Duration (seconds)", 4.0, 30.0, 14.0, 1.0, key="tactile_duration")

    if st.button("Run Search-and-Sonify", type="primary", use_container_width=True):
        if not tactile_query.strip():
            st.warning("Enter an object to search for first.")
        else:
            with st.spinner("Searching the web, consulting Gemini, and rendering tactile audio…"):
                st.session_state.tactile_result = run_search_and_sonify(
                    tactile_query, st.session_state.gemini_model, tactile_duration
                )

    result = st.session_state.tactile_result
    if result:
        plan = result["plan"]
        live_data = result["live_data"]
        audio = result["audio"]
        st.markdown(
            f'<h1 class="su-object-title">{plan.get("object_name", result["query"])}</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="su-note"><strong>Psychological concept:</strong> '
            f'{plan.get("psychological_concept", "")}</div>',
            unsafe_allow_html=True,
        )
        st.audio(audio["wav_bytes"], format="audio/wav")
        model_label = MODEL_OPTIONS.get(st.session_state.gemini_model, st.session_state.gemini_model)
        caption = f"{audio['sample_rate']:,} Hz · {audio['duration']:.0f} s · model: {model_label}"
        if result["used_fallback"]:
            caption += " · deterministic fallback (Gemini unavailable)"
        st.caption(caption)

        if not live_data.get("found"):
            st.info(
                "No live web or catalogue data was found for this query; the sound design below "
                "is a generic interpretive fallback rather than fabricated facts."
            )
        else:
            st.caption(f"Data source: {live_data.get('source')}")

        with st.expander("Audio mapping parameters"):
            st.json(plan.get("audio_mapping_parameters", {}))

        with st.expander("Scraped / resolved facts"):
            st.json({k: v for k, v in live_data.items() if k != "raw_snippets"})
            for snip in live_data.get("raw_snippets", []):
                st.write(f"**{snip.get('title', '')}** — {snip.get('snippet', '')}")

        st.markdown(
            '<div class="su-note"><strong>Note:</strong> this tactile sound design is an '
            "interpretive representation created by an AI sound-design layer from best-effort web "
            "data. It is not a literal recording or a claim of precise physical measurement.</div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Audio Lab
# ---------------------------------------------------------------------------

elif page == "Audio Lab":
    st.markdown('<div class="su-section">04 — Audio lab</div>', unsafe_allow_html=True)
    if not st.session_state.profile:
        st.info("Resolve an object first.")
    else:
        profile = st.session_state.profile
        st.markdown(f'<h1 class="su-object-title">{profile.canonical_name}</h1>', unsafe_allow_html=True)
        duration = st.slider("Duration (seconds)", 4.0, 30.0, DEFAULT_DURATION, 1.0)
        intensity = st.slider("Listening intensity", 0.15, 1.0, 0.65, 0.05)
        if st.button("Regenerate deterministic audio", type="primary", use_container_width=True):
            st.session_state.audio = synthesize_audio(profile, duration, intensity)

        audio = st.session_state.audio or synthesize_audio(profile, duration, intensity)
        st.audio(audio.wav_bytes, format="audio/wav")
        st.caption(
            f"Mode: {audio.mode} · {audio.sample_rate:,} Hz · {audio.duration:.0f} s · "
            f"deterministic seed {audio.seed}"
        )

        st.markdown('<div class="su-section">Mapping ledger</div>', unsafe_allow_html=True)
        st.dataframe(audio.mapping, use_container_width=True, hide_index=True)

        st.markdown(
            '<div class="su-note"><strong>Listening safety:</strong> start at a comfortable volume. '
            'The synthesis is peak-limited and designed to avoid excessive low/high-frequency energy, '
            'but normal hearing-safety practices still apply.</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Favorites
# ---------------------------------------------------------------------------

elif page == "Favorites":
    st.markdown('<div class="su-section">05 — Favorites</div>', unsafe_allow_html=True)
    if st.session_state.profile:
        name = st.session_state.profile.canonical_name
        if name in st.session_state.favorites:
            if st.button("Remove current object from favorites"):
                st.session_state.favorites.remove(name)
                st.rerun()
        else:
            if st.button("Add current object to favorites", type="primary"):
                st.session_state.favorites.append(name)
                st.rerun()

    if st.session_state.favorites:
        for i, favorite in enumerate(st.session_state.favorites):
            if st.button(favorite, key=f"favorite_{i}", use_container_width=True):
                st.session_state.selected_query = favorite
                st.session_state.profile = build_profile(favorite)
                st.session_state.audio = None
                st.rerun()
    else:
        st.write("No favorites yet.")

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

elif page == "Data / Export":
    st.markdown('<div class="su-section">06 — Data / export</div>', unsafe_allow_html=True)
    if not st.session_state.profile:
        st.info("Resolve an object first.")
    else:
        profile = st.session_state.profile
        audio = st.session_state.audio
        st.markdown(f'<h1 class="su-object-title">{profile.canonical_name}</h1>', unsafe_allow_html=True)

        st.download_button(
            "Download scientific profile JSON",
            data=profile_json(profile, audio),
            file_name=f"{re.sub(r'[^A-Za-z0-9._-]+', '_', profile.canonical_name)}_profile.json",
            mime="application/json",
            use_container_width=True,
        )
        st.download_button(
            "Download structured data CSV",
            data=profile_csv(profile),
            file_name=f"{re.sub(r'[^A-Za-z0-9._-]+', '_', profile.canonical_name)}_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if audio:
            st.download_button(
                "Download WAV sonification",
                data=audio.wav_bytes,
                file_name=f"{re.sub(r'[^A-Za-z0-9._-]+', '_', profile.canonical_name)}_sonification.wav",
                mime="audio/wav",
                use_container_width=True,
            )
        else:
            st.info("Generate audio first if you want the WAV and audio metadata export.")

# ---------------------------------------------------------------------------
# Scientific method
# ---------------------------------------------------------------------------

elif page == "Scientific Method":
    st.markdown('<div class="su-section">07 — Scientific method</div>', unsafe_allow_html=True)
    st.markdown("# Sound is not the same thing as sonification")
    st.write(
        "Sound is a mechanical pressure wave travelling through a medium. "
        "Sonification is the systematic conversion of data into sound. "
        "This application uses sonification and physics-informed auditory models "
        "rather than pretending that empty space contains ordinary audible noise."
    )

    sections = [
        (
            "Measured-data sonification",
            "An observed scientific time series or other measured quantity is mapped into an audible range. "
            "If frequency or time scaling is necessary, the transformation is disclosed."
        ),
        (
            "Physics-based auditory model",
            "Known physical or catalogue parameters are mapped through an explicit model. "
            "The result is a scientific auditory representation, not a literal recording."
        ),
        (
            "Interpretive representation",
            "When the available evidence is insufficient for a stronger model, the system uses a clearly labeled "
            "deterministic representation and avoids fabricating measurements."
        ),
        (
            "Evidence hierarchy",
            "The application prefers measured data, then explicit physical models, then interpretive representations. "
            "A lower evidence tier is not presented as though it were a direct observation."
        ),
    ]
    for title, body in sections:
        st.markdown(f"### {title}")
        st.write(body)

    st.markdown("### Transformation disclosure")
    st.write(
        "When an astronomical signal is outside ordinary human hearing, the application can shift its frequency "
        "into an audible range while preserving the relevant temporal relationship. When time is compressed or "
        "expanded, that is also stated. The output is never described as the literal sound of the object."
    )

# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------

else:
    st.markdown('<div class="su-section">08 — About</div>', unsafe_allow_html=True)
    st.markdown("# Sounds of the Universe")
    st.write(
        "A search-first astronomical exploration and accessibility application. "
        "The local index is an autocomplete accelerator, not a fixed universe catalogue. "
        "Scientific resolution can extend beyond the local index through CDS astronomical services, "
        "NASA Exoplanet Archive data, and NASA/JPL Horizons where applicable."
    )

    st.markdown("### System architecture")
    st.code(
        """SEARCH
  ↓
LOCAL AUTOCOMPLETE INDEX
  ↓
CDS / SIMBAD / NED / VizieR RESOLUTION
  ↓
NASA EXOPLANET ARCHIVE + NASA/JPL HORIZONS ENRICHMENT
  ↓
DATA VALIDATION + PROVENANCE
  ↓
GEMINI SCIENTIFIC SYNTHESIS (OPTIONAL)
  ↓
DETERMINISTIC PYTHON SONIFICATION
  ↓
AUDIO + EXPLANATION + EXPORT""",
        language="text",
    )

    st.markdown("### Current application version")
    st.write(APP_VERSION)
    st.caption(
        "Scientific services can change availability or schemas. The application handles remote failure "
        "gracefully and never treats a failed network call as permission to invent missing science."
    )
