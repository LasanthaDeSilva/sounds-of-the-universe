"""
Sounds of the Universe
A Streamlit application for open-ended astronomical search, scientific profiling,
and AI-driven, provenance-informed psychoacoustic sonification.

Initial deployment:
    streamlit run app.py

Recommended secrets:
    GEMINI_API_KEY = "..."
Optional model overrides:
    GEMINI_FLASH_MODEL = "gemini-3.6-flash"
    GEMINI_PRO_MODEL = "gemini-3.1-pro"
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
    from pydantic import BaseModel, Field, field_validator
except Exception:
    BaseModel = None  # type: ignore
    Field = None  # type: ignore
    field_validator = None  # type: ignore

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
GEMINI_PRO_MODEL = "gemini-3.1-pro"
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
        return "Universal Psychoacoustic Atmosphere Simulator"
    if profile.sources or any(k.startswith("web_") for k in profile.properties):
        return "Universal Psychoacoustic Atmosphere Simulator"
    return "Interpretive psychoacoustic representation"


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


# ---------------------------------------------------------------------------
# Universal Psychoacoustic Atmosphere Simulator — Gemini schema
# ---------------------------------------------------------------------------
if BaseModel is not None:
    class PsychoacousticAtmosphereVectors(BaseModel):
        abyss_depth_factor: float = Field(ge=0.0, le=1.0, description="0.0-1.0. Deep mass, gravitational presence, low-frequency enclosure and perceived weight.")
        spatial_expansion_rate: float = Field(ge=0.0, le=1.0, description="0.0-1.0. Slow evolving movement that makes the environment feel physically immense or spatially open.")
        kinetic_tension_index: float = Field(ge=0.0, le=1.0, description="0.0-1.0. Turbulence, rotation, storms, jets, pulsation or explosive dynamics become unstable auditory motion.")
        harmonic_dissonance: float = Field(ge=0.0, le=1.0, description="0.0-1.0. Micro-detuning and beating caused by complex, conflicting or irregular physical structure.")
        suffocation_coefficient: float = Field(ge=0.0, le=1.0, description="0.0-1.0. Atmospheric density, pressure and opacity become spectral enclosure and high-frequency absorption.")
        friction_texture_layer: float = Field(ge=0.0, le=1.0, description="0.0-1.0. Wind, turbulence, shocks, debris, plasma flow or other physically supported friction-like processes create textured noise.")
        thermal_hum_density: float = Field(ge=0.0, le=1.0, description="0.0-1.0. Thermal conditions control warm, dense midrange resonance versus sparse, cold hollow resonance.")

    class UniversalSomaticConfig(BaseModel):
        object_name: str
        psychological_presence_narrative: str = Field(
            min_length=30,
            description="Exactly two complete sentences. Vivid and psychologically evocative, but grounded only in the supplied physical evidence."
        )
        atmosphere_vectors: PsychoacousticAtmosphereVectors

        if field_validator is not None:
            @field_validator("psychological_presence_narrative")
            @classmethod
            def validate_two_sentences(cls, value: str) -> str:
                text = re.sub(r"\s+", " ", str(value)).strip()
                if not text:
                    raise ValueError("psychological_presence_narrative must not be empty")
                # Gemini is explicitly asked for exactly two sentences. Keep the
                # validator deliberately strict so malformed/truncated responses
                # are rejected before they reach the audio engine.
                parts = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
                if len(parts) != 2 or not re.search(r"[.!?]$", text):
                    raise ValueError("psychological_presence_narrative must contain exactly two sentences")
                return text
else:
    PsychoacousticAtmosphereVectors = None  # type: ignore
    UniversalSomaticConfig = None  # type: ignore

UNIVERSAL_ATMOSPHERE_SYSTEM_INSTRUCTION = (
    "You are a Cognitive Psychoacoustic & Cosmic Atmosphere Simulator. "
    "Your job is to translate the physical environment of ANY astronomical object into a psychologically convincing auditory atmosphere for a blind or visually impaired listener who is imagining being physically present there. "
    "Do not simply map catalog numbers to pitch, loudness, or brightness. Reason from the object's actual physical character: mass and density, gravity-related weight, atmospheric composition and pressure, temperature, storms and winds, turbulence, rotation, magnetic activity, plasma, jets, radiation environment, pulsation, shocks, explosive ejecta, surface or cloud processes, and any other explicitly supplied evidence. "
    "Use the seven vectors as psychoacoustic controls, not as measurements. Different physical environments must produce materially different vector profiles and therefore different sound textures. "
    "Never invent an unsupported physical trait. If evidence is missing, keep that dimension conservative rather than hallucinating it. "
    "The psychological_presence_narrative MUST contain exactly two sentences. It should describe what the environment would feel like to inhabit, not merely repeat scientific facts. "
    "The resulting audio is an interpretive psychoacoustic model, never a claim of literal sound traveling through vacuum."
)

def _extract_complete_json_object(text: str) -> str:
    """Extract the first complete JSON object, including when Gemini wraps it in markdown."""
    text = (text or "").strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)
    first = text.find("{")
    if first < 0:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for i in range(first, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[first:i + 1]
    return ""

def _parse_atmosphere_response(response):
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, UniversalSomaticConfig):
        return parsed
    candidates = []
    if isinstance(parsed, dict):
        candidates.append(parsed)
    elif isinstance(parsed, str):
        candidates.append(parsed)
    raw_text = getattr(response, "text", "") or ""
    if raw_text:
        candidates.append(raw_text)

    last_error = None
    for candidate in candidates:
        try:
            if isinstance(candidate, dict):
                return UniversalSomaticConfig.model_validate(candidate)
            blob = _extract_complete_json_object(candidate)
            if not blob:
                continue
            data = json.loads(blob)
            return UniversalSomaticConfig.model_validate(data)
        except Exception as exc:
            last_error = exc
    if last_error:
        raise ValueError(f"Gemini returned an invalid atmosphere profile: {last_error}") from last_error
    raise ValueError("Gemini returned no complete JSON atmosphere profile.")

def generate_universal_atmosphere_profile(profile: ObjectProfile) -> "UniversalSomaticConfig":
    if UniversalSomaticConfig is None:
        raise RuntimeError("pydantic is not installed; atmosphere schema unavailable.")
    client = get_gemini_client()
    if client is None:
        raise ValueError("Gemini Client failed to load.")
    if types is None:
        raise RuntimeError("google-genai types are unavailable.")

    # Keep the model focused on physical evidence, with source-level provenance
    # available in the prompt so it cannot confuse an arbitrary scraped phrase
    # with a measured astronomical fact.
    payload = {
        "name": profile.canonical_name,
        "type": profile.object_type,
        "physical_properties": profile.properties,
        "scientific_sources": [asdict(x) for x in profile.sources],
        "web_evidence": profile.properties.get("web_scraped_pages", []),
    }
    model_to_use = st.session_state.get("gemini_model", model_secret("GEMINI_FLASH_MODEL", GEMINI_FLASH_MODEL))
    prompt = (
        "Construct a compact but object-specific psychoacoustic atmosphere profile from this evidence. "
        "Do not reuse a generic vector pattern when the object's physical environment differs. "
        "Return ONLY the requested JSON schema. Keep the narrative exactly two sentences and concise enough to avoid truncation.\n\n"
        "OBJECT EVIDENCE:\n" + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    )
    config_args = dict(
        system_instruction=UNIVERSAL_ATMOSPHERE_SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=UniversalSomaticConfig,
        temperature=0.15,
        max_output_tokens=1200,
    )

    response = client.models.generate_content(
        model=model_to_use,
        contents=prompt,
        config=types.GenerateContentConfig(**config_args),
    )
    try:
        return _parse_atmosphere_response(response)
    except Exception as first_error:
        # A small second pass handles occasional SDK responses where `parsed`
        # contains a truncated JSON string despite the response schema.
        retry_prompt = (
            "Your previous response was incomplete. Return one COMPLETE valid JSON object only, "
            "matching the schema exactly. Use exactly two short sentences for psychological_presence_narrative. "
            "No markdown, no commentary, no trailing text.\n\nOBJECT EVIDENCE:\n" +
            json.dumps(payload, ensure_ascii=False, default=str)
        )
        retry = client.models.generate_content(
            model=model_to_use,
            contents=retry_prompt,
            config=types.GenerateContentConfig(
                system_instruction=UNIVERSAL_ATMOSPHERE_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=UniversalSomaticConfig,
                temperature=0.05,
                max_output_tokens=900,
            ),
        )
        try:
            return _parse_atmosphere_response(retry)
        except Exception as second_error:
            raise ValueError(f"Gemini atmosphere JSON could not be validated after retry: {second_error}") from first_error

def get_gemini_client():
    if genai is None:
        return None
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

UniversalBoneSomaticConfig=UniversalSomaticConfig


# ---------------------------------------------------------------------------
# Search-and-Sonify: live web data ingestion layer
# ---------------------------------------------------------------------------
_FACT_PATTERNS={
 "estimated_mass":re.compile(r"mass[^.\n]{0,80}?([\d][\d.,]*(?:\s*[×x]\s*10\^?[\d+-]+)?\s*(?:solar masses|earth masses|jupiter masses|kg|kilograms))",re.I),
 "radius":re.compile(r"radius[^.\n]{0,80}?([\d][\d.,]*\s*(?:km|kilometers|solar radii|earth radii|jupiter radii))",re.I),
 "temperature":re.compile(r"(?:temperature|effective temperature|equilibrium temperature)[^.\n]{0,80}?([\d][\d.,]*\s*(?:k\b|kelvin|°c|celsius|°f|fahrenheit))",re.I),
 "pressure":re.compile(r"pressure[^.\n]{0,80}?([\d][\d.,]*\s*(?:bar|bars|atm|atmospheres|pa|kpa|mpa))",re.I),
 "density":re.compile(r"density[^.\n]{0,80}?([\d][\d.,]*\s*(?:g/cm3|g/cm³|kg/m3|kg/m³))",re.I),
 "wind_speed":re.compile(r"(?:wind speed|winds)[^.\n]{0,80}?([\d][\d.,]*\s*(?:km/s|km/h|m/s|mph))",re.I),
 "magnetic_field":re.compile(r"magnetic field[^.\n]{0,80}?([\d][\d.,]*\s*(?:tesla|gauss|µt|μt|nt))",re.I),
 "explosion_velocity":re.compile(r"(?:explosion|ejecta|outflow|expansion)[^.\n]{0,40}?velocity[^.\n]{0,60}?([\d][\d.,]*\s*(?:km/s|kilometers per second|m/s|mph))",re.I),
}
_OBJECT_TYPE_KEYWORDS=["black hole","neutron star","pulsar","supernova remnant","supernova","nebula","exoplanet","quasar","galaxy cluster","galaxy","white dwarf","red giant","protostar","star","planet","moon"]

def _extract_physical_facts(text):
    facts={k:None for k in _FACT_PATTERNS}; facts["object_type"]=None
    for k,pat in _FACT_PATTERNS.items():
        m=pat.search(text or "")
        if m: facts[k]=m.group(1).strip()
    low=(text or "").lower()
    for k in _OBJECT_TYPE_KEYWORDS:
        if k in low: facts["object_type"]=k.title(); break
    return facts

def _search_api_key(name):
    try: v=st.secrets.get(name)
    except Exception: v=None
    if not v:
        import os; v=os.environ.get(name)
    return v or None

@st.cache_data(ttl=1800,show_spinner=False)
def _tavily_search(query,api_key):
    r=requests.post("https://api.tavily.com/search",json={"api_key":api_key,"query":f"{query} astronomical physical properties mass temperature atmosphere pressure storms winds magnetic field","max_results":6},timeout=HTTP_TIMEOUT); r.raise_for_status(); d=r.json()
    return [{"title":x.get("title",""),"snippet":x.get("content",""),"url":x.get("url","")} for x in d.get("results",[])]

@st.cache_data(ttl=1800,show_spinner=False)
def _serper_search(query,api_key):
    r=requests.post("https://google.serper.dev/search",headers={"X-API-KEY":api_key,"Content-Type":"application/json"},json={"q":f"{query} astronomical physical properties mass temperature atmosphere pressure storms winds magnetic field"},timeout=HTTP_TIMEOUT); r.raise_for_status(); d=r.json()
    return [{"title":x.get("title",""),"snippet":x.get("snippet",""),"url":x.get("link","")} for x in d.get("organic",[])]

@st.cache_data(ttl=1800,show_spinner=False)
def _scrape_search_result_pages(snippets):
    pages=[]
    for title,url,snippet in snippets[:6]:
        if not url.startswith(("http://","https://")): continue
        try:
            r=requests.get(url,headers={"User-Agent":USER_AGENT,"Accept":"text/html,application/xhtml+xml,text/plain"},timeout=HTTP_TIMEOUT); r.raise_for_status()
            html=r.text[:1500000]; html=re.sub(r"(?is)<(script|style|noscript|svg).*?>.*?</\1>"," ",html); text=re.sub(r"(?s)<[^>]+>"," ",html); text=re.sub(r"\s+"," ",text).strip()[:6000]
            pages.append({"title":title,"url":url,"snippet":snippet,"page_text":text})
        except Exception: pass
    return pages

def fetch_live_web_data(query):
    query=(query or "").strip(); result={"query":query,"found":False,"source":None,"object_type":None,"estimated_mass":None,"radius":None,"temperature":None,"pressure":None,"density":None,"wind_speed":None,"magnetic_field":None,"explosion_velocity":None,"raw_snippets":[],"scraped_pages":[]}
    if not query: return result
    snippets=[]; source=None
    try:
        tk=_search_api_key("TAVILY_API_KEY"); sk=_search_api_key("SERPER_API_KEY")
        if tk: snippets=_tavily_search(query,tk); source="tavily + direct page scrape"
        elif sk: snippets=_serper_search(query,sk); source="serper + direct page scrape"
    except Exception: snippets=[]
    if snippets:
        tuples=tuple((x.get("title",""),x.get("url",""),x.get("snippet","")) for x in snippets); pages=_scrape_search_result_pages(tuples)
        evidence=" ".join(f"{x.get('title','')}. {x.get('snippet','')} {x.get('page_text','')}" for x in pages) or " ".join(f"{x.get('title','')}. {x.get('snippet','')}" for x in snippets)
        result.update(_extract_physical_facts(evidence)); result["raw_snippets"]=snippets[:6]; result["scraped_pages"]=pages[:6]; result["source"]=source; result["found"]=True
    try: profile=build_profile(query)
    except Exception: profile=None
    if profile and profile.sources:
        result["resolver_profile"]={"canonical_name":profile.canonical_name,"object_type":profile.object_type,"properties":profile.properties,"sources":[asdict(x) for x in profile.sources]}
        props=profile.properties; result["object_type"]=result.get("object_type") or profile.object_type; result["estimated_mass"]=result.get("estimated_mass") or props.get("pl_bmasse") or props.get("st_mass"); result["radius"]=result.get("radius") or props.get("pl_rade") or props.get("st_rad"); result["temperature"]=result.get("temperature") or props.get("pl_eqt") or props.get("st_teff"); result["found"]=True; result["source"]=result["source"] or "CDS/NASA astronomical resolvers"
        result["raw_snippets"].extend({"title":x.source,"snippet":f"{x.field}: {x.value} {x.unit}".strip(),"url":x.reference} for x in profile.sources[:8])
    return result

def enrich_profile_with_live_web_data(profile,live_data):
    for key in ("estimated_mass","radius","temperature","pressure","density","wind_speed","magnetic_field","explosion_velocity","object_type"):
        value=live_data.get(key)
        if value not in (None,""):
            profile.properties[f"web_{key}"]=value
            if key=="object_type" and profile.object_type=="Astronomical object": profile.object_type=str(value)
    profile.properties["web_data_source"]=live_data.get("source") or "none"
    if live_data.get("scraped_pages"): profile.properties["web_scraped_pages"]=[{"title":x.get("title"),"url":x.get("url")} for x in live_data["scraped_pages"]]
    if live_data.get("resolver_profile"): profile.properties["resolver_evidence"]=live_data["resolver_profile"].get("properties",{})
    return profile


# ---------------------------------------------------------------------------
# Search-and-Sonify execution
# ---------------------------------------------------------------------------
def run_search_and_sonify(query,model,duration):
    live_data=fetch_live_web_data(query); profile=enrich_profile_with_live_web_data(build_profile(query),live_data)
    old=st.session_state.get("gemini_model"); st.session_state.gemini_model=model
    try: config=generate_universal_atmosphere_profile(profile)
    finally:
        if old is not None: st.session_state.gemini_model=old
    return {"query":query,"profile":profile,"live_data":live_data,"config":config,"audio":synthesize_audio(profile,config,duration,0.65)}


def make_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono or stereo floating-point audio as PCM WAV."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim != 2 or x.shape[1] not in (1, 2):
        raise ValueError("Audio must be mono or stereo.")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        x = x / peak
    pcm16 = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(int(x.shape[1]))
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Universal psychoacoustic NumPy sound synthesis engine
# ---------------------------------------------------------------------------
def clamp(x,low,high): return max(low,min(high,x))
def stable_seed(profile):
    canonical=json.dumps({"name":profile.canonical_name,"type":profile.object_type,"properties":profile.properties,"sources":[asdict(s) for s in profile.sources],"app_version":APP_VERSION},sort_keys=True,default=str)
    return int.from_bytes(hashlib.sha256(canonical.encode()).digest()[:8],"big") & 0xFFFFFFFF
def normalized_number(value,low,high,default=0.5):
    try: x=float(value)
    except Exception: return default
    if high==low: return default
    if low>0 and high>low and x>0: return clamp((math.log10(x)-math.log10(low))/(math.log10(high)-math.log10(low)),0,1)
    return clamp((x-low)/(high-low),0,1)

def _single_pole_lowpass(x, choking):
    """Simple low-pass whose cutoff falls as choking approaches 1."""
    alpha = clamp(1.0 - clamp(choking, 0, 1) * 0.985, 0.015, 1.0)
    y = np.empty_like(x)
    acc = 0.0
    inv = 1 - alpha
    for i, sample in enumerate(x):
        acc = alpha * sample + inv * acc
        y[i] = acc
    return y

def _single_pole_highpass(x, amount):
    """Complementary high-pass used to keep airy scenes from becoming bass-heavy."""
    amount = clamp(amount, 0, 1)
    if amount <= 0:
        return x
    low = _single_pole_lowpass(x, 0.15 + 0.82 * amount)
    return x - low

def _bandpass_noise(rng, n, sr, low_hz, high_hz):
    """Fast approximate band-limited noise using cascaded one-pole filters."""
    noise = rng.normal(0.0, 1.0, n)
    low_hz = max(10.0, float(low_hz))
    high_hz = max(low_hz + 10.0, float(high_hz))
    low = _single_pole_lowpass(noise, clamp(1.0 - (2.0 * math.pi * high_hz / sr), 0.01, 0.99))
    if low_hz > 10:
        low_component = _single_pole_lowpass(noise, clamp(1.0 - (2.0 * math.pi * low_hz / sr), 0.01, 0.99))
        low = low - low_component
    peak = max(float(np.max(np.abs(low))), 1e-9)
    return low / peak

def _scene_family(profile: ObjectProfile):
    """Return perceptual scene weights rather than forcing every object into a drone."""
    kind = (profile.object_type or "").lower()
    name = (profile.canonical_name or "").lower()
    text = f"{kind} {name}"
    if "pulsar" in text or "neutron star" in text:
        return {"low": .07, "air": .22, "harmonic": .16, "texture": .16, "pulse": .72, "events": .05, "shimmer": .28, "silence": .02}
    if "supernova" in text:
        return {"low": .04, "air": .30, "harmonic": .08, "texture": .38, "pulse": .04, "events": .86, "shimmer": .36, "silence": .01}
    if "black hole" in text:
        return {"low": .12, "air": .26, "harmonic": .12, "texture": .32, "pulse": .02, "events": .22, "shimmer": .22, "silence": .18}
    if "quasar" in text or "active galaxy" in text or "active galactic" in text:
        return {"low": .06, "air": .38, "harmonic": .10, "texture": .52, "pulse": .06, "events": .32, "shimmer": .48, "silence": .01}
    if "nebula" in text:
        return {"low": .025, "air": .58, "harmonic": .20, "texture": .22, "pulse": .01, "events": .04, "shimmer": .58, "silence": .02}
    if "galaxy" in text or "galaxy cluster" in text:
        return {"low": .035, "air": .52, "harmonic": .18, "texture": .22, "pulse": .01, "events": .06, "shimmer": .42, "silence": .03}
    if "gas giant" in text or any(x in text for x in ("jupiter", "saturn", "uranus", "neptune")):
        return {"low": .15, "air": .22, "harmonic": .18, "texture": .68, "pulse": .04, "events": .08, "shimmer": .16, "silence": .01}
    if "star" in text or "white dwarf" in text or "red giant" in text or "protostar" in text:
        return {"low": .045, "air": .34, "harmonic": .46, "texture": .26, "pulse": .02, "events": .08, "shimmer": .70, "silence": .01}
    if "moon" in text or "satellite" in text:
        return {"low": .045, "air": .34, "harmonic": .18, "texture": .18, "pulse": .01, "events": .03, "shimmer": .26, "silence": .10}
    if "planet" in text or "exoplanet" in text:
        return {"low": .06, "air": .36, "harmonic": .24, "texture": .34, "pulse": .02, "events": .04, "shimmer": .30, "silence": .04}
    return {"low": .06, "air": .38, "harmonic": .25, "texture": .28, "pulse": .01, "events": .05, "shimmer": .30, "silence": .04}

def _physical_signature(profile: ObjectProfile):
    """Derive conservative perceptual controls from resolved object data.

    These controls are a sound-design interpretation, not a claim that the object
    literally produces audible sound in space.
    """
    kind = (profile.object_type or "").lower()
    props = profile.properties or {}

    def num(*keys):
        expanded=[]
        for key in keys: expanded.extend((key, f"web_{key}"))
        for key in expanded:
            value=props.get(key)
            try: return float(value)
            except Exception:
                if isinstance(value,str):
                    m=re.search(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?",value.replace(",",""))
                    if m:
                        try: return float(m.group(0))
                        except Exception: pass
        return None

    mass=num("estimated_mass","pl_bmassj","pl_bmasse","st_mass")
    temp=num("temperature","pl_eqt","st_teff")
    radius=num("radius","pl_radj","pl_rade","st_rad")
    pressure=num("pressure")
    density=num("density")
    wind=num("wind_speed")
    magnetic=num("magnetic_field")
    ejecta=num("explosion_velocity")

    sig={"depth":.15,"motion":.20,"rough":.12,"enclosure":.05,"warmth":.40,"pulse":0.,"transient":0.,"diffuse":.35,"brightness":.45}
    if "gas giant" in kind or any(x in kind for x in ("jupiter","saturn","uranus","neptune")):
        sig.update(depth=.55,motion=.78,rough=.62,enclosure=.72,warmth=.42,diffuse=.25,brightness=.38)
    elif "planet" in kind or "exoplanet" in kind:
        sig.update(depth=.20,motion=.38,rough=.30,enclosure=.30,warmth=.50,diffuse=.32,brightness=.45)
    if any(x in kind for x in ("star","white dwarf","red giant","protostar")):
        sig.update(depth=.16,motion=.44,rough=.28,enclosure=.08,warmth=.90,diffuse=.45,brightness=.88)
    if "neutron star" in kind or "pulsar" in kind:
        sig.update(depth=.25,motion=.94,rough=.52,enclosure=.06,warmth=.36,pulse=.96,diffuse=.18,brightness=.76)
    if "black hole" in kind:
        sig.update(depth=.42,motion=.72,rough=.38,enclosure=.12,warmth=.28,diffuse=.42,brightness=.50)
    if "quasar" in kind or "active galaxy" in kind or "active galactic" in kind:
        sig.update(depth=.18,motion=.94,rough=.78,enclosure=.08,warmth=.74,diffuse=.58,brightness=.86)
    if "supernova" in kind:
        sig.update(depth=.10,motion=.99,rough=.90,enclosure=.04,warmth=.70,transient=.98,diffuse=.54,brightness=.92)
    if "nebula" in kind:
        sig.update(depth=.05,motion=.22,rough=.22,enclosure=.02,warmth=.55,diffuse=.96,brightness=.72)
    if "galaxy" in kind or "galaxy cluster" in kind:
        sig.update(depth=.08,motion=.30,rough=.25,enclosure=.02,warmth=.58,diffuse=.94,brightness=.64)
    if "moon" in kind or "satellite" in kind:
        sig.update(depth=.08,motion=.16,rough=.24,enclosure=.16,warmth=.26,diffuse=.54,brightness=.34)

    if pressure is not None:
        sig["enclosure"]=max(sig["enclosure"],normalized_number(pressure,.001,1000,.3))
    if density is not None:
        sig["depth"]=clamp(sig["depth"]+.22*normalized_number(density,.01,100,.35),0,1)
    if wind is not None:
        wn=normalized_number(wind,1,3000,.3)
        sig["motion"]=clamp(sig["motion"]+.30*wn,0,1)
        sig["rough"]=clamp(sig["rough"]+.24*wn,0,1)
    if magnetic is not None:
        sig["rough"]=clamp(sig["rough"]+.16*normalized_number(magnetic,1e-9,10,.2),0,1)
    if temp is not None:
        tn=normalized_number(temp,30,50000,.5)
        sig["warmth"]=clamp(.25+.75*tn,0,1)
        sig["brightness"]=clamp(.25+.75*tn,0,1)
    if mass is not None:
        sig["depth"]=clamp(sig["depth"]+.10*normalized_number(mass,.001,1000,.4),0,1)
    if radius is not None:
        sig["diffuse"]=clamp(sig["diffuse"]+.10*normalized_number(radius,.1,1000,.4),0,1)
    if ejecta is not None:
        sig["transient"]=max(sig["transient"],normalized_number(ejecta,100,300000,.5))
        sig["motion"]=max(sig["motion"],.65*sig["transient"])
    return sig

def _stereoize(field, pan, sr, delay_ms=0.0):
    """Turn a scene component into a subtle stereo field without cartoon panning."""
    pan=clamp(float(pan),-1,1)
    left=field*math.sqrt((1-pan)*0.5)
    right=field*math.sqrt((1+pan)*0.5)
    delay=int(clamp(delay_ms,0,18)*sr/1000)
    if delay:
        shifted=np.zeros_like(field)
        shifted[delay:]=field[:-delay]
        right=0.88*right+0.12*shifted*math.sqrt((1+pan)*0.5)
    return left,right

def synthesize_audio(profile, config, duration=DEFAULT_DURATION, intensity=0.65):
    """Create an object-specific perceptual scene.

    The design deliberately avoids a universal bass drone.  Each object class gets
    a different acoustic architecture (air, shimmer, turbulence, pulses, transients,
    spectral density, silence and motion), while Gemini's seven vectors shape the
    emotional qualities of that architecture.
    """
    if UniversalSomaticConfig is None or config is None:
        raise ValueError("A valid UniversalSomaticConfig from Gemini is required.")
    duration=clamp(float(duration),4,30); intensity=clamp(float(intensity),.15,1.)
    v=config.atmosphere_vectors.model_dump(); sig=_physical_signature(profile); family=_scene_family(profile)
    seed_material=json.dumps({"name":profile.canonical_name,"type":profile.object_type,"properties":profile.properties,"vectors":v},sort_keys=True,default=str)
    seed=int(hashlib.sha256(seed_material.encode()).hexdigest()[:8],16); rng=np.random.default_rng(seed)
    sr=AUDIO_SAMPLE_RATE; n=int(duration*sr); t=np.arange(n,dtype=np.float64)/sr

    # Emotional controls are intentionally weighted toward mid/high spectral identity.
    expansion=clamp(.72*v["spatial_expansion_rate"]+.28*sig["diffuse"],0,1)
    tension=clamp(.65*v["kinetic_tension_index"]+.35*sig["motion"],0,1)
    dissonance=clamp(.68*v["harmonic_dissonance"]+.32*sig["rough"],0,1)
    friction=clamp(.62*v["friction_texture_layer"]+.38*sig["rough"],0,1)
    thermal=clamp(.62*v["thermal_hum_density"]+.38*sig["warmth"],0,1)
    suffocation=clamp(.68*v["suffocation_coefficient"]+.32*sig["enclosure"],0,1)
    abyss=clamp(.72*v["abyss_depth_factor"]+.28*sig["depth"],0,1)

    # 1. AIR / SPACE: the default bed is airy and quiet, not a bass drone.
    air=_bandpass_noise(rng,n,sr,420+1100*expansion,2600+6500*expansion)
    air_mod=.35+.65*(.5+.5*np.sin(2*np.pi*(.012+.045*expansion)*t+rng.uniform(0,2*np.pi)))
    air*=air_mod*(.025+.24*family["air"]+.10*family["shimmer"])
    # Spectral sparkle gives hot/diffuse scenes an identifiable emotional "light".
    shimmer=_bandpass_noise(rng,n,sr,1800+1800*sig["brightness"],4200+5200*sig["brightness"])
    shimmer_mod=.45+.55*(.5+.5*np.sin(2*np.pi*(.08+.35*tension)*t+rng.uniform(0,2*np.pi)))
    shimmer*=shimmer_mod*(.012+.16*family["shimmer"]+.08*thermal)

    # 2. HARMONIC BODY: different intervals create different emotional colour.
    harmonic=np.zeros(n)
    center=150+330*thermal+220*sig["brightness"]
    if family["pulse"]>.5:
        ratios=[1.0,1.498,2.01,2.67]
    elif family["events"]>.6:
        ratios=[1.0,1.414,1.732,2.31]
    elif family["air"]>.45:
        ratios=[1.0,1.5,2.0,2.67]
    else:
        ratios=[1.0,1.26,1.50,1.78]
    for i,ratio in enumerate(ratios):
        detune=1+(0.002+0.014*dissonance)*np.sin(2*np.pi*(.025+.10*tension)*t+i*.9)
        harmonic+=np.sin(2*np.pi*center*ratio*detune*t+rng.uniform(0,2*np.pi))
    harmonic/=len(ratios)
    harmonic*= (.018+.26*family["harmonic"]+.14*thermal)
    if dissonance>.35:
        rough_harm=_bandpass_noise(rng,n,sr,260,1800)
        harmonic += rough_harm*(.015+.12*dissonance)*family["harmonic"]

    # 3. LOW PRESENCE: only a restrained physical-weight cue.
    low=np.zeros(n)
    low_rate=.004+.025*expansion+.035*tension
    low_phase=2*np.pi*low_rate*t+.25*np.sin(2*np.pi*.009*t)
    low_f=28+42*(1-abyss)+24*family["low"]
    low=(np.sin(2*np.pi*low_f*t+.35*np.sin(low_phase))+
         .34*np.sin(2*np.pi*low_f*1.49*t+low_phase))
    low*= (.008+.11*family["low"]+.055*abyss)

    # 4. ENVIRONMENTAL TEXTURE: wind/turbulence/grain, object-specific bandwidth.
    texture_low=90+220*friction
    texture_high=650+4200*friction+2200*family["texture"]
    texture=_bandpass_noise(rng,n,sr,texture_low,texture_high)
    turbulent_rate=.08+2.8*friction+1.8*sig["motion"]
    texture_mod=.28+.72*(.5+.5*np.sin(2*np.pi*turbulent_rate*t+rng.uniform(0,2*np.pi)))
    texture*=texture_mod*(.018+.30*family["texture"]+.20*friction)

    # 5. PULSAR: rhythm is the identity, not depth.
    pulse=np.zeros(n)
    if family["pulse"]>.1:
        period=None
        for key in ("period_s","pulse_period","pulsar_period"):
            try:
                period=float(profile.properties.get(key));
                if period>0: break
            except Exception: pass
        period=clamp(period if period else .75,.05,4.)
        phase=(t%period)/period
        envelope=np.exp(-phase*(16+18*tension))
        carrier=260+520*sig["brightness"]+180*tension
        pulse=(np.sin(2*np.pi*carrier*t)*envelope +
               .35*np.sin(2*np.pi*carrier*1.51*t)*envelope)*(.18+.42*family["pulse"])

    # 6. TRANSIENT EVENTS: explosive objects are sparse, evolving events, not constant rumble.
    events=np.zeros(n)
    if family["events"]>.1:
        event_count=max(2,int(duration*(.25+1.8*family["events"])))
        positions=rng.integers(int(.35*sr),max(int(.36*sr)+1,n-int(.5*sr)),size=event_count)
        impulses=np.zeros(n)
        impulses[np.unique(positions)]=1.0
        decay_len=max(1,int(sr*(.045+.16*(1-tension))))
        decay=np.exp(-np.arange(decay_len)/(sr*(.012+.045*(1-tension))))
        events=np.convolve(impulses,decay,mode="same")
        if np.max(events)>0: events/=np.max(events)
        blast=_bandpass_noise(rng,n,sr,120+400*thermal,1800+6500*sig["brightness"])
        events*=blast*(.08+.42*family["events"])

    # 7. BLACK-HOLE / LARGE-SCALE PRESENCE: intermittent pressure, with actual silence between swells.
    swell=np.zeros(n)
    if family["silence"]>.1:
        rate=.018+.045*tension
        gate=.18+.82*(.5+.5*np.sin(2*np.pi*rate*t+1.4))
        accretion=_bandpass_noise(rng,n,sr,70+120*thermal,700+1800*thermal)
        swell=accretion*gate*(.02+.18*family["silence"]+.14*family["texture"])

    # Combine according to scene family. No universal deep layer is allowed to dominate.
    mono=(family["air"]*air + family["shimmer"]*shimmer + harmonic +
          family["low"]*low + family["texture"]*texture + pulse + events + swell)

    # Kinetic movement: slow spectral amplitude motion creates the perception of an environment changing around the listener.
    movement_rate=.015+.13*tension+.06*expansion
    movement=.62+.38*(.5+.5*np.sin(2*np.pi*movement_rate*t+rng.uniform(0,2*np.pi)))
    mono*=movement

    # Dense atmospheres muffle the scene; this changes spectral access, not just volume.
    if suffocation>.08:
        mono=_single_pole_lowpass(mono,.30*suffocation)
        if suffocation>.72: mono=_single_pole_lowpass(mono,.25*suffocation)

    # Preserve quiet gaps. Emotional presence is stronger when the scene has contrast.
    silence_floor=.04+.10*(1-family["silence"])
    mono*=silence_floor+.96*(.5+.5*np.sin(2*np.pi*(.007+.018*expansion)*t+rng.uniform(0,2*np.pi)))

    # Gentle cinematic envelope; no artificial endless sustain from the first sample.
    attack_len=max(1,int(sr*(.35+.45*expansion)))
    release_len=max(1,int(sr*(.65+.55*(1-tension))))
    env=np.ones(n)
    env[:attack_len]=np.linspace(0,1,attack_len)
    env[-release_len:]*=np.linspace(1,0,release_len)
    mono*=env

    # Subtle stereo scene: components sit in different perceptual locations.
    # This is deliberately restrained so headphones feel spatial rather than gimmicky.
    l1,r1=_stereoize(air,-.20,sr,2.0)
    l2,r2=_stereoize(shimmer,.24,sr,4.5)
    l3,r3=_stereoize(harmonic,-.05,sr,0.0)
    l4,r4=_stereoize(low,.0,sr,0.0)
    l5,r5=_stereoize(texture,.28 if sig["motion"]>.5 else -.22,sr,3.0)
    l6,r6=_stereoize(pulse,-.08,sr,1.5)
    l7,r7=_stereoize(events,-.30 if sig["transient"]>.5 else .12,sr,5.5)
    l8,r8=_stereoize(swell,.10,sr,7.0)
    left=(family["air"]*l1+family["shimmer"]*l2+l3+family["low"]*l4+family["texture"]*l5+l6+l7+l8)
    right=(family["air"]*r1+family["shimmer"]*r2+r3+family["low"]*r4+family["texture"]*r5+r6+r7+r8)
    # Keep stereo coherent with the mono scene and avoid excessive channel divergence.
    left=.74*left+.26*mono; right=.74*right+.26*mono

    # Final perceptual mastering: soft saturation, then conservative peak normalization.
    stereo=np.column_stack((left,right))
    stereo=np.tanh(stereo*(1.0+1.45*intensity))
    peak=max(float(np.max(np.abs(stereo))),1e-9)
    stereo=.84*stereo/peak

    mapping=[{"parameter":key,"value":float(value)} for key,value in v.items()]
    mapping.extend({"parameter":f"physical_signature_{key}","value":float(value)} for key,value in sig.items())
    mapping.extend({"parameter":f"scene_weight_{key}","value":float(value)} for key,value in family.items())
    mapping.append({"parameter":"psychological_presence_narrative","value":config.psychological_presence_narrative})
    cache_key=hashlib.sha256(f"{seed_material}|{duration}|{intensity}".encode()).hexdigest()
    return AudioResult(
        make_wav(stereo,sr),duration,sr,seed,
        "Universal Psychoacoustic Atmosphere Simulator — object-specific perceptual scene",
        mapping,config.psychological_presence_narrative,cache_key,
    )

def ensure_somatic_audio(profile, duration=DEFAULT_DURATION, intensity=.65, force_refresh=False):
    config = None if force_refresh else st.session_state.get("somatic_config")
    if config is None:
        try:
            config = generate_universal_atmosphere_profile(profile)
        except Exception as exc:
            st.session_state.somatic_error = str(exc)
            st.session_state.somatic_config = None
            st.session_state.audio = None
            return
        st.session_state.somatic_config = config
        st.session_state.somatic_error = None
        profile.audio_mode = "Universal Psychoacoustic Atmosphere Simulator (Gemini + physical-object signature)"
        profile.audio_explanation = config.psychological_presence_narrative
    try:
        st.session_state.audio = synthesize_audio(profile, config, duration, intensity)
    except Exception as exc:
        st.session_state.somatic_error = f"Audio synthesis failed: {exc}"
        st.session_state.audio = None

def prepare_object(query: str) -> ObjectProfile:
    profile=build_profile(query)
    try: profile=enrich_profile_with_live_web_data(profile,fetch_live_web_data(query))
    except Exception as exc: profile.source_notes.append(f"Live web enrichment failed: {exc}")
    st.session_state.profile=profile; st.session_state.somatic_config=None; st.session_state.somatic_error=None; st.session_state.audio=None
    ensure_somatic_audio(profile); st.session_state.selected_query=query
    if query not in st.session_state.recent: st.session_state.recent.append(query)
    return profile

def build_object_description(profile: ObjectProfile) -> str:
    evidence="measured/catalogued and web-retrieved physical data" if profile.sources else "limited resolved information"
    return f"{profile.canonical_name} is resolved as {profile.object_type}. The auditory atmosphere is built from {evidence}; it is an interpretive psychoacoustic embodiment, not a literal recording from the object."

def build_object_facts(profile: ObjectProfile) -> list[str]:
    facts=[]
    if profile.ra_deg is not None and profile.dec_deg is not None: facts.append(f"Sky position: RA {profile.ra_deg:.6f}°, Dec {profile.dec_deg:.6f}°.")
    if profile.object_type: facts.append(f"Resolved object class: {profile.object_type}.")
    for source in profile.sources[:8]:
        if source.field in {"right_ascension_deg","declination_deg"}: continue
        value=f"{source.value:.6g}" if isinstance(source.value,float) else str(source.value); unit=f" {source.unit}" if source.unit else ""
        facts.append(f"{source.field}: {value}{unit} ({source.status}).")
    return facts

def build_rule_based_audio_explanation(profile: ObjectProfile) -> str:
    return f"This is a Universal Psychoacoustic Atmosphere Simulator representation of {profile.canonical_name}. Gemini interprets the supplied physical environment into psychoacoustic controls; the resulting sound is not claimed to be literal sound from space."

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
if "somatic_config" not in st.session_state:
    st.session_state.somatic_config = None
if "somatic_error" not in st.session_state:
    st.session_state.somatic_error = None


# Persistent header navigation
PAGES = ["Home", "Object", "Tactile Sonify", "Audio Lab", "Favorites", "Data / Export", "Scientific Method", "About"]
MODEL_OPTIONS = {GEMINI_FLASH_MODEL: "Gemini 3.6 Flash", GEMINI_PRO_MODEL: "Gemini 3.1 Pro", GEMINI_LITE_MODEL: "Gemini 3.5 Flash Lite"}
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
    inspect their physical data, and experience AI-generated somatic auditory
    representations grounded in measured or catalogued information.
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
                    with st.spinner("Retrieving physical evidence and building the psychoacoustic atmosphere…"):
                        prepare_object(selected)
                    st.session_state.page = "Object"
                    st.rerun()

        if st.button(f'Search astronomical databases for "{search.strip()}"', type="primary", use_container_width=True):
            selected = search.strip()
            with st.spinner("Retrieving physical evidence and building the psychoacoustic atmosphere…"):
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
        st.write("Convert defensible data into an AI-generated psychoacoustic atmosphere and explain every mapping.")

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

        st.markdown('<div class="su-section">Scientific evidence basis</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="su-note"><strong>{profile.data_quality}</strong> — this label describes the basis of the current sonification mapping, not whether the astronomical object itself exists.</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.audio is None and st.session_state.somatic_config is None and not st.session_state.somatic_error:
            with st.spinner("Building the live psychoacoustic atmosphere with Gemini…"):
                ensure_somatic_audio(profile)

        st.markdown('<div class="su-section">Psychological presence</div>', unsafe_allow_html=True)
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

        st.markdown('<div class="su-section">Immersive atmosphere</div>', unsafe_allow_html=True)
        if st.session_state.audio is None:
            st.error(
                "Could not generate audio: "
                f"{st.session_state.somatic_error or 'Gemini is not configured for this deployment.'}"
            )
        else:
            st.audio(st.session_state.audio.wav_bytes, format="audio/wav")
            st.caption(
                f"{st.session_state.audio.mode} · {st.session_state.audio.sample_rate:,} Hz · "
                f"{st.session_state.audio.duration:.0f} s"
            )
        if st.button("Regenerate atmosphere", use_container_width=True):
            with st.spinner("Requesting a fresh psychoacoustic atmosphere from Gemini…"):
                ensure_somatic_audio(profile, force_refresh=True)
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
# ---------------------------------------------------------------------------
# Tactile Sonify / Universal Atmosphere (BVI pipeline)
# ---------------------------------------------------------------------------
elif page == "Tactile Sonify":
    st.markdown('<div class="su-section">Search-and-Sonify — universal psychoacoustic atmosphere</div>', unsafe_allow_html=True)
    st.write("Search any astronomical object. Physical evidence is retrieved from web/astronomy services, Gemini translates the environment into psychological sound-design vectors, and NumPy renders the atmosphere.")
    tactile_query=st.text_input("Search any astronomical object",value=st.session_state.tactile_query,placeholder="Sagittarius A*, Venus, TON 618, Crab Nebula, Betelgeuse…",key="tactile_search_input")
    st.session_state.tactile_query=tactile_query
    tactile_duration=st.slider("Duration (seconds)",4.0,30.0,14.0,1.0,key="tactile_duration")
    if st.button("Run Search-and-Sonify",type="primary",use_container_width=True):
        if not tactile_query.strip(): st.warning("Enter an object to search for first.")
        else:
            with st.spinner("Retrieving physical evidence, building the psychological atmosphere, and rendering audio…"):
                try: st.session_state.tactile_result=run_search_and_sonify(tactile_query,st.session_state.gemini_model,tactile_duration)
                except Exception as exc: st.session_state.tactile_result={"error":str(exc),"query":tactile_query}
    result=st.session_state.tactile_result
    if result:
        if result.get("error"): st.error(f"Could not build the atmosphere: {result['error']}")
        else:
            config=result["config"]; live_data=result["live_data"]; audio=result["audio"]; profile=result["profile"]
            st.markdown(f'<h1 class="su-object-title">{config.object_name}</h1>',unsafe_allow_html=True)
            st.markdown(f'<div class="su-dark" style="font-size:1.12rem;line-height:1.7">{config.psychological_presence_narrative}</div>',unsafe_allow_html=True)
            st.markdown('<div class="su-section">Immersive atmosphere</div>',unsafe_allow_html=True); st.audio(audio.wav_bytes,format="audio/wav")
            st.caption(f"{audio.sample_rate:,} Hz · {audio.duration:.0f} s · model: {MODEL_OPTIONS.get(st.session_state.gemini_model,st.session_state.gemini_model)}")
            with st.expander("Psychoacoustic atmosphere vectors"): st.json(config.atmosphere_vectors.model_dump())
            with st.expander("Scraped and resolved physical evidence"):
                st.json({k:v for k,v in live_data.items() if k not in {"scraped_pages","raw_snippets"}})
                for x in live_data.get("scraped_pages",[]): st.write(f"**{x.get('title','')}** — {x.get('url','')}")
                for x in live_data.get("raw_snippets",[]): st.write(f"**{x.get('title','')}** — {x.get('snippet','')}")
            with st.expander("Scientific profile used by Gemini"): st.json({"canonical_name":profile.canonical_name,"object_type":profile.object_type,"properties":profile.properties,"sources":[asdict(x) for x in profile.sources]})
            st.markdown('<div class="su-note"><strong>Scientific boundary:</strong> this is an interpretive psychoacoustic embodiment of documented physical properties, not literal sound from the astronomical object.</div>',unsafe_allow_html=True)


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

        if st.button("Regenerate somatic profile from Gemini", type="primary", use_container_width=True):
            with st.spinner("Requesting a fresh psychoacoustic atmosphere from Gemini…"):
                ensure_somatic_audio(profile, duration, intensity, force_refresh=True)
        elif st.session_state.somatic_config is not None:
            # Slider-only changes: re-render locally from the cached config
            # instead of calling Gemini again.
            ensure_somatic_audio(profile, duration, intensity)
        elif st.session_state.audio is None and not st.session_state.somatic_error:
            with st.spinner("Requesting a psychoacoustic atmosphere from Gemini…"):
                ensure_somatic_audio(profile, duration, intensity)

        audio = st.session_state.audio
        if audio is None:
            st.error(
                "Could not generate audio: "
                f"{st.session_state.somatic_error or 'Gemini is not configured for this deployment.'}"
            )
        else:
            st.audio(audio.wav_bytes, format="audio/wav")
            st.caption(f"Mode: {audio.mode} · {audio.sample_rate:,} Hz · {audio.duration:.0f} s")

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
            "AI-generated psychoacoustic atmosphere (primary mode)",
            "The primary audio engine sends the resolved physical and web-retrieved profile to Gemini, which returns "
            "a two-sentence psychological presence narrative plus seven bounded atmosphere vectors. The Python "
            "NumPy engine renders those vectors directly: abyss/expansion, tension/dissonance, suffocation/friction, "
            "and thermal presence. Gemini is instructed not to invent physical measurements."
        ),
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
            "AI-generated or interpretive representation and avoids fabricating measurements."
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
GEMINI SOMATIC SOUND DESIGN (PRIMARY)
  ↓
NUMPY PSYCHOACOUSTIC RENDERING
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
