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
APP_VERSION = "1.0.0"

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
            model=model_secret("GEMINI_FLASH_MODEL", GEMINI_FLASH_MODEL),
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
            model=model_secret("GEMINI_PRO_MODEL", GEMINI_PRO_MODEL),
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

section[data-testid="stSidebar"] {
  border-right: 1px solid var(--line);
  background: var(--paper);
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

div[data-testid="stTextInput"] input {
  border: 1px solid #bfc4cb;
  border-radius: 0;
  min-height: 3.8rem;
  font-size: 1.1rem;
  background: white;
}

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


# Sidebar navigation
with st.sidebar:
    st.markdown(f"### {APP_NAME}")
    st.caption("Astronomical audio exploration")
    page = st.radio(
        "Navigate",
        ["Home", "Object", "Audio Lab", "Favorites", "Data / Export", "Scientific Method", "About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Search remains the central interaction.")
    if st.session_state.recent:
        st.markdown("**Recent searches**")
        for recent in st.session_state.recent[-6:][::-1]:
            if st.button(recent, key=f"recent_{recent}", use_container_width=True):
                st.session_state.selected_query = recent
                st.session_state.profile = build_profile(recent)
                st.session_state.audio = None
                st.rerun()


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
                    st.session_state.selected_query = selected
                    st.session_state.search = selected
                    st.session_state.profile = build_profile(selected)
                    st.session_state.audio = None
                    if selected not in st.session_state.recent:
                        st.session_state.recent.append(selected)
                    st.rerun()

        if st.button(f'Search astronomical databases for "{search.strip()}"', type="primary", use_container_width=True):
            selected = search.strip()
            st.session_state.selected_query = selected
            st.session_state.profile = build_profile(selected)
            st.session_state.audio = None
            if selected not in st.session_state.recent:
                st.session_state.recent.append(selected)
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

        if st.button("Generate auditory representation", type="primary", use_container_width=True):
            with st.spinner("Building deterministic scientific audio…"):
                audio = synthesize_audio(profile)
                plan = gemini_plan(profile)
                if plan:
                    profile.audio_mode = plan.data_basis
                    profile.audio_explanation = (
                        f"{plan.primary_phenomenon}. {plan.recommended_mapping} "
                        f"Listener focus: {plan.listener_focus} "
                        f"Limitations: {plan.limitations}"
                    )
                    profile.limitations = [plan.limitations]
                    st.session_state.profile = profile
                audio = synthesize_audio(profile)
                st.session_state.audio = audio
            st.success("Auditory representation generated.")

        st.markdown('<div class="su-section">Why does this sound like this?</div>', unsafe_allow_html=True)
        if st.session_state.audio:
            st.audio(st.session_state.audio.wav_bytes, format="audio/wav")
            st.write(st.session_state.audio.explanation)
        else:
            st.write("Generate the auditory representation to hear the object-specific model.")

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
