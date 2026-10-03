"""
Sounds of the Universe - Environmental Sonification for Blind and Visually
Impaired (BVI) Listeners
-----------------------------------------------------------------------------
PRODUCT GOAL (not "make space sounds"): use sound to help a listener who
cannot see an environment feel what it might be like to stand there -
scale, openness, enclosure, density, stillness, movement, temperature,
pressure, isolation, calm, tension, vastness - without pretending there is
one objectively correct emotional response to any place.

ARCHITECTURE
    USER INPUT
      -> OBJECT IDENTIFICATION (free text, no catalogue)
      -> PHYSICAL FACTS, each carrying its own confidence/provenance
           (cross-checked against embedded reference data where the object
            is a well-known body; otherwise explicitly marked as reasoning,
            not verified fact)
      -> DETERMINISTIC PHYSICS MODEL (SurfacePhysics)
      -> DETERMINISTIC PERCEPTUAL MODEL (PerceptualState: 12 independent
           0..1 dimensions derived only from the physics, never from a
           single emotion label)
      -> PSYCHOACOUSTIC VECTORS proposed by Gemini, then VALIDATED and
           PARTLY GROUNDED against the deterministic perceptual model, then
           hard-capped by physical plausibility rules that have final say
      -> DSP SYNTHESIS (NumPy/SciPy)
      -> AUDIO MASTERING & VALIDATION (DC removal, true-peak estimate,
           approximate loudness, clipping/NaN checks, safety limiting)
      -> IMMERSIVE STEREO PLAYBACK, with a parallel text description for
           screen readers

Gemini's role is explicitly a PROPOSAL, not a scientific authority: it
states the physical boundary conditions of the place a listener would
stand and a candidate psychoacoustic interpretation, but every number it
returns is checked - against embedded reference data for well-known
bodies, and against hard physical-plausibility rules for everything else -
before it is allowed to reach the synthesiser. Confidence is tracked
explicitly as VERIFIED / DERIVED / ESTIMATED / HYPOTHETICAL / UNKNOWN and
shown to the user; nothing unknown is silently turned into an invented
number, and no single environment is ever reduced to one emotion label
("Mars = sad"). Silence itself is a deliberate design tool: an airless
world is allowed to be sparse and near-silent rather than padded with a
generic drone.

Three modes (toggle in the UI):
    EXPERIENCE  - minimal explanation, just listen.
    UNDERSTAND  - adds a plain-language "why does it sound like this?"
                  causal explanation, built only from transformations the
                  engine actually performed.
    SCIENCE     - exposes full provenance, the perceptual-state breakdown,
                  raw psychoacoustic vectors, audio metadata/validation
                  results, the deterministic soundscape ID, and an optional
                  anonymous feedback form for future human evaluation.

Failure handling is honest, never generic:
  * The selected Gemini model gets a few retries. Errors are classified
    from the real API response (HTTP code / status), not guessed.
  * Only genuinely traffic-related failures (HTTP 429/5xx, timeouts,
    dropped connections) trigger one real attempt with the lighter
    Gemini 3.5 Flash Lite model - and the app says when and why.
  * Any other failure (bad request, missing model, bad key) is shown as-is.
    A "Technical details" panel always shows what Gemini returned.
  * There is no canned or generic sound anywhere in this app, and there is
    no fallback audio if a real soundscape cannot be produced.

Requirements:
    pip install -r requirements.txt   (streamlit>=1.56, numpy, scipy,
                                       pydantic>=2, google-genai)

Configuration (all optional except the API key):
    st.secrets["GEMINI_API_KEY"]      or env var GEMINI_API_KEY
    st.secrets["GEMINI_FLASH_MODEL"]  or env var GEMINI_FLASH_MODEL
    st.secrets["GEMINI_PRO_MODEL"]    or env var GEMINI_PRO_MODEL
    st.secrets["GEMINI_LITE_MODEL"]   or env var GEMINI_LITE_MODEL

Run:
    streamlit run app.py

Offline self-tests (no API key / network required - exercises the physics,
perceptual model, provenance logic and DSP engine directly, bypassing
Gemini entirely):
    python app.py --selftest
"""

import dataclasses
import hashlib
import io
import json
import math
import os
import re
import sys
import time
import wave
from collections import OrderedDict
from typing import Optional

import numpy as np
import streamlit as st
from pydantic import BaseModel, Field
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - the UI still loads without the SDK
    genai = None
    types = None


# ---------------------------------------------------------------------------
# 0. ENGINE VERSIONS - every generated soundscape carries these, so the same
#    inputs plus the same versions reproduce the same result, and a stale
#    cache entry from before a DSP/mastering change can never be served as
#    if it were newly generated.
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "2.0.0"              # CosmicSomaticConfig / AtmosphereVectors
PHYSICS_MODEL_VERSION = "2.0.0"       # SurfacePhysics + reconciliation rules
PROVENANCE_MODEL_VERSION = "1.0.0"    # confidence classification + reference data
PSYCHOACOUSTIC_VERSION = "2.0.0"      # PerceptualState + grounding/blend logic
DSP_SYNTHESIS_VERSION = "1.3.0"       # the NumPy/SciPy synthesis engine itself
MASTERING_VERSION = "1.0.0"           # post-synthesis mastering/validation

ENGINE_VERSIONS = {
    "schema": SCHEMA_VERSION,
    "physics_model": PHYSICS_MODEL_VERSION,
    "provenance_model": PROVENANCE_MODEL_VERSION,
    "psychoacoustic": PSYCHOACOUSTIC_VERSION,
    "dsp_synthesis": DSP_SYNTHESIS_VERSION,
    "mastering": MASTERING_VERSION,
}


# ---------------------------------------------------------------------------
# 1. APPLICATION CONFIGURATION & GLOBAL STATE MANAGEMENT
# ---------------------------------------------------------------------------

def _secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read a value from st.secrets, falling back to an environment variable,
    and never raising even if no secrets.toml exists at all."""
    try:
        if key in st.secrets:
            value = st.secrets[key]
            if value:
                return value
    except Exception:
        pass
    return os.environ.get(key, default)


GEMINI_FLASH_MODEL = _secret("GEMINI_FLASH_MODEL", "gemini-3.6-flash")
GEMINI_PRO_MODEL = _secret("GEMINI_PRO_MODEL", "gemini-3.1-pro-preview")
GEMINI_LITE_MODEL = _secret("GEMINI_LITE_MODEL", "gemini-3.5-flash-lite")

MODEL_OPTIONS = {
    "Gemini 3.6 Flash": GEMINI_FLASH_MODEL,
    "Gemini 3.1 Pro": GEMINI_PRO_MODEL,
    "Gemini 3.5 Flash Lite": GEMINI_LITE_MODEL,
}

GEMINI_MAX_ATTEMPTS = 3          # attempts on the selected model
GEMINI_FALLBACK_ATTEMPTS = 2     # attempts on the lite model, if needed
GEMINI_RETRY_BACKOFF_SECONDS = 1.4
GEMINI_HTTP_TIMEOUT_MS = 90_000  # never let a hung request freeze the UI

SOUNDSCAPE_SECONDS_DEFAULT = 24.0
SOUNDSCAPE_SECONDS_MIN = 8
SOUNDSCAPE_SECONDS_MAX = 60
MAX_QUERY_CHARS = 120             # basic input-size guard
CACHE_MAX_ENTRIES = 30            # practical safeguard against unbounded growth

FEEDBACK_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "feedback_log.jsonl"
)


def init_session_state() -> None:
    defaults = {
        "selected_query": None,
        "profile_data": None,
        "audio_bytes": None,
        "audio_metadata": None,
        "gemini_model": GEMINI_FLASH_MODEL,
        "search_error": None,
        "fallback_notice": None,
        "model_used": None,
        "gemini_diagnostics": [],
        "physics_adjustments": [],
        "duration": SOUNDSCAPE_SECONDS_DEFAULT,
        "provenance": None,
        "perceptual": None,
        "explanation_lines": [],
        "accessible_text": None,
        "emotional_sentence": None,
        "soundscape_id": None,
        "soundscape_cache": OrderedDict(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _model_label(model_value: Optional[str]) -> str:
    for label, value in MODEL_OPTIONS.items():
        if value == model_value:
            return label
    return str(model_value)


class GeminiUnavailableError(Exception):
    """Raised when Gemini cannot produce a real psychoacoustic profile.

    reason:
      "not_configured" - no API key / SDK, nothing was sent
      "traffic"        - 429 / 5xx / timeout / connection problems
      "rejected"       - the request itself was refused (400/401/403/404...)
      "bad_response"   - Gemini answered, but not with usable schema JSON
    """

    def __init__(self, message: str, reason: str = "traffic",
                 short_reason: str = "", details: Optional[list] = None):
        super().__init__(message)
        self.reason = reason
        self.short_reason = short_reason
        self.details = list(details or [])


# ---------------------------------------------------------------------------
# 2. PYDANTIC STRUCTURED OUTPUT SCHEMA
#    This is Gemini's PROPOSAL, not a scientific record. Every physical field
#    is independently re-checked in section 2b/2c below; every psychoacoustic
#    vector is treated as a creative suggestion that is validated and partly
#    grounded against a deterministic perceptual model before it is allowed
#    anywhere near the DSP engine (see PerceptualState, section 2c).
# ---------------------------------------------------------------------------

class AtmosphereVectors(BaseModel):
    # 1. Emotional Temperature & Materiality (Cold Isolation vs. Blazing Chaos)
    hollow_isolation_factor: float = Field(
        description=(
            "0.0 to 1.0. High values represent freezing voids or dark nebulae. "
            "Generates vast, empty, resonant reverb tails and sweeping wind "
            "filters to trigger deep mental loneliness."
        )
    )
    thermal_manic_hum: float = Field(
        description=(
            "0.0 to 1.0. High values represent extreme heat/stars: turbulent, "
            "roaring convective heat that creates intense physical and mental "
            "agitation. Must be near 0.0 for any surface cooler than about "
            "350 K."
        )
    )

    # 2. Mental Tension & Threat (Safe/Calm vs. Hostile/Terrifying)
    harmonic_dissonance_index: float = Field(
        description=(
            "0.0 to 1.0. High values represent violent environments (black "
            "holes, supernovae). Introduces clashing intervals (minor "
            "seconds/tritones) to trigger physiological and mental "
            "threat/danger. Near 0.0 for calm, habitable places."
        )
    )
    rhythmic_unpredictability: float = Field(
        description=(
            "0.0 to 1.0. Controls how often and how irregularly discrete "
            "events (birdsong, cracking, gusts, flares, debris) occur in the "
            "timeline. Calm living places have a gentle, steady scatter."
        )
    )

    # 3. Spatial Scale & Presence (Claustrophobic Enclosure vs. Infinite Vastness)
    claustrophobic_suffocation: float = Field(
        description=(
            "0.0 to 1.0. High values represent dense, high-pressure "
            "atmospheres (like Venus or gas giant cores). Activates a rolling "
            "low-pass filter that swallows bright frequencies, wrapping the "
            "listener in a heavy, muffled cage. Must be near 0.0 in a "
            "vacuum or thin atmosphere."
        )
    )
    infinite_spatial_drift: float = Field(
        description=(
            "0.0 to 1.0. Controls a slow, deep LFO sweep over a wide stereo "
            "field: how open and unanchored the horizon feels."
        )
    )

    # 4. Fluid Friction (Magnetic Fields, Tearing Winds)
    tearing_friction_index: float = Field(
        description=(
            "0.0 to 1.0. Turbulent shear of moving fluid: tearing winds, "
            "violent storms, plasma streaming past a magnetic boundary. "
            "Near 0.0 wherever there is no moving fluid at the surface."
        )
    )


class CosmicSomaticConfig(BaseModel):
    """Field ORDER matters: the physical facts come first so Gemini commits to
    them before it is allowed to choose anything psychological."""

    object_name: str = Field(
        description="Canonical name of the object the listener typed."
    )
    surface_gravity_g: float = Field(
        description=(
            "Literal surface gravity in units of Earth gravity (Earth = 1.0) at "
            "the location you chose. For objects with no solid surface use the "
            "closest physically meaningful boundary (the 1-bar level of a gas "
            "giant, the photosphere of a star) and say so in "
            "surface_composition_notes."
        )
    )
    surface_temperature_k: float = Field(
        description="Literal temperature in kelvin at that same location."
    )
    atmospheric_pressure_atm: float = Field(
        description=(
            "Atmospheric pressure at that location in atmospheres: 0.0 for a "
            "vacuum, 1.0 for Earth at sea level, about 92 for the surface of "
            "Venus. Use a tiny positive number for a trace exosphere."
        )
    )
    surface_composition_notes: str = Field(
        description=(
            "One or two brief, highly factual sentences on what is physically "
            "at the listener's feet and in the air around them (e.g. soil, "
            "grass and open water; basaltic dust; water ice over a frozen "
            "ocean; hot ionised plasma). Begin with 'UNVERIFIED:' if you do "
            "not genuinely know this object and are inferring it from its "
            "most likely class."
        )
    )
    mental_presence_narrative: str = Field(
        description=(
            "A highly artistic, emotionally vivid 2-sentence description "
            "detailing exactly what it feels like mentally and psychologically "
            "to stand inside this environment, built only from the physics "
            "stated above."
        )
    )
    vectors: AtmosphereVectors
    grounding_tone_hz: float = Field(
        description=(
            "Pitch, 55 to 330 Hz, of the soft tonal ground the listener's mind "
            "rests on. Massive, dense or heavy places sit low (55-90 Hz); "
            "tranquil habitable worlds around 110-165 Hz; light, thin, small "
            "places higher (180-330 Hz)."
        )
    )


VECTOR_NAMES = (
    "hollow_isolation_factor", "thermal_manic_hum", "harmonic_dissonance_index",
    "rhythmic_unpredictability", "claustrophobic_suffocation",
    "infinite_spatial_drift", "tearing_friction_index",
)


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _bounded(x, lo: float, hi: float, default: float) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(x):
        return default
    return float(max(lo, min(hi, x)))


def _validate_physics(cfg: CosmicSomaticConfig) -> None:
    """Refuse a response whose physical fields are missing or nonsensical.
    Raising here makes the caller retry instead of synthesising a guess."""
    for name in ("surface_gravity_g", "surface_temperature_k", "atmospheric_pressure_atm"):
        value = getattr(cfg, name, None)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} is missing or not a finite number")
    if cfg.surface_temperature_k <= 0:
        raise ValueError("surface_temperature_k must be above absolute zero")
    if cfg.surface_gravity_g < 0 or cfg.atmospheric_pressure_atm < 0:
        raise ValueError("gravity and pressure cannot be negative")
    if not str(cfg.surface_composition_notes).strip():
        raise ValueError("surface_composition_notes is empty")


def _sanitize_config(config: CosmicSomaticConfig) -> CosmicSomaticConfig:
    """Clamp every number into a legal range so the DSP engine can always
    trust its inputs (idempotent)."""
    v = config.vectors
    vectors = AtmosphereVectors(**{
        name: _bounded(getattr(v, name), 0.0, 1.0, 0.0) for name in VECTOR_NAMES
    })
    return CosmicSomaticConfig(
        object_name=str(config.object_name).strip() or "Unknown object",
        surface_gravity_g=_bounded(config.surface_gravity_g, 0.0, 1e12, 1.0),
        surface_temperature_k=_bounded(config.surface_temperature_k, 0.5, 1e9, 288.0),
        atmospheric_pressure_atm=_bounded(config.atmospheric_pressure_atm, 0.0, 1e9, 0.0),
        surface_composition_notes=" ".join(str(config.surface_composition_notes).split())[:400],
        mental_presence_narrative=str(config.mental_presence_narrative).strip(),
        vectors=vectors,
        grounding_tone_hz=_bounded(config.grounding_tone_hz, 55.0, 330.0, 110.0),
    )


# ---------------------------------------------------------------------------
# 2b. PHYSICS INTERPRETATION: what the stated conditions actually allow
# ---------------------------------------------------------------------------
# These keyword lists only decide WHICH natural sources can exist (a frozen
# surface can crack, a molten one can bubble). Every quantity - how deep the
# air rolls, how bright it is, how fast things settle - comes from the
# numbers Gemini stated.

_LIFE_WORDS = ("vegetation", "forest", "grass", "meadow", "soil", "jungle", "tundra", "moss",
               "lichen", "plant", "tree", "biosphere", "ocean", "river", "lake", "coast",
               "wetland", "savanna", "farmland", "shore", "beach")
_ICE_WORDS = ("ice", "frost", "frozen", "snow", "glacier", "permafrost")
_MOLTEN_WORDS = ("lava", "magma", "molten", "volcan")
_HOT_WORDS = ("plasma", "photosphere", "corona", "chromosphere", "accretion", "ionised",
              "ionized", "lightning", "aurora", "flare", "convect")


def _has_any(text: str, words) -> bool:
    return any(w in text for w in words)


@dataclasses.dataclass(frozen=True)
class SurfacePhysics:
    gravity: float        # Earth g
    temperature: float    # K
    pressure: float       # atm
    notes: str
    air: float            # gas density relative to Earth sea level (ideal gas: P/T)
    presence: float       # 0..1 how well airborne sound is carried
    airborne: bool        # is there an atmosphere to carry sound at all
    habitable: bool       # Earth-like, calm, temperate
    biological: bool      # plausibly living surface
    energetic: bool       # ionised / very hot / electrically active
    icy: bool
    molten: bool
    dense: bool           # crushing atmosphere
    sound_speed: float    # relative to 288 K air (proportional to sqrt(T))
    unverified: bool


def _surface_physics(cfg: CosmicSomaticConfig) -> SurfacePhysics:
    g, T, p = cfg.surface_gravity_g, cfg.surface_temperature_k, cfg.atmospheric_pressure_atm
    notes = cfg.surface_composition_notes.lower()
    air = p * 288.0 / max(T, 3.0)
    airborne = p >= 0.002
    presence = float(np.clip(math.log10(1.0 + air * 40.0) / 2.2, 0.0, 1.0)) if airborne else 0.0
    habitable = (0.35 <= p <= 3.0) and (245.0 <= T <= 318.0) and (0.3 <= g <= 2.5)
    biological = habitable or (_has_any(notes, _LIFE_WORDS) and 230.0 <= T <= 335.0 and p >= 0.1)
    molten = T > 600.0 and _has_any(notes, _MOLTEN_WORDS)
    energetic = T > 1800.0 or _has_any(notes, _HOT_WORDS)
    icy = (T < 273.15 and _has_any(notes, _ICE_WORDS)) or T < 120.0
    return SurfacePhysics(
        gravity=g, temperature=T, pressure=p, notes=notes, air=air, presence=presence,
        airborne=airborne, habitable=habitable, biological=biological,
        energetic=energetic, icy=icy, molten=molten, dense=air >= 12.0,
        sound_speed=float(np.clip(math.sqrt(T / 288.0), 0.5, 2.0)),
        unverified=cfg.surface_composition_notes.strip().upper().startswith("UNVERIFIED"),
    )


def _reconcile_with_physics(config: CosmicSomaticConfig):
    """Check every vector against the physics Gemini just stated and correct
    anything impossible (heat with no temperature, suffocating air in a
    vacuum...). Returns (config, list of human-readable corrections). This is
    the deterministic validation layer that has final authority over the
    LLM's psychoacoustic proposal (see product architecture note at the top
    of this file)."""
    phys = _surface_physics(config)
    v = {name: getattr(config.vectors, name) for name in VECTOR_NAMES}
    notes = []

    def cap(name, limit, why):
        if v[name] > limit + 1e-9:
            notes.append(f"{name.replace('_', ' ')} lowered {v[name]:.2f} -> {limit:.2f}: {why}")
            v[name] = limit

    def floor(name, limit, why):
        if v[name] < limit - 1e-9:
            notes.append(f"{name.replace('_', ' ')} raised {v[name]:.2f} -> {limit:.2f}: {why}")
            v[name] = limit

    T = phys.temperature
    heat_cap = float(np.clip((math.log10(T) - 1.8) / 2.2, 0.05, 1.0))
    cap("thermal_manic_hum", heat_cap, f"a {T:.0f} K surface has little heat to drive it")
    if phys.habitable:
        cap("thermal_manic_hum", 0.2, "habitable, temperate surface")
        cap("harmonic_dissonance_index", 0.2, "a habitable surface is not hostile")
        cap("claustrophobic_suffocation", 0.3, "breathable-pressure air is not crushing")
    if not phys.airborne:
        cap("claustrophobic_suffocation", 0.2, "no atmosphere to press on the listener")
        if T < 1800.0:
            cap("tearing_friction_index", 0.2, "no moving fluid to shear against")
    if phys.dense:
        floor("claustrophobic_suffocation", 0.55,
              f"air here is about {phys.air:.0f}x Earth sea-level density")

    rebuilt = CosmicSomaticConfig(
        object_name=config.object_name,
        surface_gravity_g=config.surface_gravity_g,
        surface_temperature_k=config.surface_temperature_k,
        atmospheric_pressure_atm=config.atmospheric_pressure_atm,
        surface_composition_notes=config.surface_composition_notes,
        mental_presence_narrative=config.mental_presence_narrative,
        vectors=AtmosphereVectors(**v),
        grounding_tone_hz=config.grounding_tone_hz,
    )
    return rebuilt, notes


def _choose_material(phys: SurfacePhysics) -> str:
    """Which physical substance dominates the bed of sound, from the physics."""
    if phys.molten or (phys.energetic and phys.temperature > 1800.0):
        return "plasma"        # roaring, crackling hot matter
    if not phys.airborne:
        return "crystal" if phys.icy else "stone"   # nothing to carry air sound
    if phys.dense:
        return "rumble"        # crushing atmosphere: deep pressure
    return "wind"


def _event_plan(phys: SurfacePhysics):
    """(kind, events-per-second at neutral unpredictability) for this place."""
    plan = []
    if phys.biological:
        plan += [("bird", 0.32), ("rustle", 0.10)]
    if phys.energetic:
        plan += [("whistler", 0.14), ("chorus", 0.10)]
    if phys.molten:
        plan += [("bubble", 0.70), ("pop", 0.90)]
    if phys.icy:
        plan += [("crack", 0.22), ("tick", 0.50)]
    if not (phys.airborne or phys.icy or phys.molten or phys.energetic):
        plan += [("tick", 0.45), ("creak", 0.05)]
    if phys.airborne and not (phys.biological or phys.energetic):
        plan += [("gust", 0.14)]
    if phys.dense:
        plan += [("swell", 0.10)]
    return plan or [("tick", 0.30)]


def _is_tranquil(phys: SurfacePhysics, v: AtmosphereVectors) -> bool:
    return phys.habitable or (
        v.harmonic_dissonance_index < 0.2 and v.thermal_manic_hum < 0.2
        and v.rhythmic_unpredictability < 0.4
    )


def _phenomena_report(phys: SurfacePhysics) -> list:
    """Transparency list for Understand/Science mode: which categories of
    physical sound phenomenon are allowed here and why, including the ones
    that were explicitly suppressed because the stated physics forbids them
    (section 7 of the product spec: the validation layer has final say over
    what is allowed to sound, never the other way around)."""
    report = []
    report.append(("Airborne sound (wind, open air)", phys.airborne,
                    "an atmosphere is present to carry it" if phys.airborne
                    else "no meaningful atmosphere is present to carry it"))
    report.append(("Living/biological sound (birdsong, rustling)", phys.biological,
                    "temperature and pressure fall in a plausibly habitable range"
                    if phys.biological else
                    "temperature, pressure or composition rule out a living surface"))
    report.append(("Molten/volcanic activity (bubbling, popping)", phys.molten,
                    "the stated temperature and composition indicate molten material"
                    if phys.molten else "no molten material is indicated"))
    report.append(("Ice cracking / settling", phys.icy,
                    "the stated temperature and composition indicate a frozen surface"
                    if phys.icy else "no frozen surface is indicated"))
    report.append(("Ionised/electrical activity (plasma, flares)", phys.energetic,
                    "extreme heat or an ionised composition is stated"
                    if phys.energetic else "no extreme heat or ionisation is stated"))
    report.append(("Crushing/dense-atmosphere pressure texture", phys.dense,
                    f"the stated air is roughly {phys.air:.0f}x Earth sea-level density"
                    if phys.dense else "the stated atmosphere is not unusually dense"))
    return report


def _engine_summary(config: CosmicSomaticConfig) -> dict:
    """What the engine derived from the stated physics - shown to the user."""
    phys = _surface_physics(config)
    return {
        "material": _choose_material(phys),
        "events": [k for k, _ in _event_plan(phys)],
        "tranquil": _is_tranquil(phys, config.vectors),
        "airborne": phys.airborne,
        "air_density": phys.air,
        "unverified": phys.unverified,
    }


# ---------------------------------------------------------------------------
# 2c. SCIENTIFIC PROVENANCE
#    Gemini's stated physical values are never treated as ground truth. For
#    well-known bodies this app independently cross-checks them against a
#    small embedded reference table and corrects them if they disagree; for
#    anything else they remain explicitly labelled as reasoning, not fact.
# ---------------------------------------------------------------------------

CONFIDENCE_LEVELS = ("VERIFIED", "DERIVED", "ESTIMATED", "HYPOTHETICAL", "UNKNOWN")

CONFIDENCE_EXPLANATION = {
    "VERIFIED": "Cross-checked by this app against embedded reference data.",
    "DERIVED": "Mathematically calculated from a verified value.",
    "ESTIMATED": "Gemini's scientific reasoning; not independently verified by this app.",
    "HYPOTHETICAL": "An explicit assumption for a fictional or hypothetical object.",
    "UNKNOWN": "No defensible value is available.",
}


@dataclasses.dataclass(frozen=True)
class PhysicalValue:
    value: float
    unit: str
    confidence: str
    source: str
    notes: str = ""


@dataclasses.dataclass(frozen=True)
class Provenance:
    gravity: PhysicalValue
    temperature: PhysicalValue
    pressure: PhysicalValue
    overall_confidence: str
    matched_reference: Optional[str]
    unverified_flag: bool


# A small, deliberately conservative reference table of well-known bodies.
# Values are approximate figures consistent with commonly published
# planetary fact sheets, not a live data feed - the app says so plainly.
# Several bodies have genuinely large natural variation (Mercury and the
# Moon's day/night swing, Mars's seasonal range); that variation is noted
# rather than hidden behind a falsely precise single number.
REFERENCE_BODIES = {
    "earth": dict(
        gravity_g=1.0, temperature_k=288.0, pressure_atm=1.0,
        source="Standard terrestrial reference values (approximate global averages)",
        temp_note="Global mean surface temperature; varies by location and season.",
        pressure_note="Mean sea-level pressure.",
    ),
    "mars": dict(
        gravity_g=0.38, temperature_k=210.0, pressure_atm=0.0063,
        source="Approximate values consistent with published Mars fact sheets",
        temp_note="Mean surface temperature; varies roughly 130-300 K with "
                   "latitude, season and time of day.",
        pressure_note="Mean surface pressure; varies seasonally by roughly 20%.",
    ),
    "venus": dict(
        gravity_g=0.904, temperature_k=737.0, pressure_atm=92.0,
        source="Approximate values consistent with published Venus fact sheets",
        temp_note="Surface temperature is relatively uniform due to the thick atmosphere.",
        pressure_note="Surface pressure near the mean planetary radius.",
    ),
    "mercury": dict(
        gravity_g=0.38, temperature_k=340.0, pressure_atm=1e-14,
        source="Approximate values consistent with published Mercury fact sheets",
        temp_note="Surface temperature swings roughly 100-700 K between night "
                   "and day; 340 K is a rough midpoint, not a single measured value.",
        pressure_note="Essentially a vacuum (trace exosphere).",
    ),
    "moon": dict(
        gravity_g=0.166, temperature_k=220.0, pressure_atm=3e-15,
        source="Approximate values consistent with published lunar fact sheets",
        temp_note="Surface temperature swings roughly 100-390 K between lunar "
                   "night and day; 220 K is a rough midpoint.",
        pressure_note="Essentially a vacuum (trace exosphere).",
    ),
    "jupiter": dict(
        gravity_g=2.53, temperature_k=165.0, pressure_atm=1.0,
        source="Approximate values at the conventional 1-bar reference level (no solid surface)",
        temp_note="Temperature at the 1-bar level; Jupiter has no solid surface.",
        pressure_note="1 bar is the conventional reference level by definition.",
    ),
    "saturn": dict(
        gravity_g=1.065, temperature_k=134.0, pressure_atm=1.0,
        source="Approximate values at the conventional 1-bar reference level (no solid surface)",
        temp_note="Temperature at the 1-bar level; Saturn has no solid surface.",
        pressure_note="1 bar is the conventional reference level by definition.",
    ),
    "uranus": dict(
        gravity_g=0.886, temperature_k=76.0, pressure_atm=1.0,
        source="Approximate values at the conventional 1-bar reference level (no solid surface)",
        temp_note="Temperature at the 1-bar level; Uranus has no solid surface.",
        pressure_note="1 bar is the conventional reference level by definition.",
    ),
    "neptune": dict(
        gravity_g=1.14, temperature_k=72.0, pressure_atm=1.0,
        source="Approximate values at the conventional 1-bar reference level (no solid surface)",
        temp_note="Temperature at the 1-bar level; Neptune has no solid surface.",
        pressure_note="1 bar is the conventional reference level by definition.",
    ),
    "pluto": dict(
        gravity_g=0.063, temperature_k=44.0, pressure_atm=1e-5,
        source="Approximate values consistent with published Pluto fact sheets",
        temp_note="Surface temperature varies roughly 33-55 K.",
        pressure_note="Very thin, seasonally variable nitrogen atmosphere.",
    ),
    "titan": dict(
        gravity_g=0.14, temperature_k=94.0, pressure_atm=1.45,
        source="Approximate values consistent with published Titan fact sheets",
        temp_note="Surface temperature is relatively stable due to the thick atmosphere.",
        pressure_note="Titan's surface pressure is unusually higher than Earth's.",
    ),
    "europa": dict(
        gravity_g=0.134, temperature_k=102.0, pressure_atm=1e-15,
        source="Approximate values consistent with published Europa fact sheets",
        temp_note="Surface temperature varies roughly 50-125 K.",
        pressure_note="Essentially a vacuum (trace oxygen exosphere).",
    ),
    "io": dict(
        gravity_g=0.183, temperature_k=110.0, pressure_atm=1e-15,
        source="Approximate values consistent with published Io fact sheets",
        temp_note="Background surface temperature; active volcanic hotspots are far hotter.",
        pressure_note="Essentially a vacuum (trace sulfur dioxide exosphere).",
    ),
    "enceladus": dict(
        gravity_g=0.0113, temperature_k=75.0, pressure_atm=1e-15,
        source="Approximate values consistent with published Enceladus fact sheets",
        temp_note="Mean surface temperature; the south-polar vent region is warmer.",
        pressure_note="Essentially a vacuum.",
    ),
    "triton": dict(
        gravity_g=0.0797, temperature_k=38.0, pressure_atm=1.4e-5,
        source="Approximate values consistent with published Triton fact sheets",
        temp_note="One of the coldest known surfaces in the solar system.",
        pressure_note="Very thin nitrogen atmosphere.",
    ),
    "sun": dict(
        gravity_g=27.9, temperature_k=5778.0, pressure_atm=1.0,
        pressure_comparable=False,
        source="Approximate photospheric values consistent with published solar fact sheets",
        temp_note="Effective photospheric temperature.",
        pressure_note="The photosphere is a tenuous plasma; a terrestrial-style "
                       "pressure-in-atmospheres figure is not physically meaningful "
                       "here, so pressure is not independently verified even though "
                       "gravity and temperature are.",
    ),
}

_REFERENCE_ALIASES = {
    "the earth": "earth", "planet earth": "earth", "our home planet": "earth",
    "the moon": "moon", "earth's moon": "moon", "luna": "moon",
    "the sun": "sun", "sol": "sun", "our sun": "sun", "the star sol": "sun",
    "the red planet": "mars", "red planet": "mars",
}


def _normalize_object_key(raw: str) -> str:
    s = (raw or "").strip().lower()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    s = re.sub(r"[^a-z0-9\s'-]", "", s)
    return " ".join(s.split())


def _match_reference(raw: str) -> Optional[str]:
    key = _normalize_object_key(raw)
    key = _REFERENCE_ALIASES.get(key, key)
    if key in REFERENCE_BODIES:
        return key
    return None


def _within_tolerance_linear(a: float, b: float, tol: float = 0.35) -> bool:
    denom = max(abs(b), 1e-9)
    return abs(a - b) / denom <= tol


def _within_tolerance_log(a: float, b: float, tol_decades: float = 0.5) -> bool:
    a = max(abs(a), 1e-18)
    b = max(abs(b), 1e-18)
    return abs(math.log10(a) - math.log10(b)) <= tol_decades


_HYPOTHETICAL_WORDS = ("hypothetical", "fictional", "imaginary", "made up", "made-up",
                        "invented", "what if", "a world where", "a planet where")


def _build_provenance(raw_query: str, cfg: CosmicSomaticConfig):
    """Independently classifies confidence for the three core physical
    quantities, cross-checking well-known bodies against REFERENCE_BODIES
    and correcting Gemini's values if they disagree. Returns
    (Provenance, possibly-corrected config, list of correction notes)."""
    unverified_flag = cfg.surface_composition_notes.strip().upper().startswith("UNVERIFIED")
    hypothetical_hint = any(w in raw_query.lower() for w in _HYPOTHETICAL_WORDS)
    matched = _match_reference(raw_query) or _match_reference(cfg.object_name)
    corrections = []

    if matched:
        ref = REFERENCE_BODIES[matched]
        pressure_comparable = ref.get("pressure_comparable", True)

        g_ok = _within_tolerance_linear(cfg.surface_gravity_g, ref["gravity_g"])
        t_ok = _within_tolerance_linear(cfg.surface_temperature_k, ref["temperature_k"], tol=0.40)
        p_ok = (not pressure_comparable) or _within_tolerance_log(
            cfg.atmospheric_pressure_atm, ref["pressure_atm"])

        gravity_val = cfg.surface_gravity_g if g_ok else ref["gravity_g"]
        temp_val = cfg.surface_temperature_k if t_ok else ref["temperature_k"]
        pressure_val = cfg.atmospheric_pressure_atm if (p_ok or not pressure_comparable) else ref["pressure_atm"]

        if not g_ok:
            corrections.append(
                f"surface gravity corrected to reference value ({ref['gravity_g']:.2f} g) "
                f"for {matched.title()}")
        if not t_ok:
            corrections.append(
                f"surface temperature corrected to reference value ({ref['temperature_k']:.0f} K) "
                f"for {matched.title()}")
        if pressure_comparable and not p_ok:
            corrections.append(
                f"atmospheric pressure corrected to reference value ({ref['pressure_atm']:.3g} atm) "
                f"for {matched.title()}")

        if (gravity_val != cfg.surface_gravity_g or temp_val != cfg.surface_temperature_k
                or pressure_val != cfg.atmospheric_pressure_atm):
            cfg = cfg.model_copy(update={
                "surface_gravity_g": gravity_val,
                "surface_temperature_k": temp_val,
                "atmospheric_pressure_atm": pressure_val,
            })

        pressure_conf = "VERIFIED" if pressure_comparable else "ESTIMATED"
        prov = Provenance(
            gravity=PhysicalValue(gravity_val, "g (Earth=1)", "VERIFIED", ref["source"],
                                   notes=""),
            temperature=PhysicalValue(temp_val, "K", "VERIFIED", ref["source"],
                                       notes=ref.get("temp_note", "")),
            pressure=PhysicalValue(pressure_val, "atm", pressure_conf, ref["source"],
                                    notes=ref.get("pressure_note", "")),
            overall_confidence="VERIFIED",
            matched_reference=matched,
            unverified_flag=unverified_flag,
        )
        return prov, cfg, corrections

    conf = "HYPOTHETICAL" if hypothetical_hint else "ESTIMATED"
    source = "Gemini reasoning (not independently cross-checked by this app)"
    prov = Provenance(
        gravity=PhysicalValue(cfg.surface_gravity_g, "g (Earth=1)", conf, source),
        temperature=PhysicalValue(cfg.surface_temperature_k, "K", conf, source),
        pressure=PhysicalValue(cfg.atmospheric_pressure_atm, "atm", conf, source),
        overall_confidence=conf,
        matched_reference=None,
        unverified_flag=unverified_flag,
    )
    return prov, cfg, corrections


# ---------------------------------------------------------------------------
# 2d. PERCEPTUAL STATE - a deterministic, physics-grounded, multidimensional
#    model. This is a designed psychoacoustic interpretation, not a measured
#    or universal mapping of what any listener will actually feel, and it is
#    computed independently of anything Gemini proposes (section 6 of the
#    product spec: emotion must never collapse into a single label).
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PerceptualState:
    spatial_scale: float
    acoustic_density: float
    enclosure: float
    environmental_motion: float
    temporal_activity: float
    spectral_weight: float
    unpredictability: float
    resonance: float
    perceived_mass: float
    isolation: float
    proximity_activity: float
    distant_activity: float


def _derive_perceptual_state(phys: SurfacePhysics) -> PerceptualState:
    air_log = _clamp01(math.log10(1.0 + phys.air) / 1.3)
    presence = _clamp01(phys.presence)
    gravity_norm = _clamp01(phys.gravity / 2.5)
    heat_norm = _clamp01((phys.temperature - 150.0) / 700.0)

    spatial_scale = _clamp01(
        1.0 - 0.55 * presence - (0.25 if phys.habitable else 0.0)
        + (0.15 if phys.energetic else 0.0) + (0.10 if not phys.airborne else 0.0))
    acoustic_density = _clamp01(0.15 + 0.55 * air_log + (0.20 if phys.dense else 0.0))
    enclosure = _clamp01(air_log * (1.3 if phys.dense else 0.55))
    environmental_motion = _clamp01(
        0.15 + 0.50 * air_log * (1.0 if phys.airborne else 0.0)
        + (0.35 if (phys.molten or phys.energetic) else 0.0))
    temporal_activity = _clamp01(
        0.20 + 0.50 * (1.0 if phys.biological else 0.0)
        + 0.45 * (1.0 if phys.molten else 0.0)
        + 0.35 * (1.0 if phys.energetic else 0.0)
        + (0.15 if phys.icy else 0.0))
    spectral_weight = _clamp01(0.50 * air_log + 0.30 * (1.0 if phys.dense else 0.0)
                                + 0.20 * (1.0 - heat_norm))
    unpredictability = _clamp01(
        0.15 + 0.35 * (1.0 if phys.molten else 0.0)
        + 0.30 * (1.0 if phys.energetic else 0.0)
        + 0.15 * (1.0 if phys.icy else 0.0)
        + (0.05 if phys.biological else 0.0))
    resonance = _clamp01(0.6 * spatial_scale + 0.2 * (1.0 - acoustic_density)
                          + (0.2 if not phys.airborne else 0.0))
    perceived_mass = _clamp01(0.30 * gravity_norm + 0.40 * air_log
                               + (0.30 if phys.dense else 0.0))
    isolation = _clamp01(1.0 - (0.6 if phys.biological else 0.0)
                          - (0.2 if phys.habitable else 0.0) + 0.3 * (1.0 - presence))
    proximity_activity = _clamp01(0.6 * temporal_activity + (0.3 if phys.biological else 0.0))
    distant_activity = _clamp01(0.5 * spatial_scale + 0.3 * isolation)

    return PerceptualState(
        spatial_scale=spatial_scale, acoustic_density=acoustic_density, enclosure=enclosure,
        environmental_motion=environmental_motion, temporal_activity=temporal_activity,
        spectral_weight=spectral_weight, unpredictability=unpredictability, resonance=resonance,
        perceived_mass=perceived_mass, isolation=isolation,
        proximity_activity=proximity_activity, distant_activity=distant_activity,
    )


def _apply_perceptual_grounding(config: CosmicSomaticConfig, perceptual: PerceptualState):
    """Blends the two most physically-grounded psychoacoustic vectors toward
    the independent, deterministic PerceptualState baseline, then re-applies
    every hard physical cap so blending can never reintroduce an
    impossibility. This gives the deterministic physical/perceptual layer
    real, final authority over the LLM's proposal rather than just a label."""
    v = {name: getattr(config.vectors, name) for name in VECTOR_NAMES}
    notes = []

    target_hollow = _clamp01(0.6 * perceptual.isolation + 0.4 * perceptual.spatial_scale)
    target_suffocation = perceptual.enclosure

    blended_hollow = _clamp01(0.6 * v["hollow_isolation_factor"] + 0.4 * target_hollow)
    blended_suffocation = _clamp01(0.6 * v["claustrophobic_suffocation"] + 0.4 * target_suffocation)

    if abs(blended_hollow - v["hollow_isolation_factor"]) > 0.03:
        notes.append(
            f"hollow isolation factor grounded against the independent perceptual "
            f"model: {v['hollow_isolation_factor']:.2f} -> {blended_hollow:.2f}")
    if abs(blended_suffocation - v["claustrophobic_suffocation"]) > 0.03:
        notes.append(
            f"claustrophobic suffocation grounded against the independent perceptual "
            f"model: {v['claustrophobic_suffocation']:.2f} -> {blended_suffocation:.2f}")

    v["hollow_isolation_factor"] = blended_hollow
    v["claustrophobic_suffocation"] = blended_suffocation

    rebuilt = CosmicSomaticConfig(**{
        **config.model_dump(exclude={"vectors"}),
        "vectors": AtmosphereVectors(**v),
    })
    rebuilt, recap_notes = _reconcile_with_physics(rebuilt)
    return rebuilt, notes + recap_notes


DESCRIPTOR_MAP = {
    "spatial_scale": "vastness",
    "acoustic_density": "density",
    "enclosure": "a sense of enclosure",
    "environmental_motion": "movement",
    "temporal_activity": "liveliness",
    "spectral_weight": "heaviness",
    "unpredictability": "unpredictability",
    "resonance": "resonance and depth",
    "perceived_mass": "a felt sense of mass",
    "isolation": "solitude",
    "proximity_activity": "intimacy",
    "distant_activity": "distance and mystery",
}


def _emotional_sentence(perceptual: PerceptualState) -> str:
    """Multidimensional, explicitly hedged language - never a single emotion
    label, and never a claim that the response is universal (section 28)."""
    d = dataclasses.asdict(perceptual)
    ranked = sorted(d.items(), key=lambda kv: -kv[1])
    picks = [name for name, val in ranked if val >= 0.55][:3]
    if not picks:
        picks = [name for name, _ in ranked[:2]]
    phrases = [DESCRIPTOR_MAP[name] for name in picks]
    if len(phrases) == 1:
        joined = phrases[0]
    else:
        joined = ", ".join(phrases[:-1]) + " and " + phrases[-1]
    return (f"This soundscape is designed to evoke {joined}. Perception is "
            f"subjective - some listeners may experience it differently.")


def _explanation_lines(cfg: CosmicSomaticConfig, phys: SurfacePhysics, material: str,
                        adjustments: list) -> list:
    """'Why does THIS object sound like this?' Every line is anchored to the
    actual searched object by name and by its own stated numbers - never a
    generic, interchangeable category sentence - and only explains
    transformations the engine actually performed (section 16), never a
    fabricated causal story."""
    name = cfg.object_name
    lines = []

    # Always lead with a concrete, numeric, object-specific summary so this
    # section can never read as a content-free label wrapped around generic
    # text: whatever else is true, this one line is unique to this object.
    clean_notes = cfg.surface_composition_notes.replace("UNVERIFIED:", "").strip()
    lines.append(
        f"{name} is {cfg.surface_gravity_g:.2f}g, {cfg.surface_temperature_k:.0f} K and "
        f"{cfg.atmospheric_pressure_atm:.3g} atm of atmosphere - {clean_notes}"
    )

    material_reason = {
        "wind": (f"On {name}, that atmosphere is thin or temperate enough to carry ordinary "
                 f"airborne sound, so the base texture you hear is moving air."),
        "plasma": (f"On {name}, that heat and an ionised or molten composition mean the base "
                   f"texture is roaring, crackling plasma or molten matter rather than ordinary air."),
        "rumble": (f"On {name}, that air is roughly {phys.air:.0f}x Earth sea-level density, "
                   f"which presses in, so the base texture is a deep, contained pressure rumble "
                   f"instead of open wind."),
        "crystal": (f"On {name}, a frozen surface with little or no atmosphere means sound is "
                    f"sparse, high and mostly structure-borne rather than airborne."),
        "stone": (f"On {name}, with effectively no atmosphere there is nothing to carry "
                  f"conventional airborne sound, so this soundscape is kept deliberately "
                  f"near-silent rather than filled with an invented texture."),
    }
    if material in material_reason:
        lines.append(material_reason[material])
    if phys.habitable:
        lines.append(f"{name}'s calm, breathable, temperate surface keeps tension and agitation "
                      f"low here and favours gentle, organic events over mechanical-sounding ones.")
    if not phys.airborne:
        lines.append(f"{name} has no meaningful atmosphere: conventional wind and birdsong are "
                      f"suppressed; only faint structure-borne grit and settling remain.")
    if phys.dense:
        lines.append(f"{name}'s very dense air pulls this mix toward low-frequency emphasis and "
                      f"a muffled, heavy quality - this follows from acoustic density and "
                      f"enclosure here specifically, not a generic 'scary bass' choice.")
    if phys.molten:
        lines.append(f"{name}'s molten material produces the bubbling and popping events you "
                      f"hear; there is no dedicated liquid-water phenomenon in this version of "
                      f"the engine, so water sounds never appear regardless of composition.")
    if phys.icy:
        lines.append(f"{name}'s frozen surface produces the occasional cracking and ticking you "
                      f"hear as it contracts and settles.")
    if phys.energetic:
        lines.append(f"{name}'s high-energy, ionised conditions add the rapid, agitated texture "
                      f"and dissonant intervals you hear, scaled to how extreme its stated "
                      f"temperature is.")
    for note in adjustments:
        lines.append(f"Validation layer correction for {name}: {note}")
    return lines


def _accessible_description(cfg: CosmicSomaticConfig, perceptual: PerceptualState,
                             provenance: Provenance) -> str:
    """Structured text description that complements, not replaces, the audio
    (section 14) - usable with or without a screen reader."""
    conf_line = f"Scientific confidence: {provenance.overall_confidence.title()}"
    if provenance.matched_reference:
        conf_line += f" (cross-checked against reference data for {provenance.matched_reference.title()})."
    else:
        conf_line += " (based on reasoning only; not independently cross-checked by this app)."
    phys_char = cfg.surface_composition_notes.replace("UNVERIFIED:", "").strip()
    d = dataclasses.asdict(perceptual)
    top = sorted(d.items(), key=lambda kv: -kv[1])[:3]
    perc_char = ", ".join(DESCRIPTOR_MAP[name] for name, _ in top)
    return (
        f"Environment: {cfg.object_name}\n\n"
        f"Physical character: {phys_char}\n\n"
        f"Perceptual character: {perc_char}.\n\n"
        f"Experience: {cfg.mental_presence_narrative}\n\n"
        f"{conf_line}"
    )


# ---------------------------------------------------------------------------
# 3. INGESTION - fully open-ended, no local catalogue
# ---------------------------------------------------------------------------
# There is no hardcoded object list. Gemini itself must know or reason about
# whatever was typed; fetch_astronomical_context only trims the raw text.

def fetch_astronomical_context(query: str) -> str:
    """Clean the user's raw query. No lookup, no catalogue, no guessing -
    the physics comes from Gemini's own stated CosmicSomaticConfig fields,
    checked in _validate_physics/_reconcile_with_physics and independently
    cross-checked in _build_provenance."""
    cleaned = " ".join((query or "").split())[:MAX_QUERY_CHARS]
    return cleaned or "an unnamed astronomical object"


def _get_gemini_client():
    if genai is None:
        return None
    api_key = _secret("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=GEMINI_HTTP_TIMEOUT_MS),
        )
    except Exception:
        try:
            return genai.Client(api_key=api_key)
        except Exception:
            return None


_TRAFFIC_CODES = {429, 500, 502, 503, 504}
_TRAFFIC_STATUSES = {"RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL"}


def _describe_gemini_error(exc: Exception) -> dict:
    """Classify a failed Gemini call from what the API actually said instead
    of assuming. google-genai errors carry .code (HTTP status), .status and
    .message; timeouts and dropped connections don't, so they're recognised
    by type."""
    raw_code = getattr(exc, "code", None)
    try:
        code = int(raw_code) if raw_code is not None else None
    except (TypeError, ValueError):
        code = None
    status = getattr(exc, "status", None)
    status = str(status).upper() if status else None
    message = str(getattr(exc, "message", None) or exc or exc.__class__.__name__)
    message = " ".join(message.split())[:400]
    type_name = exc.__class__.__name__.lower()

    is_network = (
        isinstance(exc, (TimeoutError, ConnectionError))
        or any(word in type_name for word in ("timeout", "connect", "network"))
    )
    traffic = (
        (code in _TRAFFIC_CODES) or (status in _TRAFFIC_STATUSES) or is_network
    )
    if code:
        label = f"HTTP {code}" + (f" {status}" if status else "")
    elif is_network:
        label = exc.__class__.__name__
    else:
        label = status or exc.__class__.__name__
    return {
        "traffic": traffic, "code": code, "status": status,
        "label": label, "message": message,
    }


SYSTEM_INSTRUCTION = (
    "You are a factual, zero-hallucination Somatic Surface Simulator. You "
    "must look up the real-world physical boundary data for the object the "
    "user typed - its surface (or, for a body with no solid surface, its "
    "most meaningful physical boundary: the 1-bar level of a gas giant, the "
    "photosphere of a star, the event horizon of a black hole) - and report "
    "surface_gravity_g, surface_temperature_k, atmospheric_pressure_atm and "
    "surface_composition_notes BEFORE choosing anything else. Every "
    "psychoacoustic vector and grounding_tone_hz must be derived strictly "
    "from those numbers, not chosen first and rationalised afterward.\n\n"
    "If you do not genuinely know the object (an obscure catalogue entry, a "
    "fictional or hypothetical body), infer its physics from its stated or "
    "most likely class using real thermodynamics and fluid mechanics, and "
    "begin surface_composition_notes with 'UNVERIFIED:'. Never fabricate "
    "specific numbers you are not reasonably confident in - reasoned "
    "estimates are fine, invented false precision is not. Note that for "
    "well-known solar-system bodies this application independently "
    "cross-checks and may correct your stated values against its own "
    "reference data - you are providing the best available reasoning, not "
    "the final scientific authority.\n\n"
    "If the environment is habitable and calm like Earth, map the vectors to "
    "an open, tranquil space (low hollow_isolation_factor, low "
    "thermal_manic_hum, low harmonic_dissonance_index, low "
    "claustrophobic_suffocation) and let rhythmic_unpredictability come from "
    "gentle, organic micro-transients - breezes, rustling, distant animal "
    "calls, water, wingbeats - never from mechanical events. For a strange "
    "or hypothetical object, infer the surface texture strictly from its "
    "physical class (icy, molten, airless, crushing, plasma) rather than "
    "guessing sci-fi textures. Never invent artificial mechanical buzzes, "
    "sirens, alarms or 'refrigerator hums' - every sound must trace back to "
    "a real physical cause you named in surface_composition_notes.\n\n"
    "Use the whole 0.0-1.0 range decisively; do not cluster around 0.4-0.6. "
    "Two physically different objects must end up with clearly different "
    "numbers. mental_presence_narrative is exactly two vivid sentences, "
    "second person, present tense, built only from the physics you stated. "
    "Never claim a listener's emotional reaction is universal; describe what "
    "the soundscape is designed to evoke, not what everyone will feel."
)


def _parse_gemini_response(response) -> CosmicSomaticConfig:
    """Parse Gemini's JSON into the schema and require the physical fields to
    actually be present and sane - a response with the physics missing or
    broken is treated as unusable so the caller retries."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, CosmicSomaticConfig):
        config = parsed
    else:
        text = getattr(response, "text", None)
        if not text:
            raise ValueError("Gemini returned an empty response.")
        config = CosmicSomaticConfig.model_validate_json(text)
    _validate_physics(config)
    return config


def _call_gemini_model(client, model_name: str, prompt: str, max_attempts: int) -> CosmicSomaticConfig:
    """Calls one specific Gemini model. Traffic-class errors are retried with
    backoff; anything else fails immediately with the real error, because
    retrying a rejected request only wastes time. Raises
    GeminiUnavailableError carrying per-attempt details."""
    details: list = []
    last_kind = "traffic"
    last_short = "was unavailable"
    label = _model_label(model_name)

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=CosmicSomaticConfig,
                    temperature=1.0,
                ),
            )
        except Exception as exc:
            info = _describe_gemini_error(exc)
            details.append(
                f"{label} ({model_name}) attempt {attempt + 1}: {info['label']} - {info['message']}"
            )
            if not info["traffic"]:
                raise GeminiUnavailableError(
                    f"Gemini rejected the request to {label} ({info['label']}): "
                    f"{info['message']}  This is not a traffic problem - check "
                    f"the model name, API key and account access.",
                    reason="rejected", short_reason=f"returned {info['label']}",
                    details=details,
                ) from exc
            last_kind = "traffic"
            if info["code"] == 429:
                last_short = f"was rate-limited or over quota ({info['label']})"
            else:
                last_short = f"was overloaded or unreachable ({info['label']})"
            if info["code"] == 429 and attempt >= 1:
                break  # a quota wall won't clear in a couple of seconds
            if attempt < max_attempts - 1:
                time.sleep(GEMINI_RETRY_BACKOFF_SECONDS * (attempt + 1))
            continue

        try:
            return _parse_gemini_response(response)
        except Exception as exc:
            details.append(
                f"{label} ({model_name}) attempt {attempt + 1}: unreadable response "
                f"({exc.__class__.__name__}) - {' '.join(str(exc).split())[:200]}"
            )
            last_kind = "bad_response"
            last_short = "returned a response the app couldn't read"
            if attempt < max_attempts - 1:
                time.sleep(0.4)

    if last_kind == "bad_response":
        raise GeminiUnavailableError(
            f"{label} answered, but not with usable data. Please try again.",
            reason="bad_response", short_reason=last_short, details=details,
        )
    raise GeminiUnavailableError(
        "Too much traffic right now - please try again in a moment.",
        reason="traffic", short_reason=last_short, details=details,
    )


def generate_psychoacoustic_schema(query: str) -> CosmicSomaticConfig:
    """Asks Gemini to state the real surface physics of the named object and
    propose a psychoacoustic profile strictly from them - fully open-ended,
    no local catalogue. The selected model gets its full retries; if (and
    only if) it failed for traffic reasons, one real attempt is made with
    Gemini 3.5 Flash Lite. Non-traffic failures are raised as-is. Every
    vector is then reconciled against the physics Gemini itself stated, and
    any correction is recorded for the UI. Never returns a fabricated result.
    This function returns Gemini's PROPOSAL; independent scientific
    cross-checking happens afterward in _build_provenance, and perceptual
    grounding happens in _apply_perceptual_grounding - see _run_search."""
    model_name = st.session_state.get("gemini_model", GEMINI_FLASH_MODEL)
    client = _get_gemini_client()

    if client is None:
        raise GeminiUnavailableError(
            "No Gemini API key is configured, so a real soundscape can't be "
            "generated. Add GEMINI_API_KEY to .streamlit/secrets.toml or your "
            "environment (and make sure google-genai is installed).",
            reason="not_configured",
        )

    prompt = (
        f"Object: {query}\n\n"
        "State the real physical boundary conditions of this object's surface "
        "(or nearest meaningful physical boundary) and propose a psychoacoustic "
        "profile for a listener standing there, strictly derived from those "
        "physics."
    )

    try:
        config = _call_gemini_model(client, model_name, prompt, GEMINI_MAX_ATTEMPTS)
    except GeminiUnavailableError as primary_error:
        st.session_state.gemini_diagnostics = list(primary_error.details)
        if primary_error.reason != "traffic" or model_name == GEMINI_LITE_MODEL:
            raise
        try:
            config = _call_gemini_model(client, GEMINI_LITE_MODEL, prompt, GEMINI_FALLBACK_ATTEMPTS)
        except GeminiUnavailableError as fallback_error:
            merged = primary_error.details + fallback_error.details
            st.session_state.gemini_diagnostics = merged
            raise GeminiUnavailableError(
                str(primary_error), reason="traffic",
                short_reason=primary_error.short_reason, details=merged,
            ) from fallback_error
        st.session_state.fallback_notice = (
            f"{_model_label(model_name)} {primary_error.short_reason}, so this "
            f"soundscape was generated with {_model_label(GEMINI_LITE_MODEL)} instead."
        )
        st.session_state.model_used = GEMINI_LITE_MODEL
        config = _sanitize_config(config)
    else:
        st.session_state.model_used = model_name
        config = _sanitize_config(config)

    config, adjustments = _reconcile_with_physics(config)
    st.session_state.physics_adjustments = adjustments
    return config


# ---------------------------------------------------------------------------
# 4. THE DYNAMIC PSYCHOACOUSTIC AUDIO ENGINE (NUMPY/SCIPY DSP)
# ---------------------------------------------------------------------------
# Design: nothing here is chosen from a category label. Every material bed,
# every discrete event and the drone's own character are derived directly
# from SurfacePhysics - the gravity, temperature, pressure and composition
# Gemini stated (and which _reconcile_with_physics already checked for
# plausibility). A habitable world drives the "wind" bed toward open,
# breathing air and the event planner toward birdsong; a crushing atmosphere
# drives the same "wind"/"rumble" beds toward deep, slow, rolling currents;
# an airless world gets almost no airborne bed at all, only the faintest
# structure-borne grit - because that is what the stated physics allows.

TWO_PI = 2.0 * np.pi


def _seeded_rng(name: str) -> np.random.Generator:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16) % (2**32)
    return np.random.default_rng(seed)


def _butter_sos(cutoff, sample_rate: int, btype: str, order: int = 2):
    nyq = sample_rate / 2.0
    if isinstance(cutoff, (list, tuple)):
        wn = [max(1e-6, min(0.999, c / nyq)) for c in cutoff]
    else:
        wn = max(1e-6, min(0.999, cutoff / nyq))
    return butter(order, wn, btype=btype, output="sos")


def _lowpass(x: np.ndarray, sr: int, cutoff: float, passes: int = 1, order: int = 2) -> np.ndarray:
    sos = _butter_sos(cutoff, sr, "low", order)
    for _ in range(passes):
        x = sosfilt(sos, x)
    return x


def _highpass(x: np.ndarray, sr: int, cutoff: float, order: int = 2) -> np.ndarray:
    return sosfilt(_butter_sos(cutoff, sr, "high", order), x)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))) + 1e-12)


def _rms_norm(x: np.ndarray, target: float) -> np.ndarray:
    return x * (target / _rms(x))


def _fold_freq(f: float, lo: float, hi: float) -> float:
    """Shift a frequency by octaves into [lo, hi] (hi must be >= 2*lo)."""
    while f < lo:
        f *= 2.0
    while f > hi:
        f /= 2.0
    return f


def _colored_noise(n: int, rng: np.random.Generator, exponent: float) -> np.ndarray:
    """White noise shaped to 1/f**exponent amplitude (0 white, .5 pink, 1 brown)."""
    spec = np.fft.rfft(rng.normal(0.0, 1.0, n))
    freqs = np.fft.rfftfreq(n, 1.0)
    freqs[0] = freqs[1]
    out = np.fft.irfft(spec / freqs ** exponent, n)
    return out / (np.std(out) + 1e-12)


def _sweep_curve(num_blocks: int, seconds: float, rate: float, phase: float,
                 lo: float, hi: float) -> np.ndarray:
    t = np.linspace(0.0, seconds, num_blocks, endpoint=False)
    return lo + (hi - lo) * (0.5 + 0.5 * np.sin(TWO_PI * rate * t + phase))


def _sweeping_bandpass_centers(signal: np.ndarray, sample_rate: int, centers: np.ndarray,
                               bandwidth: float) -> np.ndarray:
    """A resonant bandpass whose centre frequency follows `centers` (one value
    per block), overlap-added so the sweep is smooth and click-free."""
    n = len(signal)
    num_blocks = len(centers)
    block = max(1, n // num_blocks)
    fade = min(400, block // 4)
    out = np.zeros(n)
    win_sum = np.zeros(n)
    pos, i = 0, 0
    while pos < n:
        end = min(n, pos + block + fade)
        seg = signal[pos:end]
        if len(seg) == 0:
            break
        center = float(centers[min(i, num_blocks - 1)])
        lo = max(20.0, center - bandwidth / 2)
        hi = min(sample_rate / 2 - 50, center + bandwidth / 2)
        if hi <= lo:
            hi = lo + 10.0
        filtered = sosfilt(_butter_sos((lo, hi), sample_rate, "bandpass", 2), seg)
        w = np.ones(len(seg))
        if fade > 0 and len(seg) > 2 * fade:
            ramp = np.linspace(0.0, 1.0, fade)
            w[:fade] = ramp
            w[-fade:] = ramp[::-1]
        out[pos:end] += filtered * w
        win_sum[pos:end] += w
        pos += block
        i += 1
    win_sum[win_sum == 0] = 1.0
    return out / win_sum


def _sweeping_bandpass(signal: np.ndarray, sample_rate: int, low_start: float, low_end: float,
                       bandwidth: float, num_blocks: int = 36) -> np.ndarray:
    """Resonant bandpass whose centre sweeps linearly from low_start to
    low_end Hz over the length of the signal."""
    return _sweeping_bandpass_centers(
        signal, sample_rate, np.linspace(low_start, low_end, num_blocks), bandwidth)


def _recursive_lowpass(signal: np.ndarray, sample_rate: int, cutoff_hz: float, passes: int = 3) -> np.ndarray:
    """A digital filter loop: the same low-pass applied recursively to strip
    brightness away, simulating a heavy, muffled cage."""
    return _lowpass(signal, sample_rate, cutoff_hz, passes=passes)


def _echo(x: np.ndarray, sr: int, delay_s: float, feedback: float, taps: int = 4,
          cutoff: float = 3500.0) -> np.ndarray:
    """Feedback echo where every repeat is darker, like sound crossing a void."""
    out = x.copy()
    d = int(delay_s * sr)
    tap = x
    for k in range(1, taps + 1):
        tap = _lowpass(np.concatenate([np.zeros(d), tap])[: len(x)], sr, cutoff)
        out += (feedback ** k) * tap
    return out


def _autopan(mono, t, depth, rate, phase):
    """Slow equal-power panning LFO (0.05 Hz for the main voice), depth 0..1."""
    pan = np.clip(np.sin(TWO_PI * rate * t + phase) * depth, -1.0, 1.0)
    angle = (pan + 1.0) * (np.pi / 4.0)
    return mono * np.cos(angle), mono * np.sin(angle)


def _structured_ir(sr: int, rng: np.random.Generator, rt60: float, max_len_s: float = 5.0) -> np.ndarray:
    """A room impulse response with structure instead of plain noise:
    6-10 discrete early reflections at 5-80 ms (random gains in [-0.4, 0.4])
    blended into a late, exponentially decaying noise tail that fades in
    smoothly behind them. The reflections are what let the ear judge the
    size and shape of the space; the tail supplies the vastness."""
    length = max(int(min(rt60 * 1.1, max_len_s) * sr), int(0.15 * sr))
    tk = np.arange(length) / sr
    tail = _lowpass(rng.normal(0.0, 1.0, length) * np.exp(-6.91 * tk / rt60), sr, 5000) * 0.12
    ramp = np.clip((tk - 0.005) / 0.075, 0.0, 1.0)
    tail = tail * (ramp * ramp * (3.0 - 2.0 * ramp))  # smoothstep: reflections lead, tail blooms
    ir = tail.copy()

    count = int(rng.integers(6, 11))
    times = np.sort(rng.uniform(0.005, 0.080, count))
    gains = rng.uniform(-0.4, 0.4, count)
    half = max(2, int(0.00025 * sr))
    pulse = np.hanning(2 * half + 1)
    pulse = pulse / pulse.max()
    for tm, g in zip(times, gains):
        c = int(tm * sr)
        lo, hi = max(0, c - half), min(length, c + half + 1)
        ir[lo:hi] += g * pulse[lo - (c - half): hi - (c - half)]
    return ir / (np.sqrt(np.sum(ir ** 2)) + 1e-12)


def _reverb(left, right, sr, rng, hollow):
    """Stereo room whose size follows hollow_isolation_factor, built from
    structured impulse responses (decorrelated per ear). The return is
    high-passed so the tail adds space without adding low-end mud."""
    rt60 = 0.3 + 3.0 * hollow
    wet = 0.06 + 0.40 * hollow
    hp = _butter_sos(180.0, sr, "high", 2)
    out = []
    for ch in (left, right):
        tail = fftconvolve(ch, _structured_ir(sr, rng, rt60))[: len(ch)]
        tail = sosfilt(hp, tail)
        out.append((1.0 - 0.5 * wet) * ch + wet * 1.3 * tail)
    return out[0], out[1]


def _void_layer(n: int, sr: int, rng: np.random.Generator, hollow: float):
    """Layer 1 - The Cold, Isolating Void. Slow wide noise through a sweeping
    resonant bandpass (120 -> 1500 Hz: the mid-range and air that make a
    space readable), then through a structured room impulse response for the
    vast, empty, resonant tails. One independent pair per ear."""
    channels = []
    for _ in range(2):
        swept = _sweeping_bandpass(rng.normal(0.0, 1.0, n), sr, low_start=120.0,
                                   low_end=1500.0, bandwidth=250.0)
        ir = _structured_ir(sr, rng, 0.8 + 3.2 * hollow)
        reverbed = fftconvolve(swept, ir)[:n]
        channels.append(_rms_norm(0.5 * _rms_norm(swept, 1.0) + 0.9 * _rms_norm(reverbed, 1.0), 1.0))
    return channels[0], channels[1]


def _air_layer(n: int, sr: int, rng: np.random.Generator, t: np.ndarray, suffocation: float) -> np.ndarray:
    """High-frequency environmental air/pressure: fresh noise high-passed at
    a cutoff that follows openness, gently swirled by a ~0.2 Hz current."""
    cutoff = max(2000.0, 16000.0 * (1.0 - suffocation))
    sos = _butter_sos(cutoff, sr, "high", 2)
    air = sosfilt(sos, rng.normal(0.0, 1.0, n))
    lfo = 0.65 + 0.35 * np.sin(TWO_PI * 0.2 * t + rng.uniform(0, TWO_PI))
    return _rms_norm(air * lfo, 1.0)


def _peaking_sos(sr: int, f0: float, gain_db: float, q: float) -> np.ndarray:
    """RBJ peaking-EQ biquad as one second-order section."""
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = TWO_PI * f0 / sr
    alpha = math.sin(w0) / (2.0 * q)
    b = np.array([1 + alpha * a_lin, -2 * math.cos(w0), 1 - alpha * a_lin])
    a = np.array([1 + alpha / a_lin, -2 * math.cos(w0), 1 - alpha / a_lin])
    return np.array([[b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]])


def _apply_stereo_drift(signal: np.ndarray, sample_rate: int, drift_amount: float,
                        rate_hz: float = 0.05) -> np.ndarray:
    """Slow equal-power drift (0.05 Hz) PLUS true phase decorrelation.

    Above 400 Hz the right channel is delayed 15-35 ms (scaled by drift) and
    that delayed high band is cross-mixed, phase-inverted, into the left
    channel. The two ears then hear different arrival times and opposite
    polarity in the region where the brain localises, which throws the image
    wide open, while everything below 400 Hz stays identical in both ears so
    the low end remains tight and centred.

    Accepts mono (n,) - panned by the LFO - or stereo (n, 2), where the LFO
    becomes a gentle overall balance drift. Returns (n, 2)."""
    drift = _clamp01(drift_amount)
    sig = np.asarray(signal, dtype=float)
    n = sig.shape[0]
    t = np.arange(n) / sample_rate
    if sig.ndim == 1:
        pan = np.clip(np.sin(TWO_PI * rate_hz * t) * drift, -1.0, 1.0)
        angle = (pan + 1.0) * (np.pi / 4.0)
        left, right = sig * np.cos(angle), sig * np.sin(angle)
    else:
        pan = np.clip(np.sin(TWO_PI * rate_hz * t) * drift * 0.6, -1.0, 1.0)
        angle = (pan + 1.0) * (np.pi / 4.0)
        left = sig[:, 0] * np.cos(angle) * math.sqrt(2.0)
        right = sig[:, 1] * np.sin(angle) * math.sqrt(2.0)

    hp = _butter_sos(400.0, sample_rate, "high", 2)
    hf_l, hf_r = sosfilt(hp, left), sosfilt(hp, right)
    lo_l, lo_r = left - hf_l, right - hf_r

    delay = int((15.0 + 20.0 * drift) / 1000.0 * sample_rate)
    hf_r_delayed = np.concatenate([np.zeros(delay), hf_r])[:n]
    cross = 0.25 + 0.45 * drift
    out_l = (lo_l + hf_l - cross * hf_r_delayed) / math.sqrt(1.0 + cross ** 2)
    out_r = lo_r + hf_r_delayed
    return np.stack([out_l, out_r], axis=-1)


def _fade_envelope(n: int, sample_rate: int, attack_s: float = 1.2, release_s: float = 2.0) -> np.ndarray:
    env = np.ones(n)
    a = min(n // 2, int(attack_s * sample_rate))
    r = min(n // 2, int(release_s * sample_rate))
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a)
    if r > 0:
        env[-r:] = np.linspace(1.0, 0.0, r)
    return env


# ---- material beds: derived straight from the stated surface physics ------

def _bed_wind(n, sr, rng, phys: SurfacePhysics):
    """Moving air. Thin, cold air keens high and fast (a Martian whistle);
    warm, near-Earth air breathes at a natural, gentle rate; dense, heavy air
    rolls in deep, slow currents - the depth and speed both come straight
    from atmospheric_pressure_atm and surface_temperature_k, not a label."""
    dur = n / sr
    depth = _clamp01(math.log10(1.0 + phys.air) / 1.3)          # 0 thin, 1 dense
    warmth = _clamp01((phys.temperature - 180.0) / 160.0)        # 0 frigid, 1 temperate+
    rate = (0.028 + 0.10 * (1.0 - depth)) * (0.8 + 0.4 * warmth)
    lo_center = 130.0 - 55.0 * depth
    hi_center = 2500.0 - 1300.0 * depth
    nb = max(8, int(dur * 2.5))
    pink = _colored_noise(n, rng, 0.45 + 0.35 * depth)
    a = _sweeping_bandpass_centers(
        pink, sr, _sweep_curve(nb, dur, rate, rng.uniform(0, TWO_PI), lo_center, lo_center + 650.0),
        220.0 + 160.0 * depth)
    b = _sweeping_bandpass_centers(
        pink, sr, _sweep_curve(nb, dur, rate * 1.7, rng.uniform(0, TWO_PI), hi_center * 0.4, hi_center),
        380.0 + 220.0 * depth)
    low = _lowpass(pink, sr, 220.0 + 260.0 * depth)
    return _rms_norm(
        _rms_norm(a, 1.0) * (0.55 + 0.45 * depth)
        + _rms_norm(b, 1.0) * (0.75 - 0.30 * depth)
        + _rms_norm(low, 1.0) * (0.20 + 0.55 * depth), 1.0)


def _bed_plasma(n, sr, rng, phys: SurfacePhysics):
    """Hot, ionised, roaring matter. Crackle rate and brightness both climb
    with the stated temperature - a lava lake and a stellar photosphere use
    the same physics, scaled by how hot each one actually is."""
    heat = _clamp01((math.log10(max(phys.temperature, 10.0)) - 2.8) / 3.4)
    hiss = _highpass(rng.normal(0.0, 1.0, n), sr, 1300.0 + 3200.0 * heat)
    flutter = _lowpass(rng.normal(0.0, 1.0, n), sr, 18.0 + 42.0 * heat)
    flutter = np.clip(flutter / (np.std(flutter) + 1e-12), -1.5, 1.5) / 1.5
    sizzle = hiss * np.clip(1.0 + 0.7 * flutter, 0.1, None)
    rate = 25.0 + 180.0 * heat
    impulses = (rng.random(n) < rate / sr) * rng.choice([-1.0, 1.0], n) * rng.uniform(0.3, 1.0, n)
    tk = np.arange(int(0.008 * sr)) / sr
    crackle = fftconvolve(impulses, np.exp(-tk / (0.0028 - 0.0016 * heat)))[:n]
    body = _lowpass(_colored_noise(n, rng, 1.0), sr, 380.0 + 340.0 * heat)
    return _rms_norm(_rms_norm(sizzle, 1.0) * 0.5 + _rms_norm(crackle, 1.0) * (0.55 + 0.4 * heat)
                     + _rms_norm(body, 1.0) * 0.35, 1.0)


def _bed_rumble(n, sr, rng, phys: SurfacePhysics):
    """Crushing atmosphere. A contained bass body (never pure sub-boom), a
    gritty mid growl and fine grain on top, all scaled by how many times
    denser than Earth's the stated air actually is."""
    t = np.arange(n) / sr
    depth = _clamp01((phys.air - 8.0) / 40.0)
    cutoff = 300.0 - 110.0 * depth
    body = _highpass(_lowpass(_colored_noise(n, rng, 1.0), sr, cutoff, passes=2), sr, 45)
    heave = 1.0 + (0.22 + 0.18 * depth) * np.sin(TWO_PI * (0.09 + 0.06 * (1.0 - depth)) * t + rng.uniform(0, TWO_PI))
    pink = _colored_noise(n, rng, 0.5)
    growl = sosfilt(_butter_sos((190.0, 900.0), sr, "bandpass", 2), pink) * (
        0.55 + 0.45 * np.sin(TWO_PI * 0.21 * t + rng.uniform(0, TWO_PI)))
    flicker = np.clip(_lowpass(rng.normal(0.0, 1.0, n), sr, 8) * 3.0, 0.0, None)
    grain = _highpass(pink, sr, 1500) * flicker * (0.65 - 0.30 * depth)
    return _rms_norm(0.55 * _rms_norm(body * heave, 1.0) + 0.9 * _rms_norm(growl, 1.0)
                     + 0.35 * _rms_norm(grain, 1.0), 1.0)


def _bed_crystal(n, sr, rng, phys: SurfacePhysics):
    """Ice and frozen silence. Sparse high sparkle plus thin air noise if any
    atmosphere at all is present; colder stated temperatures narrow the
    sparkle band and slow it down."""
    t = np.arange(n) / sr
    cold = _clamp01((230.0 - phys.temperature) / 180.0)
    out = np.zeros(n)
    for f in rng.uniform(1300.0 + 900.0 * cold, 5000.0 - 700.0 * cold, 7):
        sparkle = np.clip(np.sin(TWO_PI * rng.uniform(0.03, 0.30) * (1.0 - 0.4 * cold) * t
                                  + rng.uniform(0, TWO_PI)), 0.0, None) ** 2
        out += sparkle * np.sin(TWO_PI * f * t + rng.uniform(0, TWO_PI))
    air = _highpass(_colored_noise(n, rng, 0.5), sr, 3200.0) * (0.5 if phys.airborne else 0.15)
    return _rms_norm(_rms_norm(out, 1.0) + 0.3 * _rms_norm(air, 1.0), 1.0)


def _bed_stone(n, sr, rng, phys: SurfacePhysics):
    """No atmosphere: no air to carry sound. What little a listener could
    'feel' is structure-borne - faint grit and thermal-stress settling
    conducted through solid ground - kept deliberately near silent rather
    than inventing an airborne texture that could not physically exist."""
    grit = _highpass(_colored_noise(n, rng, 0.3), sr, 2600.0) * 0.5
    settle = _lowpass(_colored_noise(n, rng, 1.0), sr, 90.0) * 0.2
    return _rms_norm(grit + settle, 1.0)


_MATERIAL_BED_FN = {
    "wind": _bed_wind, "plasma": _bed_plasma, "rumble": _bed_rumble,
    "crystal": _bed_crystal, "stone": _bed_stone,
}
# Stone represents near-silence (no atmosphere to carry sound at all), so it
# gets far less gain in the final mix than an actual audible medium.
_MATERIAL_BED_GAIN = {"wind": 0.15, "plasma": 0.15, "rumble": 0.15, "crystal": 0.15, "stone": 0.045}
# Each material gets a stable timbral colour of its own on top of its source
# sound: (centre Hz, gain dB, Q) peaking filters.
_MATERIAL_EQ = {
    "wind": [],
    "plasma": [(3500.0, 5.0, 0.6)],
    "rumble": [(3500.0, -5.0, 0.6), (900.0, 2.0, 0.8)],
    "crystal": [(300.0, -5.0, 0.7), (5000.0, 4.0, 0.7)],
    "stone": [(250.0, -3.0, 0.7), (4000.0, 2.0, 1.0)],
}


def _material_bed(material: str, n: int, sr: int, rng: np.random.Generator, phys: SurfacePhysics):
    return _MATERIAL_BED_FN.get(material, _bed_wind)(n, sr, rng, phys)


# ---- tonal identity --------------------------------------------------------

def _organic_lfo(n: int, sr: int, rng: np.random.Generator, rate_hz: float = 0.05) -> np.ndarray:
    """Smooth, irregularly wandering modulation (band-limited noise), roughly
    bounded to [-1, 1], instead of a fixed-rate sine. Real ambiences drift
    unevenly, not on a metronome - a perfectly periodic vibrato or breathing
    cycle is exactly what reads as electronic/mechanical (a fridge compressor
    cycling, a synth LFO) rather than alive. Used everywhere a continuous
    tone needs to move without ever settling into a repeating cycle."""
    smoothed = _lowpass(rng.normal(0.0, 1.0, n), sr, max(rate_hz, 0.01), passes=2, order=2)
    return smoothed / (np.max(np.abs(smoothed)) + 1e-9)


def _root_drone(t, sr, rng, root, thermal, dissonance, tranquil: bool):
    """The place's tonal centre - deliberately NOT a clean tone generator.
    A pure, static, harmonically-simple sustained pitch is exactly what a
    refrigerator or AC compressor hum sounds like; every choice below exists
    to keep this from ever collapsing into that: fewer and softer harmonics,
    a breath of filtered noise blended into the tone itself, and irregular
    (band-limited-noise) drift in both pitch and level instead of a fixed
    sine cycle anywhere.

    For an earth-like or otherwise tranquil world, `spread` is hardcoded to
    0.0: the voices lock to one warm pitch instead of detuning against each
    other. Everywhere else spread still grows gently with heat.

    For very low roots the fundamental is turned down and the interval
    partials are lifted by octaves: the ear still hears the low pitch from
    the harmonics (the "missing fundamental" effect), but the mix stops
    piling energy into the sub-bass where it turns to mud."""
    n = len(t)
    # Two scales of irregular pitch movement: a slow multi-second drift, plus
    # a subtle natural-vibrato-rate flutter (~5 Hz, itself irregular rather
    # than a locked sine). The fast component is what actually spreads a
    # sustained tone's energy across neighbouring frequencies from moment to
    # moment - without it a "slowly drifting" tone can still look and sound
    # like a pure, static test-tone within any short listening window.
    wander_slow = _organic_lfo(n, sr, rng, rate_hz=0.05)
    wander_fast = _organic_lfo(n, sr, rng, rate_hz=5.0)
    vibrato = 1.0 + 0.0025 * wander_slow + 0.007 * wander_fast
    phase = TWO_PI * np.cumsum(root * vibrato) / sr

    n_max = int(np.clip(1400.0 / root, 3, 10))
    n_harm = int(2 + (n_max - 2) * (0.25 + 0.75 * thermal))
    slope = 1.8 - 0.5 * thermal
    spread = 0.0 if tranquil else (0.003 + 0.015 * thermal)
    fund_weight = float(np.clip(root / 160.0, 0.2, 0.75))
    sig = np.zeros_like(t)
    for ratio in (1.0, 1.0 + spread, 1.0 - 0.8 * spread):
        ph = phase * ratio + rng.uniform(0, TWO_PI)
        for k in range(1, n_harm + 1):
            if root * k * ratio > 0.4 * sr:
                break
            weight = fund_weight if k == 1 else (0.55 + 0.45 * fund_weight if k == 2 else 0.6)
            sig += weight * np.sin(k * ph) / k ** slope
    sig = _rms_norm(sig, 1.0)

    # A soft breath of filtered noise around the fundamental - the single
    # biggest thing separating "wind resonating in a pipe" from "an
    # electrical hum" is that the natural version is never perfectly pure.
    breath = sosfilt(_butter_sos((max(root * 0.6, 30.0), min(root * 2.6, sr / 2 - 100)),
                                  sr, "bandpass", 2), rng.normal(0.0, 1.0, n))
    breath_env = 0.55 + 0.45 * _organic_lfo(n, sr, rng, rate_hz=0.09)
    breath_mix = 0.42 - 0.14 * thermal  # hotter places lean toward tone over breath
    sig = sig * (1.0 - breath_mix) + _rms_norm(breath * breath_env, 1.0) * breath_mix

    lift = 1.0 if root >= 100.0 else 2.0 ** math.ceil(math.log2(100.0 / root))
    lp = phase * lift
    consonant = np.sin(1.5 * lp + 0.7) + 0.6 * np.sin(2.0 * lp + 1.9)
    clash = np.sin(math.sqrt(2.0) * lp + 0.3) + 0.8 * np.sin((16.0 / 15.0) * lp + 2.2)
    sig = sig + (1.0 - dissonance) * 0.28 * _rms_norm(consonant, 1.0) \
        + (dissonance ** 1.2) * 0.7 * _rms_norm(clash, 1.0)

    # Slow, irregular swell in level - never a fixed breathing rate. Deliberately
    # wide (0.55-1.0) so the tone visibly rises and falls rather than sitting at
    # one constant level the whole time, which is what reads as a static,
    # appliance-like hum regardless of how much organic drift is underneath it.
    swell = 0.55 + 0.45 * (0.5 + 0.5 * _organic_lfo(n, sr, rng, rate_hz=0.02))
    return _rms_norm(sig * swell, 1.0)


def _thermal_cluster(t, sr, rng, root, thermal, dissonance):
    """Layer 2 - the manic star core: three mid-range oscillators, octave-
    folded from the place's root so it stays in tune with it, detuned
    further by the dissonance index, given richer harmonics as things get
    hotter. The agitation tremor is irregular (band-passed noise around the
    target rate), not a metronomic sine, because real turbulence flickers
    unevenly. Gain-staged by thermal_manic_hum downstream, which
    _reconcile_with_physics already pinned near 0 for cool surfaces - so this
    simply does not sound on a temperate world."""
    mid = _fold_freq(root, 170.0, 340.0)
    freqs = (mid, mid * (1.0 + 0.007 + 0.05 * dissonance), mid * (1.0 + 0.014 + 0.09 * dissonance))
    richness = int(2 + 8 * thermal)
    sig = np.zeros_like(t)
    for f in freqs:
        ph = TWO_PI * f * t
        for k in range(1, richness + 1):
            sig += np.sin(k * ph + 0.9 * k) / k ** (1.4 - 0.6 * thermal)
    n = len(t)
    center = 3.0 + 5.0 * thermal
    flicker = sosfilt(_butter_sos((max(center - 1.5, 0.3), center + 1.5), sr, "bandpass", 2),
                       rng.normal(0.0, 1.0, n))
    flicker = flicker / (np.max(np.abs(flicker)) + 1e-9)
    tremor = 1.0 + 0.3 * thermal * flicker
    return _rms_norm(sig * tremor, 1.0)


def _sub_drone(t, sr, rng, root):
    """A felt, not heard, sense of scale - softened with a touch of filtered
    noise and irregular swell so it reads as ground presence rather than an
    electrical sub-hum, and kept deliberately quiet in the final mix."""
    n = len(t)
    sub_f = _fold_freq(root, 30.0, 62.0)
    swell = 0.6 + 0.4 * (0.5 + 0.5 * _organic_lfo(n, sr, rng, rate_hz=0.018))
    tone = np.sin(TWO_PI * sub_f * t) + 0.25 * np.sin(TWO_PI * 2 * sub_f * t + 1.1)
    breath = _lowpass(rng.normal(0.0, 1.0, n), sr, sub_f * 1.6)
    return _rms_norm(tone, 1.0) * 0.75 * swell + _rms_norm(breath, 1.0) * 0.25 * swell


def _shear_layer(n, sr, rng, t, friction):
    """Turbulent fluid shear: swept narrow noise, amplitude-modulated at a
    grind rate that climbs with friction. Generic to any moving fluid -
    tearing winds, storm shear, plasma streaming past a field line - rather
    than tied to one material."""
    dur = n / sr
    nb = max(8, int(dur * 3))
    centers = 700.0 + 2600.0 * rng.random(nb)
    centers = np.convolve(centers, np.ones(3) / 3.0, mode="same")
    band = _sweeping_bandpass_centers(rng.normal(0.0, 1.0, n), sr, centers, 500.0)
    grind = np.clip(0.5 + 0.5 * np.sin(TWO_PI * (6.0 + 8.0 * friction) * t), 0.05, 1.0)
    return _rms_norm(band * grind, 1.0)


def _phase_tearing(signal: np.ndarray, sample_rate: int, rate_hz: float,
                   depth_ms: float, mix: float, phase: float = 0.0) -> np.ndarray:
    """Rapid, sweeping phase modulation via a modulated fractional delay line,
    mimicking magnetic fields tearing against vacuum."""
    n = len(signal)
    t = np.arange(n) / sample_rate
    depth_samples = (depth_ms / 1000.0) * sample_rate
    lfo = (np.sin(TWO_PI * rate_hz * t + phase) + 1.0) / 2.0
    src_idx = np.clip(np.arange(n) - lfo * depth_samples, 0, n - 1)
    delayed = np.interp(src_idx, np.arange(n), signal)
    return (1.0 - mix) * signal + mix * delayed


def _distant_calls(n, sr, rng, root, hollow, dissonance):
    """Sparse, swelling tones that call across emptiness and come back as
    long echoes - the audible face of hollow_isolation_factor, independent
    of surface material."""
    left, right = np.zeros(n), np.zeros(n)
    if hollow < 0.15:
        return left, right
    count = int(round((n / sr) / 5.0 * (hollow - 0.1) * 1.6))
    ratios = (2.0, 3.0, 4.0, 5.0, 6.0) if dissonance < 0.5 else (2.0, 2.0 * math.sqrt(2.0), 3.0 * 16.0 / 15.0, 3.0, 7.0)
    for _ in range(count):
        f = _fold_freq(root * rng.choice(ratios), 180.0, 1400.0)
        tk = np.arange(int(3.5 * sr)) / sr
        env = (1.0 - np.exp(-tk / 0.7)) * np.exp(-tk / 1.6)
        env = env / env.max()
        vib = 1.0 + 0.004 * np.sin(TWO_PI * 4.5 * tk)
        tone = (np.sin(TWO_PI * f * tk * vib) + 0.45 * np.sin(TWO_PI * 2.0 * f * tk + 1.3)) * env
        pos = int(rng.uniform(0.02, 0.85) * n)
        end = min(n, pos + len(tone))
        pan = rng.uniform(-0.8, 0.8)
        angle = (pan + 1.0) * (np.pi / 4.0)
        left[pos:end] += tone[: end - pos] * np.cos(angle)
        right[pos:end] += tone[: end - pos] * np.sin(angle)
    delay = 0.28 + 0.5 * hollow
    return (_echo(left, sr, delay, 0.25 + 0.5 * hollow),
            _echo(right, sr, delay * 1.12, 0.25 + 0.5 * hollow))


# ---- discrete events: derived from physics, never a fixed material menu ---
# _event_plan (in the physics section above) decides WHICH natural sources
# can exist here from gravity/temperature/pressure/composition; the
# generators below are all light, organic, physically-motivated transients -
# never a mechanical buzz, siren or hum.

def _event_sound(kind: str, sr: int, rng: np.random.Generator, phys: SurfacePhysics) -> np.ndarray:
    if kind == "bird":
        length = int(rng.uniform(0.14, 0.30) * sr)
        tk = np.arange(length) / sr
        dur = tk[-1] if length > 1 else 1.0
        f0, f1 = rng.uniform(1800.0, 3200.0), rng.uniform(1800.0, 3200.0) * rng.uniform(1.3, 2.3)
        sweep = f0 + (f1 - f0) * (tk / dur) ** rng.uniform(0.5, 1.8)
        phase = TWO_PI * np.cumsum(sweep) / sr
        env = np.sin(np.pi * np.clip(tk / dur, 0.0, 1.0)) ** 0.7
        tone = (np.sin(phase) + 0.35 * np.sin(2.02 * phase)) * env
        if rng.random() < 0.4:  # an occasional two-note call
            gap = np.zeros(int(0.03 * sr))
            tone = np.concatenate([tone, gap, tone[: length // 2]])
        return tone / (np.max(np.abs(tone)) + 1e-9)

    if kind == "rustle":
        length = int(rng.uniform(0.25, 0.55) * sr)
        band = sosfilt(_butter_sos((900.0, 3800.0), sr, "bandpass", 2), rng.normal(0.0, 1.0, length))
        env = np.zeros(length)
        for _ in range(int(rng.integers(3, 7))):
            c, w = int(rng.uniform(0.1, 0.9) * length), int(rng.uniform(0.02, 0.06) * sr)
            lo, hi = max(0, c - w), min(length, c + w)
            env[lo:hi] += np.hanning(max(hi - lo, 1))[: hi - lo]
        sig = band * env
        return sig / (np.max(np.abs(sig)) + 1e-9)

    if kind in ("whistler", "chorus"):
        length = int(rng.uniform(0.8, 1.8) * sr)
        tk = np.arange(length) / sr
        dur = tk[-1] if length > 1 else 1.0
        sweep = rng.uniform(2500.0, 5000.0) + (rng.uniform(300.0, 900.0) - rng.uniform(2500.0, 5000.0)) * (tk / dur)
        phase = TWO_PI * np.cumsum(np.abs(sweep)) / sr
        env = np.sin(np.pi * np.clip(tk / dur, 0.0, 1.0)) ** 1.3
        return np.sin(phase) * env

    if kind == "bubble":
        length = int(rng.uniform(0.15, 0.4) * sr)
        tk = np.arange(length) / sr
        chirp = rng.uniform(90.0, 260.0) * (1.0 + 1.5 * np.exp(-tk * 40.0))
        phase = TWO_PI * np.cumsum(chirp) / sr
        return np.sin(phase) * np.exp(-tk * rng.uniform(10.0, 22.0))

    if kind == "pop":
        length = int(0.05 * sr)
        tk = np.arange(length) / sr
        band = sosfilt(_butter_sos((600.0, 3200.0), sr, "bandpass", 2), rng.normal(0, 1, length))
        return band * np.exp(-tk * 90.0)

    if kind == "crack":
        length = int(rng.uniform(0.08, 0.22) * sr)
        tk = np.arange(length) / sr
        band = _highpass(rng.normal(0.0, 1.0, length), sr, 1800.0)
        sig = band * np.exp(-tk * rng.uniform(18.0, 40.0))
        thud_tk = np.arange(int(0.12 * sr)) / sr
        thud = np.sin(TWO_PI * rng.uniform(60.0, 110.0) * thud_tk) * np.exp(-thud_tk * 14.0) * 0.5
        k = min(len(thud), len(sig))
        sig[:k] += thud[:k]
        return sig / (np.max(np.abs(sig)) + 1e-9)

    if kind == "tick":
        length = int(0.02 * sr)
        tk = np.arange(length) / sr
        return _highpass(rng.normal(0.0, 1.0, length), sr, 3000.0) * np.exp(-tk * 220.0) * 0.6

    if kind == "creak":
        length = int(rng.uniform(0.3, 0.7) * sr)
        tk = np.arange(length) / sr
        dur = tk[-1] if length > 1 else 1.0
        wobble = 1.0 + 0.015 * np.sin(TWO_PI * rng.uniform(3.0, 7.0) * tk)
        env = np.sin(np.pi * np.clip(tk / dur, 0.0, 1.0)) ** 2
        return np.sin(TWO_PI * rng.uniform(70.0, 160.0) * tk * wobble) * env * 0.4

    if kind == "gust":
        length = int(rng.uniform(0.8, 1.8) * sr)
        tk = np.arange(length) / sr
        dur = tk[-1] if length > 1 else 1.0
        band = sosfilt(_butter_sos((250.0, 2200.0), sr, "bandpass", 2), rng.normal(0.0, 1.0, length))
        return band * np.sin(np.pi * np.clip(tk / dur, 0.0, 1.0)) ** 1.5

    if kind == "swell":
        length = int(rng.uniform(1.5, 3.0) * sr)
        tk = np.arange(length) / sr
        dur = tk[-1] if length > 1 else 1.0
        band = _lowpass(rng.normal(0.0, 1.0, length), sr, 500.0)
        return band * np.sin(np.pi * np.clip(tk / dur, 0.0, 1.0)) ** 1.2

    length = int(0.1 * sr)
    return rng.normal(0.0, 0.1, length)


def _events(n: int, sr: int, rng: np.random.Generator, phys: SurfacePhysics, unpredictability: float):
    """Discrete transients whose very existence and character come from the
    physics plan (_event_plan), not from a hardcoded per-material switch."""
    left, right = np.zeros(n), np.zeros(n)
    if unpredictability <= 0.02:
        return left, right
    plan = _event_plan(phys)
    kinds = [k for k, _ in plan]
    weights = np.array([r for _, r in plan], dtype=float)
    weights = weights / weights.sum()
    total_rate = sum(r for _, r in plan)
    count = int(round((n / sr) * total_rate * (0.35 + 1.3 * unpredictability)))
    for _ in range(count):
        kind = kinds[rng.choice(len(kinds), p=weights)]
        pos = int(rng.uniform(0.02, 0.98) * n)
        amp = rng.uniform(0.5, 1.0) * (0.5 + 0.5 * unpredictability)
        pan = rng.uniform(-1.0, 1.0)
        seg = _event_sound(kind, sr, rng, phys)
        seg = seg / (np.max(np.abs(seg)) + 1e-9)
        end = min(n, pos + len(seg))
        seg = seg[: end - pos]
        angle = (pan + 1.0) * (np.pi / 4.0)
        left[pos:end] += seg * amp * np.cos(angle)
        right[pos:end] += seg * amp * np.sin(angle)
    return left, right


def _synthesize_raw_stereo(config: CosmicSomaticConfig, duration: float,
                            sample_rate: int = 44100):
    """The full DSP synthesis chain, unchanged from the original engine,
    stopping just short of int16 encoding so the mastering/validation layer
    (section 4b) can be inserted before the signal ever becomes a file.
    Returns (stereo float64 array shape (n, 2), sample_rate)."""
    cfg = _sanitize_config(config)
    v = cfg.vectors
    hollow, thermal = v.hollow_isolation_factor, v.thermal_manic_hum
    dissonance, unpredictability = v.harmonic_dissonance_index, v.rhythmic_unpredictability
    suffocation, drift, friction = v.claustrophobic_suffocation, v.infinite_spatial_drift, v.tearing_friction_index
    root = cfg.grounding_tone_hz

    phys = _surface_physics(cfg)
    material = _choose_material(phys)
    tranquil = _is_tranquil(phys, v)

    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n, endpoint=False)
    rng = _seeded_rng(cfg.object_name)
    sr = sample_rate

    # Slow "weather": the whole place breathes over time so it never sits
    # still - irregular (band-limited-noise) drift, not a fixed sine cycle,
    # so nothing in the mix ever settles into a mechanically repeating rate.
    breathe_rate = 0.025 if tranquil else 0.05
    evolve = 0.88 + 0.12 * (0.5 + 0.5 * _organic_lfo(n, sr, rng, rate_hz=breathe_rate))

    # Material bed, generated independently per ear for natural width, driven
    # entirely by the stated surface physics (gravity/temperature/pressure).
    bed_l = _material_bed(material, n, sr, rng, phys)
    bed_r = _material_bed(material, n, sr, rng, phys)
    width = 0.3 + 0.7 * drift
    mid, side = (bed_l + bed_r) / 2.0, (bed_l - bed_r) / 2.0 * width
    bed_l, bed_r = mid + side, mid - side

    # Tonal identity + layer 2 (manic star core), with 0.05 Hz drifting pans.
    # spread is hardcoded to 0.0 inside _root_drone whenever tranquil is True.
    drone = _root_drone(t, sr, rng, root, thermal, dissonance, tranquil)
    cluster = _thermal_cluster(t, sr, rng, root, thermal, dissonance) * (
        1.0 + 0.25 * (0.5 + 0.5 * _organic_lfo(n, sr, rng, rate_hz=0.09)))
    dl, dr = _autopan(drone, t, 0.15 + 0.85 * drift, 0.05, rng.uniform(0, TWO_PI))
    cl, cr = _autopan(cluster, t, 0.15 + 0.85 * drift, 0.031, rng.uniform(0, TWO_PI))

    g_bed = _MATERIAL_BED_GAIN.get(material, 0.15)
    # The grounding tone is a resonance carried through a medium: with no
    # atmosphere there is nothing for a continuous airborne tone to resonate
    # in, so it is reduced to a faint trace rather than the same audible hum
    # every airless world would otherwise share. This is what keeps the
    # tonal "grounding" layer feeling important where it belongs (a breathing
    # world) without it becoming a generic background hum everywhere else.
    air_gate = 1.0 if phys.airborne else 0.12
    g_drone = (0.08 if tranquil else 0.13) * air_gate
    g_cluster = 0.20 * thermal ** 1.1
    body_l = (bed_l * g_bed + dl * g_drone + cl * g_cluster) * evolve
    body_r = (bed_r * g_bed + dr * g_drone + cr * g_cluster) * evolve

    # Layer 1 - the cold, isolating void (swept 120-1500 Hz + structured reverb).
    if hollow > 0.02:
        void_l, void_r = _void_layer(n, sr, rng, hollow)
        body_l, body_r = body_l + void_l * 0.15 * hollow ** 0.9, body_r + void_r * 0.15 * hollow ** 0.9

    # Friction: turbulent shear texture (amplitude follows friction).
    if friction > 0.02:
        shear = _shear_layer(n, sr, rng, t, friction) * 0.13 * friction
        sl, sr_ = _autopan(shear, t, 0.5 + 0.5 * drift, 0.07, rng.uniform(0, TWO_PI))
        body_l, body_r = body_l + sl, body_r + sr_

    # Unpredictability: organic events this specific place's physics allows -
    # birdsong and rustle on a living world, cracking on ice, bubbling and
    # popping in lava, ticking/creaking on bare airless rock. Never a
    # mechanical buzz: every kind traces to a real physical cause.
    ev_l, ev_r = _events(n, sr, rng, phys, unpredictability)
    if hollow > 0.15 and unpredictability > 0.02:
        ev_l = _echo(ev_l, sr, 0.28 + 0.5 * hollow, 0.2 + 0.5 * hollow)
        ev_r = _echo(ev_r, sr, (0.28 + 0.5 * hollow) * 1.12, 0.2 + 0.5 * hollow)
    body_l, body_r = body_l + ev_l * 0.9, body_r + ev_r * 0.9

    # Distant calls that return as long echoes (more isolation, more distance).
    call_l, call_r = _distant_calls(n, sr, rng, root, hollow, dissonance)
    body_l, body_r = body_l + call_l * 0.30 * hollow, body_r + call_r * 0.30 * hollow

    # Heat brightens the place, distance dulls it - but calm places stay clear.
    tilt_cut = float(np.clip((4500.0 + 9000.0 * thermal) * (1.0 - 0.2 * hollow), 4500.0, 14000.0))
    body_l = _lowpass(body_l, sr, tilt_cut, order=1)
    body_r = _lowpass(body_r, sr, tilt_cut, order=1)

    # Layer 3 - the claustrophobic choke: recursive low-pass swallows treble.
    if suffocation > 0.02:
        cutoff = max(600.0, 9000.0 - 6000.0 * suffocation)
        wet_l = _recursive_lowpass(body_l, sr, cutoff, passes=2)
        wet_r = _recursive_lowpass(body_r, sr, cutoff, passes=2)
        thick = _rms_norm(_lowpass(_colored_noise(n, rng, 1.0), sr, 400), 0.03 * suffocation)
        body_l = (1.0 - suffocation) * body_l + suffocation * wet_l + thick
        body_r = (1.0 - suffocation) * body_r + suffocation * wet_r + thick

    # High-frequency air/pressure texture, present while the place is open
    # enough to let it through; added after the choke so its own cutoff holds.
    if suffocation < 0.8:
        openness = 1.0 - suffocation
        air_gain = (0.03 + 0.05 * openness) * (1.0 if phys.airborne else 0.15)
        body_l = body_l + _air_layer(n, sr, rng, t, suffocation) * air_gain
        body_r = body_r + _air_layer(n, sr, rng, t, suffocation) * air_gain

    # Sub-bass presence kept deliberately small - a felt sense of scale, not
    # a boom, and smaller still on an airless world with nothing to carry it.
    sub_scale = 1.0 if phys.airborne else 0.12
    sub = _sub_drone(t, sr, rng, root) * (0.008 + 0.04 * hollow) * sub_scale
    left = body_l + sub
    right = body_r + sub

    # Layer 4 - phase-modulation sweeps for turbulent shear / magnetic fields.
    if friction > 0.02:
        depth_ms, rate = 2.0 + 12.0 * friction, 4.0 + 6.0 * friction
        mixamt = min(0.85, friction)
        left = _phase_tearing(left, sr, rate, depth_ms, mixamt, 0.0)
        right = _phase_tearing(right, sr, rate * 1.13, depth_ms, mixamt, 1.7)

    # Space: structured-reflection reverb sized by hollow_isolation_factor.
    left, right = _reverb(left, right, sr, rng, hollow)

    # Drift + phase decorrelation: wide image, tight low end.
    stereo = _apply_stereo_drift(np.stack([left, right], axis=-1), sr, drift, rate_hz=0.05)

    stereo = stereo * _fade_envelope(n, sr)[:, None]

    # Clarity EQ: remove sub-rumble and a touch of low-mid mud, then apply
    # this specific material's own stable timbral colour.
    stereo = sosfilt(_butter_sos(40.0, sr, "high", 2), stereo, axis=0)
    stereo = sosfilt(_peaking_sos(sr, 240.0, -3.5, 0.8), stereo, axis=0)
    for f0, gain_db, q in _MATERIAL_EQ.get(material, []):
        stereo = sosfilt(_peaking_sos(sr, f0, gain_db, q), stereo, axis=0)

    # Master: soft transient compression stops flares from shrinking the
    # body, then peak normalisation and tanh saturation. Loudness and drive
    # follow overall intensity, so a tranquil world stays clean and quiet
    # and a violent one pushes hard.
    intensity = _clamp01((thermal + dissonance + unpredictability + friction) / 4.0)
    knee = 4.0 * _rms(stereo)
    stereo = np.tanh(stereo / knee) * knee
    target_peak = 0.40 + 0.50 * intensity
    peak = np.max(np.abs(stereo)) + 1e-9
    stereo = stereo / peak * target_peak
    stereo = np.tanh(stereo * (1.2 + 0.6 * intensity))
    stereo = np.clip(stereo, -1.0, 1.0)

    return stereo, sample_rate


# ---------------------------------------------------------------------------
# 4b. AUDIO MASTERING & VALIDATION
#    Runs after synthesis, before anything is written to disk or played:
#    DC removal, NaN/Inf sanitisation, clipping detection, a safety limiter,
#    an approximate true-peak estimate (4x oversampled), an approximate
#    loudness measurement, and a final bounds check. Figures here are
#    explicitly labelled "approximate" rather than claimed as a certified
#    broadcast-loudness measurement, consistent with this project's own
#    rule against overclaiming precision it cannot actually verify.
# ---------------------------------------------------------------------------

def _master_and_validate(stereo: np.ndarray, sr: int):
    issues = []

    if not np.all(np.isfinite(stereo)):
        issues.append("non-finite samples detected and zeroed")
        stereo = np.nan_to_num(stereo, nan=0.0, posinf=0.0, neginf=0.0)

    # DC removal
    stereo = stereo - np.mean(stereo, axis=0, keepdims=True)

    pre_peak = float(np.max(np.abs(stereo)) + 1e-12)
    clipped_fraction = float(np.mean(np.abs(stereo) >= 0.999))
    if clipped_fraction > 0.0001:
        issues.append(f"{clipped_fraction * 100:.3f}% of samples were at or above full "
                       f"scale before safety limiting")

    # Safety limiter: guarantee headroom without touching signals that are
    # already safely under the ceiling.
    if pre_peak > 0.999:
        stereo = stereo / pre_peak * 0.999

    # Approximate true peak via 4x oversampling (inter-sample peaks that a
    # plain sample-peak reading would miss).
    try:
        oversampled = resample_poly(stereo, 4, 1, axis=0)
        true_peak = float(np.max(np.abs(oversampled)) + 1e-12)
    except Exception:
        true_peak = float(np.max(np.abs(stereo)) + 1e-12)
        issues.append("true-peak oversampling unavailable; reporting sample peak instead")

    peak = float(np.max(np.abs(stereo)) + 1e-12)
    rms = float(np.sqrt(np.mean(np.square(stereo))) + 1e-12)
    loudness_dbfs_approx = 20.0 * math.log10(rms)
    peak_dbfs = 20.0 * math.log10(peak)
    true_peak_dbfs = 20.0 * math.log10(true_peak)

    final_ok = bool(
        np.all(np.isfinite(stereo)) and stereo.ndim == 2 and stereo.shape[1] == 2
        and peak <= 1.0 + 1e-6 and stereo.shape[0] > 0
    )
    if not final_ok:
        issues.append("final validation failed one or more checks")

    duration_s = stereo.shape[0] / sr
    stereo_int16 = np.int16(np.clip(stereo, -1.0, 1.0) * 32767)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(stereo_int16.tobytes())

    metadata = {
        "duration_s": round(duration_s, 3),
        "sample_rate": sr,
        "channels": 2,
        "bit_depth": 16,
        "peak_dbfs": round(peak_dbfs, 2),
        "true_peak_dbfs_approx": round(true_peak_dbfs, 2),
        "loudness_dbfs_approx": round(loudness_dbfs_approx, 2),
        "clipped_fraction": round(clipped_fraction, 6),
        "validation_ok": final_ok,
        "validation_issues": issues,
        "dsp_version": DSP_SYNTHESIS_VERSION,
        "mastering_version": MASTERING_VERSION,
    }
    return buf.getvalue(), metadata


def synthesize_psychoacoustic_audio(config: CosmicSomaticConfig,
                                     duration: float = SOUNDSCAPE_SECONDS_DEFAULT,
                                     sample_rate: int = 44100):
    """Public entry point: DSP synthesis -> mastering/validation -> WAV
    bytes. Returns (wav_bytes, metadata_dict)."""
    stereo, sr = _synthesize_raw_stereo(config, duration, sample_rate)
    return _master_and_validate(stereo, sr)


# ---------------------------------------------------------------------------
# 4c. DETERMINISTIC SOUNDSCAPE IDENTITY & CACHING
# ---------------------------------------------------------------------------

def _soundscape_id(raw_query: str, model_name: str, duration: float,
                    cfg: CosmicSomaticConfig) -> str:
    payload = {
        "query": _normalize_object_key(raw_query),
        "model": model_name,
        "duration": round(float(duration), 2),
        "versions": ENGINE_VERSIONS,
        "object_name": cfg.object_name,
        "vectors": cfg.vectors.model_dump(),
        "grounding_tone_hz": round(cfg.grounding_tone_hz, 3),
        "gravity": round(cfg.surface_gravity_g, 6),
        "temperature": round(cfg.surface_temperature_k, 6),
        "pressure": round(cfg.atmospheric_pressure_atm, 9),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def _cache_key(raw_query: str, model_name: str, duration: float) -> str:
    payload = {
        "q": _normalize_object_key(raw_query),
        "model": model_name,
        "duration": round(float(duration), 2),
        "versions": ENGINE_VERSIONS,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def _safe_filename(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "atmosphere"


def _store_feedback(object_name: str, soundscape_id: Optional[str], data: dict) -> bool:
    record = {
        "timestamp": time.time(),
        "object_name": object_name,
        "soundscape_id": soundscape_id,
        **data,
        "schema_version": SCHEMA_VERSION,
        "psychoacoustic_version": PSYCHOACOUSTIC_VERSION,
    }
    try:
        with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return True
    except Exception:
        return False


def _run_search(raw_query: str, duration: float, force_refresh: bool) -> None:
    st.session_state.search_error = None
    st.session_state.fallback_notice = None
    st.session_state.gemini_diagnostics = []
    st.session_state.physics_adjustments = []
    st.session_state.model_used = None

    model_name = st.session_state.gemini_model
    cache = st.session_state.soundscape_cache
    key = _cache_key(raw_query, model_name, duration)

    if not force_refresh and key in cache:
        st.session_state.update(cache[key])
        st.session_state.search_error = None
        return

    with st.spinner(f"Reasoning about the real physics of {raw_query}..."):
        try:
            query = fetch_astronomical_context(raw_query)
            config = generate_psychoacoustic_schema(query)
            provenance, config, correction_notes = _build_provenance(raw_query, config)
            if correction_notes:
                config, recap_notes = _reconcile_with_physics(config)
                st.session_state.physics_adjustments = (
                    st.session_state.physics_adjustments + correction_notes + recap_notes)
            phys = _surface_physics(config)
            perceptual = _derive_perceptual_state(phys)
            config, blend_notes = _apply_perceptual_grounding(config, perceptual)
            st.session_state.physics_adjustments = st.session_state.physics_adjustments + blend_notes
            material = _choose_material(phys)
            audio_bytes, audio_metadata = synthesize_psychoacoustic_audio(config, duration=duration)
        except GeminiUnavailableError as e:
            st.session_state.search_error = str(e)
            if e.details and not st.session_state.gemini_diagnostics:
                st.session_state.gemini_diagnostics = list(e.details)
        else:
            soundscape_id = _soundscape_id(raw_query, model_name, duration, config)
            explanation_lines = _explanation_lines(
                config, phys, material, st.session_state.physics_adjustments)
            accessible_text = _accessible_description(config, perceptual, provenance)
            emotional_sentence = _emotional_sentence(perceptual)

            result_state = {
                "selected_query": raw_query,
                "profile_data": config,
                "audio_bytes": audio_bytes,
                "audio_metadata": audio_metadata,
                "provenance": provenance,
                "perceptual": perceptual,
                "explanation_lines": explanation_lines,
                "accessible_text": accessible_text,
                "emotional_sentence": emotional_sentence,
                "soundscape_id": soundscape_id,
                "model_used": st.session_state.model_used,
                "fallback_notice": st.session_state.fallback_notice,
                "gemini_diagnostics": st.session_state.gemini_diagnostics,
                "physics_adjustments": st.session_state.physics_adjustments,
                "search_error": None,
            }
            st.session_state.update(result_state)
            cache[key] = dict(result_state)
            while len(cache) > CACHE_MAX_ENTRIES:
                cache.popitem(last=False)


# ---------------------------------------------------------------------------
# 5. ACCESSIBLE STREAMLIT USER INTERFACE
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Space+Grotesk:wght@400;500;600&display=swap');

.stApp {
    background: #07060d;
    color: #f1eef7;
}
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}
h1.cosmic-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.6rem;
    text-align: center;
    margin-bottom: 0.2rem;
    color: #f1eef7;
}
p.cosmic-subtitle {
    text-align: center;
    color: #9891ab;
    font-size: 1.05rem;
    max-width: 620px;
    margin: 0 auto 2.5rem auto;
}
h2.cosmic-object-name {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.9rem;
    color: #f1eef7;
    margin-top: 1.5rem;
}
div.stButton button {
    font-family: 'Space Grotesk', sans-serif;
    background: #ff6a4d;
    color: #07060d;
    border: none;
    border-radius: 24px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    font-size: 1.05rem;
    transition: transform 0.15s ease, background 0.15s ease;
}
div.stButton button:hover {
    transform: translateY(-1px);
    background: #ff8266;
    color: #07060d;
}
:focus-visible {
    outline: 2px solid #ff6a4d !important;
    outline-offset: 2px;
}
/* Give the main search box a rounded, pill-shaped "Google-style" look
   (Streamlit adds a st-key-<key> class to any widget given a key=...). */
.st-key-cosmic_query_input div[data-baseweb="select"] > div {
    border-radius: 28px !important;
    border: 2px solid #2a2740 !important;
    background: #12111f !important;
    font-size: 1.1rem !important;
}
.st-key-cosmic_query_input div[data-baseweb="select"]:focus-within > div {
    border-color: #ff6a4d !important;
}
.narrative-block {
    font-family: 'Fraunces', serif;
    font-style: italic;
    font-size: 1.3rem;
    line-height: 1.65;
    color: #f1eef7;
    border-left: 2px solid #ff6a4d;
    padding-left: 1.2rem;
    margin: 1.8rem 0;
}
.physics-line {
    font-size: 0.95rem;
    color: #9891ab;
    margin: 0.15rem 0 1.2rem 0;
}
.emotional-sentence {
    color: #cfc9de;
    font-size: 1.0rem;
    font-style: italic;
    margin: 0 0 1.4rem 0;
}
.unverified-tag, .confidence-tag {
    display: inline-block;
    font-size: 0.72rem;
    letter-spacing: 0.02em;
    border-radius: 10px;
    padding: 0.05rem 0.5rem;
    margin-left: 0.5rem;
    border: 1px solid #ff6a4d;
    color: #ff6a4d;
    vertical-align: middle;
}
.vector-row {
    display: flex;
    justify-content: space-between;
    padding: 0.35rem 0;
    border-bottom: 1px solid #1c1a2b;
    font-size: 0.95rem;
    color: #cfc9de;
}
.sr-only {
    position: absolute;
    width: 1px; height: 1px;
    padding: 0; margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}
</style>
"""

# Suggestions are just a typing shortcut for the search box - not a lookup
# table. Any name at all, known or not, goes straight to Gemini.
SUGGESTION_LIST = sorted([
    "Earth", "Venus", "Mars", "Jupiter", "Saturn", "Mercury", "Neptune", "Uranus", "Pluto",
    "The Moon", "Europa", "Titan", "Io", "Ganymede", "Enceladus", "Triton",
    "The Sun", "Betelgeuse", "Sirius", "Proxima Centauri", "Alpha Centauri",
    "A black hole", "A neutron star", "A pulsar", "A white dwarf", "A supernova",
    "Andromeda Galaxy", "The Milky Way", "Crab Nebula", "Orion Nebula", "Sagittarius A*",
    "Halley's Comet", "the Kuiper Belt", "the Asteroid Belt", "TRAPPIST-1e", "Kepler-452b",
])

SEARCH_PLACEHOLDER = "Type ANY object: Earth, Venus, a black hole, TRAPPIST-1e..."

_CONFIDENCE_COLORS = {
    "VERIFIED": "#4caf7d", "DERIVED": "#6aa6ff", "ESTIMATED": "#ffb84d",
    "HYPOTHETICAL": "#c792ea", "UNKNOWN": "#ff6a4d",
}


def _confidence_badge_html(confidence: str) -> str:
    color = _CONFIDENCE_COLORS.get(confidence, "#9891ab")
    return (f"<span class='confidence-tag' style='color:{color};border-color:{color};' "
            f"title='{CONFIDENCE_EXPLANATION.get(confidence, '')}'>{confidence}</span>")


def _render_search_box() -> Optional[str]:
    """Searchable dropdown with live suggestions from the first letter typed
    (fuzzy matching on Streamlit >= 1.56). Degrades gracefully on older
    Streamlit versions instead of crashing. Purely a typing aid - any text
    typed is accepted and sent straight to Gemini."""
    base = dict(
        options=SUGGESTION_LIST,
        index=None,
        placeholder=SEARCH_PLACEHOLDER,
        label_visibility="collapsed",
        key="cosmic_query_input",
    )
    label = "Search any astronomical object"
    try:
        return st.selectbox(label, accept_new_options=True, filter_mode="fuzzy", **base)
    except TypeError:
        pass
    try:
        return st.selectbox(label, accept_new_options=True, **base)
    except TypeError:
        st.caption("Upgrade Streamlit (pip install -U streamlit) for live search suggestions.")
        return st.text_input(
            label, placeholder=SEARCH_PLACEHOLDER,
            label_visibility="collapsed", key="cosmic_query_input",
        )


def _render_mode_selector() -> str:
    options = ["Experience", "Understand", "Science"]
    help_text = ("Experience: just listen. Understand: see why this specific object sounds "
                 "this way. Science: this object's full physical data, provenance and raw "
                 "vectors.")
    try:
        mode = st.segmented_control("View", options, default="Experience", help=help_text,
                                     key="view_mode")
        return mode or "Experience"
    except Exception:
        return st.radio("View", options, index=0, horizontal=True, help=help_text,
                         key="view_mode_radio")


def _render_result_header(config: CosmicSomaticConfig) -> None:
    provenance: Optional[Provenance] = st.session_state.provenance
    badge = _confidence_badge_html(provenance.overall_confidence) if provenance else ""
    st.markdown(
        f"<h2 class='cosmic-object-name'>{config.object_name}{badge}</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='physics-line'>{config.surface_gravity_g:.2f} g &nbsp;&middot;&nbsp; "
        f"{config.surface_temperature_k:.0f} K &nbsp;&middot;&nbsp; "
        f"{config.atmospheric_pressure_atm:.3g} atm &nbsp;&middot;&nbsp; "
        f"{config.surface_composition_notes}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='narrative-block'>{config.mental_presence_narrative}</div>",
        unsafe_allow_html=True,
    )
    if st.session_state.emotional_sentence:
        st.markdown(
            f"<p class='emotional-sentence'>{st.session_state.emotional_sentence}</p>",
            unsafe_allow_html=True,
        )
    if st.session_state.accessible_text:
        hidden = st.session_state.accessible_text.replace("\n", " ")
        st.markdown(
            f"<div class='sr-only' aria-live='polite'>{hidden}</div>",
            unsafe_allow_html=True,
        )

    st.audio(st.session_state.audio_bytes, format="audio/wav")
    if st.session_state.model_used:
        st.caption(f"Analyzed by {_model_label(st.session_state.model_used)}")
    st.download_button(
        "Download soundscape (WAV)",
        data=st.session_state.audio_bytes,
        file_name=f"{_safe_filename(config.object_name)}_atmosphere.wav",
        mime="audio/wav",
        use_container_width=True,
    )
    with st.expander("Full text description (for screen readers and low-vision users)"):
        st.markdown(st.session_state.accessible_text or "")


def _render_understand_section(config: CosmicSomaticConfig) -> None:
    st.markdown(f"### Why does {config.object_name} sound like this?")
    lines = st.session_state.explanation_lines or []
    if not lines:
        st.caption(f"No notable transformations to report for {config.object_name}.")
    for line in lines:
        st.markdown(f"- {line}")
    st.caption("Only transformations the engine actually applied are listed above; "
               "nothing here is invented after the fact.")


def _render_feedback_form(soundscape_id: Optional[str], object_name: str) -> None:
    with st.expander(f"Help improve {object_name}'s soundscape (optional, anonymous feedback)"):
        st.caption("Stored locally in a JSONL file next to the app for future research "
                   "use. No personal data is collected beyond what you enter here.")
        with st.form(key="feedback_form"):
            scale = st.slider("Perceived scale (small - vast)", 0.0, 1.0, 0.5)
            openness = st.slider("Perceived openness (enclosed - open)", 0.0, 1.0, 0.5)
            calm = st.slider("Perceived calm (tense - calm)", 0.0, 1.0, 0.5)
            isolation = st.slider("Perceived isolation (crowded - solitary)", 0.0, 1.0, 0.5)
            activity = st.slider("Perceived environmental activity (still - busy)", 0.0, 1.0, 0.5)
            realism = st.slider("Perceived realism (artificial - believable)", 0.0, 1.0, 0.5)
            clarity = st.slider("Perceived clarity (confusing - clear)", 0.0, 1.0, 0.5)
            helped = st.radio("Did this help you imagine the environment?",
                               ["Yes", "Somewhat", "No"], horizontal=True)
            comment = st.text_area("Anything else? (optional)", max_chars=500)
            submitted = st.form_submit_button("Submit feedback")
        if submitted:
            ok = _store_feedback(object_name, soundscape_id, {
                "perceived_scale": scale, "perceived_openness": openness,
                "perceived_calm": calm, "perceived_isolation": isolation,
                "perceived_activity": activity, "perceived_realism": realism,
                "perceived_clarity": clarity, "helped_imagine": helped,
                "comment": comment.strip()[:500],
            })
            if ok:
                st.success("Thank you - feedback recorded.")
            else:
                st.warning("Feedback could not be saved to disk in this environment, "
                           "but thank you for trying.")


def _render_science_section(config: CosmicSomaticConfig) -> None:
    obj_name = config.object_name
    st.markdown(f"### Scientific provenance for {obj_name}")
    provenance: Optional[Provenance] = st.session_state.provenance
    if provenance:
        rows = [
            ("Surface gravity", provenance.gravity),
            ("Surface temperature", provenance.temperature),
            ("Atmospheric pressure", provenance.pressure),
        ]
        for label, pv in rows:
            badge = _confidence_badge_html(pv.confidence)
            st.markdown(
                f"**{label}:** {pv.value:.3g} {pv.unit} {badge}  \n"
                f"<span style='color:#9891ab;font-size:0.85rem;'>{pv.source}"
                + (f" - {pv.notes}" if pv.notes else "") + "</span>",
                unsafe_allow_html=True,
            )
        if provenance.matched_reference:
            st.caption(f"{obj_name} was cross-checked against embedded reference data for "
                       f"{provenance.matched_reference.title()}.")
        else:
            st.caption(f"No embedded reference entry matched {obj_name}; these values are "
                       f"Gemini's reasoning about {obj_name} specifically, not independently "
                       f"verified by this app.")

    st.markdown(f"### {obj_name}'s perceptual state (0.0-1.0, designed interpretation)")
    perceptual: Optional[PerceptualState] = st.session_state.perceptual
    if perceptual:
        for dim_name, value in dataclasses.asdict(perceptual).items():
            st.markdown(
                f"<div class='vector-row'><span>{dim_name.replace('_', ' ').capitalize()}"
                f"</span><span>{value:.2f}</span></div>",
                unsafe_allow_html=True,
            )
    st.caption(f"A deterministic, physics-grounded design model computed from {obj_name}'s own "
               f"stated gravity, temperature and pressure - not a measured or universal mapping "
               f"of what any listener will actually feel.")

    st.markdown(f"### Psychoacoustic vectors driving {obj_name}'s soundscape")
    summary = _engine_summary(config)
    rows = [
        (vector_name.replace("_", " ").capitalize(), f"{value:.2f}")
        for vector_name, value in config.vectors.model_dump().items()
    ]
    rows += [
        ("Grounding tone", f"{config.grounding_tone_hz:.0f} Hz"),
        ("Engine-derived material", summary["material"]),
        (f"Natural events {obj_name} allows", ", ".join(summary["events"])),
        ("Atmosphere carries sound",
         "yes" if summary["airborne"] else "no (near-silent, structure-borne only)"),
    ]
    for label, value in rows:
        st.markdown(
            f"<div class='vector-row'><span>{label}</span><span>{value}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown(f"### What phenomena {obj_name} allows, and why")
    phys = _surface_physics(config)
    for category, allowed, reason in _phenomena_report(phys):
        mark = "Allowed" if allowed else "Suppressed"
        st.markdown(
            f"<div class='vector-row'><span>{category}</span>"
            f"<span>{mark} - {reason}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown(f"### Audio metadata & validation for {obj_name}")
    meta = st.session_state.audio_metadata or {}
    for k, v in meta.items():
        if k == "validation_issues":
            continue
        st.markdown(
            f"<div class='vector-row'><span>{k.replace('_', ' ')}</span><span>{v}</span></div>",
            unsafe_allow_html=True,
        )
    issues = meta.get("validation_issues") or []
    if issues:
        st.warning(f"Audio validation notes for {obj_name}: " + "; ".join(issues))
    else:
        st.caption(f"{obj_name}'s audio passed all validation checks: no NaN/Inf, no clipping, "
                   f"bounded amplitude.")

    st.markdown(f"### Reproducing {obj_name}'s soundscape")
    st.code(
        f"Soundscape ID: {st.session_state.soundscape_id}\n"
        f"Schema v{SCHEMA_VERSION}  Physics v{PHYSICS_MODEL_VERSION}  "
        f"Provenance v{PROVENANCE_MODEL_VERSION}\n"
        f"Psychoacoustic v{PSYCHOACOUSTIC_VERSION}  DSP v{DSP_SYNTHESIS_VERSION}  "
        f"Mastering v{MASTERING_VERSION}",
        language=None,
    )

    _render_feedback_form(st.session_state.soundscape_id, config.object_name)


def main() -> None:
    st.set_page_config(
        page_title="Sounds of the Universe",
        page_icon="\U0001F30C",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_session_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown("<h1 class='cosmic-title'>Sounds of the Universe</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='cosmic-subtitle'>Type any astronomical object, real or hypothetical. "
        "Gemini reasons out its real surface physics live, with no built-in catalogue; this "
        "app then checks that reasoning, derives an independent perceptual model from it, and "
        "turns the two together into what it might feel like to stand there - scale, openness, "
        "stillness, pressure, isolation - without claiming any single emotional response is "
        "universal.</p>",
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1, 3, 1])
    with center:
        # Placeholder containers, created in the order they should appear on
        # screen: engine selector, then search, then the View toggle and its
        # Advanced settings (now BELOW the search box per the UI reorder),
        # then messages, technical details, and finally the results. A
        # container keeps its reserved position regardless of which order it
        # is filled in below, which lets the Advanced settings values
        # (duration, force refresh) be read before the Search button's own
        # container is filled, even though they are drawn further down the page.
        container_engine = st.container()
        container_search = st.container()
        container_view = st.container()
        container_messages = st.container()
        container_tech = st.container()
        container_results = st.container()

        with container_view:
            mode = _render_mode_selector()
            with st.expander("Advanced settings"):
                duration = st.slider(
                    "Soundscape duration (seconds)",
                    SOUNDSCAPE_SECONDS_MIN, SOUNDSCAPE_SECONDS_MAX,
                    value=int(st.session_state.duration), step=2,
                    help="How long the generated soundscape should be.",
                    key="duration_slider",
                )
                force_refresh = st.checkbox(
                    "Force regenerate (ignore cache)", value=False,
                    help="Gemini's reasoning can vary between calls; check this to get a "
                         "fresh interpretation instead of a cached one for the same object.",
                    key="force_refresh_checkbox",
                )
            st.session_state.duration = float(duration)

        with container_engine:
            current_label = next(
                (label for label, model in MODEL_OPTIONS.items() if model == st.session_state.gemini_model),
                list(MODEL_OPTIONS.keys())[0],
            )
            selected_label = st.selectbox(
                "Synthesis engine",
                list(MODEL_OPTIONS.keys()),
                index=list(MODEL_OPTIONS.keys()).index(current_label),
                help="Choose which Gemini model reasons about the object's physics.",
                key="synthesis_engine_select",
            )
            st.session_state.gemini_model = MODEL_OPTIONS[selected_label]

        with container_search:
            typed_query = _render_search_box()
            search_clicked = st.button("Search", use_container_width=True, key="search_button")

            if search_clicked:
                clean_query = (typed_query or "").strip()[:MAX_QUERY_CHARS]
                if not clean_query:
                    st.warning("Type or pick the name of an astronomical object first.")
                else:
                    _run_search(clean_query, float(duration), force_refresh)

        with container_messages:
            if st.session_state.search_error:
                st.error(st.session_state.search_error)
            elif st.session_state.fallback_notice:
                st.info(st.session_state.fallback_notice)

        with container_tech:
            if st.session_state.gemini_diagnostics or st.session_state.physics_adjustments:
                with st.expander("Technical details (what Gemini stated, and any corrections made)"):
                    if st.session_state.gemini_diagnostics:
                        st.code("\n".join(st.session_state.gemini_diagnostics), language=None)
                    if st.session_state.physics_adjustments:
                        st.caption("Vectors adjusted before synthesis (physical caps, "
                                   "reference-data corrections, and perceptual grounding):")
                        st.code("\n".join(st.session_state.physics_adjustments), language=None)

        with container_results:
            if st.session_state.profile_data is not None:
                config: CosmicSomaticConfig = st.session_state.profile_data
                _render_result_header(config)
                if mode in ("Understand", "Science"):
                    _render_understand_section(config)
                if mode == "Science":
                    _render_science_section(config)


# ---------------------------------------------------------------------------
# 6. OFFLINE SELF-TESTS
#    Exercises the physics model, perceptual model, provenance logic and DSP
#    engine directly, entirely bypassing Gemini (no API key or network
#    required). This is not a substitute for a real pytest suite split into
#    tests/unit, tests/scientific and tests/audio as the project grows - see
#    the limitations note in the project summary - but it gives fast,
#    dependency-free coverage of the invariants that matter most: physical
#    plausibility, numerical safety, and determinism.
#
#    Run with:  python app.py --selftest
# ---------------------------------------------------------------------------

def _make_test_config(name, gravity, temperature, pressure, notes, tone) -> CosmicSomaticConfig:
    return CosmicSomaticConfig(
        object_name=name,
        surface_gravity_g=gravity,
        surface_temperature_k=temperature,
        atmospheric_pressure_atm=pressure,
        surface_composition_notes=notes,
        mental_presence_narrative=(
            f"You stand on {name}, and the air around you carries exactly what its "
            f"physics allows - nothing more, nothing invented."
        ),
        vectors=AtmosphereVectors(
            hollow_isolation_factor=0.5, thermal_manic_hum=0.5,
            harmonic_dissonance_index=0.5, rhythmic_unpredictability=0.5,
            claustrophobic_suffocation=0.5, infinite_spatial_drift=0.5,
            tearing_friction_index=0.5,
        ),
        grounding_tone_hz=tone,
    )


_TEST_ENVIRONMENTS = [
    _make_test_config("Earth-like", 1.0, 288.0, 1.0, "grass, soil and open water", 120.0),
    _make_test_config("Mars-like thin atmosphere", 0.38, 210.0, 0.0063,
                       "basaltic dust under a thin carbon dioxide atmosphere", 150.0),
    _make_test_config("Venus-like dense hot atmosphere", 0.9, 737.0, 92.0,
                       "a dense, scorching carbon dioxide atmosphere over basalt", 70.0),
    _make_test_config("Moon-like near-vacuum", 0.166, 220.0, 1e-14,
                       "UNVERIFIED: bare regolith dust in a near-total vacuum", 220.0),
    _make_test_config("Titan-like dense cold atmosphere", 0.14, 94.0, 1.45,
                       "a dense, cold nitrogen atmosphere over hydrocarbon lakes", 95.0),
    _make_test_config("Icy world", 0.3, 95.0, 0.0,
                       "UNVERIFIED: a water-ice crust under vacuum", 260.0),
    _make_test_config("Molten world", 1.4, 1500.0, 0.4,
                       "UNVERIFIED: an open lava surface with volcanic gases", 60.0),
    _make_test_config("Hypothetical airless world", 0.05, 50.0, 0.0,
                       "UNVERIFIED: a hypothetical bare airless rock", 300.0),
]


def _check(label: str, condition: bool, failures: list) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        failures.append(label)


def run_self_tests() -> int:
    failures = []
    print("Running offline self-tests (no network / API key required)...\n")

    print("Physics & reconciliation invariants:")
    for cfg in _TEST_ENVIRONMENTS:
        sanitized = _sanitize_config(cfg)
        reconciled, _ = _reconcile_with_physics(sanitized)
        phys = _surface_physics(reconciled)
        material = _choose_material(phys)
        events = [k for k, _ in _event_plan(phys)]
        name = cfg.object_name

        _check(f"{name}: vectors stay within [0, 1]",
               all(0.0 <= getattr(reconciled.vectors, n) <= 1.0 for n in VECTOR_NAMES),
               failures)

        if not phys.airborne:
            _check(f"{name}: airless world does not choose an airborne material",
                   material not in ("wind", "rumble"), failures)
            _check(f"{name}: airless world does not generate biological or "
                   f"wind/weather events",
                   not any(k in events for k in ("bird", "rustle", "gust", "swell")),
                   failures)
        if phys.habitable:
            _check(f"{name}: habitable world keeps dissonance and heat low",
                   reconciled.vectors.harmonic_dissonance_index <= 0.2
                   and reconciled.vectors.thermal_manic_hum <= 0.2, failures)
        if phys.dense:
            _check(f"{name}: dense atmosphere floors claustrophobic suffocation",
                   reconciled.vectors.claustrophobic_suffocation >= 0.55 - 1e-9, failures)
        if not phys.molten:
            _check(f"{name}: non-molten world does not plan bubble/pop events",
                   not any(k in events for k in ("bubble", "pop")), failures)

    print("\nPerceptual-state bounds:")
    for cfg in _TEST_ENVIRONMENTS:
        phys = _surface_physics(_sanitize_config(cfg))
        perceptual = _derive_perceptual_state(phys)
        values = dataclasses.asdict(perceptual)
        _check(f"{cfg.object_name}: all perceptual dimensions in [0, 1]",
               all(0.0 <= v <= 1.0 for v in values.values()), failures)

    print("\nProvenance classification:")
    earth_cfg = _make_test_config("Earth", 1.0, 288.0, 1.0, "grass and water", 120.0)
    prov, corrected, notes = _build_provenance("Earth", earth_cfg)
    _check("Earth matches the reference table", prov.matched_reference == "earth", failures)
    _check("Earth (accurate input) is classified VERIFIED",
           prov.overall_confidence == "VERIFIED", failures)
    bad_venus = _make_test_config("Venus", 1.0, 288.0, 1.0, "grass and water", 120.0)
    prov2, corrected2, notes2 = _build_provenance("Venus", bad_venus)
    _check("Wildly wrong Venus input gets corrected toward reference data",
           abs(corrected2.surface_temperature_k - 737.0) < 1.0 and len(notes2) > 0, failures)
    fictional = _make_test_config("Zorblax Prime", 0.5, 400.0, 2.0,
                                   "UNVERIFIED: a hypothetical world", 100.0)
    prov3, _, _ = _build_provenance("a hypothetical world called Zorblax Prime", fictional)
    _check("An explicitly hypothetical, unmatched object is classified HYPOTHETICAL",
           prov3.overall_confidence == "HYPOTHETICAL", failures)

    print("\nAudio synthesis & mastering (short 2s renders for speed):")
    for cfg in _TEST_ENVIRONMENTS:
        sanitized = _sanitize_config(cfg)
        reconciled, _ = _reconcile_with_physics(sanitized)
        stereo, sr = _synthesize_raw_stereo(reconciled, duration=2.0)
        name = cfg.object_name
        _check(f"{name}: audio contains no NaN/Inf", bool(np.all(np.isfinite(stereo))), failures)
        _check(f"{name}: audio peak stays within [-1, 1]",
               float(np.max(np.abs(stereo))) <= 1.0 + 1e-6, failures)
        _check(f"{name}: correct sample count for 2s @ {sr} Hz",
               abs(stereo.shape[0] - int(2.0 * sr)) <= 1, failures)
        _check(f"{name}: output is stereo", stereo.shape[1] == 2, failures)

        wav_bytes, meta = _master_and_validate(stereo, sr)
        _check(f"{name}: mastered audio passes validation", meta["validation_ok"], failures)
        _check(f"{name}: mastered audio has no clipping",
               meta["clipped_fraction"] < 0.0005, failures)
        _check(f"{name}: WAV bytes are non-empty", len(wav_bytes) > 100, failures)

        stereo2, _ = _synthesize_raw_stereo(reconciled, duration=2.0)
        _check(f"{name}: synthesis is deterministic for identical input",
               np.array_equal(stereo, stereo2), failures)

    print(f"\n{len(_TEST_ENVIRONMENTS) * 10 + 8 - len(failures)} checks passed, "
          f"{len(failures)} failed.")
    if failures:
        print("\nFailed checks:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll self-tests passed.")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(run_self_tests())
    main()
