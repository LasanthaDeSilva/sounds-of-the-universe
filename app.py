"""
Sounds of the Universe
-----------------------
A self-analysing psychoacoustic simulator for blind and visually impaired
(BVI) users. Type ANY astronomical object - real, obscure or hypothetical.
There is no catalogue inside this app. Gemini is asked, as a factual
Somatic Surface Simulator, for the real physical boundary conditions of the
place a listener would stand (surface gravity, temperature, atmospheric
pressure, what is under their feet) BEFORE it is allowed to choose any sound
vectors. Those stated physics then drive the synthesiser directly:

  * pressure and temperature decide whether air carries sound at all, how
    deep the currents roll, and how fast sound travels (pitch of the air);
  * composition and heat decide which natural sources exist - breezes and
    birdsong on a habitable world, cracking ice on a frozen one, lava
    crackle on a molten one, dust and settling grit on an airless rock;
  * gravity sets how quickly falling grit and cracks settle;
  * every vector Gemini returns is checked against the physics it just
    stated, and anything physically impossible is corrected and reported.

No artificial hums, buzzes or sirens are ever synthesised. Where Gemini does
not genuinely know an object it must say so ("UNVERIFIED"), and the app
shows that plainly instead of dressing a guess up as fact.

Failure handling is honest, never generic:
  * The selected Gemini model gets a few retries. Errors are classified from
    the real API response (HTTP code / status), not guessed.
  * Only genuinely traffic-related failures (HTTP 429/5xx, timeouts,
    dropped connections) trigger one real attempt with the lighter
    Gemini 3.5 Flash Lite - and the app says when and why.
  * Any other failure (bad request, missing model, bad key) is shown as-is.
    A "Technical details" panel always shows what Gemini returned.
  * There is no canned or generic sound anywhere in this app.

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
"""

import dataclasses
import hashlib
import io
import json
import math
import os
import re
import time
import wave
from typing import Optional

import numpy as np
import streamlit as st
from pydantic import BaseModel, Field
from scipy.signal import butter, fftconvolve, sosfilt

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - the UI still loads without the SDK
    genai = None
    types = None


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

SOUNDSCAPE_SECONDS = 24.0


def init_session_state() -> None:
    defaults = {
        "selected_query": None,
        "profile_data": None,
        "audio_bytes": None,
        "gemini_model": GEMINI_FLASH_MODEL,
        "search_error": None,
        "fallback_notice": None,
        "model_used": None,
        "gemini_diagnostics": [],
        "physics_adjustments": [],
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
# 2. PYDANTIC STRUCTURED OUTPUT SCHEMA (THE PSYCHOLOGICAL ENGINE)
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
    vacuum...). Returns (config, list of human-readable corrections)."""
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
# 3. INGESTION - fully open-ended, no local catalogue
# ---------------------------------------------------------------------------
# There is no hardcoded object list. Gemini itself must know or reason about
# whatever was typed; fetch_astronomical_context only trims the raw text.

def fetch_astronomical_context(query: str) -> str:
    """Clean the user's raw query. No lookup, no catalogue, no guessing -
    the physics comes from Gemini's own stated CosmicSomaticConfig fields,
    checked in _validate_physics/_reconcile_with_physics."""
    cleaned = " ".join((query or "").split())
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
    "estimates are fine, invented false precision is not.\n\n"
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
    "second person, present tense, built only from the physics you stated."
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
    derive the psychoacoustic profile strictly from them - fully open-ended,
    no local catalogue. The selected model gets its full retries; if (and
    only if) it failed for traffic reasons, one real attempt is made with
    Gemini 3.5 Flash Lite. Non-traffic failures are raised as-is. Every
    vector is then reconciled against the physics Gemini itself stated, and
    any correction is recorded for the UI. Never returns a fabricated result."""
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
        "(or nearest meaningful physical boundary) and derive the "
        "psychoacoustic profile for a listener standing there, strictly from "
        "those physics."
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

def _root_drone(t, sr, rng, root, thermal, dissonance, tranquil: bool):
    """The place's tonal centre: a harmonic stack that gets brighter and more
    chorused with heat, sweetened with a fifth/octave when calm and clashed
    with a tritone/minor second when hostile.

    For an earth-like or otherwise tranquil world, `spread` is hardcoded to
    0.0: the three voices lock to one pure, warm, unbeating pitch instead of
    detuning against each other, so the mind has a clean, peaceful tone to
    rest on. Everywhere else spread still grows gently with heat.

    For very low roots the fundamental is turned down and the interval
    partials are lifted by octaves: the ear still hears the low pitch from
    the harmonics (the "missing fundamental" effect), but the mix stops
    piling energy into the sub-bass where it turns to mud."""
    vibrato = 1.0 + 0.003 * np.sin(TWO_PI * 0.17 * t + rng.uniform(0, TWO_PI))
    phase = TWO_PI * np.cumsum(root * vibrato) / sr
    n_max = int(np.clip(2200.0 / root, 6, 24))
    n_harm = int(4 + (n_max - 4) * (0.3 + 0.7 * thermal))
    slope = 1.1 - 0.5 * thermal
    spread = 0.0 if tranquil else (0.0035 + 0.02 * thermal)
    fund_weight = float(np.clip(root / 120.0, 0.12, 1.0))
    sig = np.zeros_like(t)
    for ratio in (1.0, 1.0 + spread, 1.0 - 0.8 * spread):
        ph = phase * ratio + rng.uniform(0, TWO_PI)
        for k in range(1, n_harm + 1):
            if root * k * ratio > 0.45 * sr:
                break
            weight = fund_weight if k == 1 else (0.5 + 0.5 * fund_weight if k == 2 else 1.0)
            sig += weight * np.sin(k * ph) / k ** slope
    sig = _rms_norm(sig, 1.0)
    lift = 1.0 if root >= 100.0 else 2.0 ** math.ceil(math.log2(100.0 / root))
    lp = phase * lift
    consonant = np.sin(1.5 * lp + 0.7) + 0.6 * np.sin(2.0 * lp + 1.9)
    clash = np.sin(math.sqrt(2.0) * lp + 0.3) + 0.8 * np.sin((16.0 / 15.0) * lp + 2.2)
    sig = sig + (1.0 - dissonance) * 0.4 * _rms_norm(consonant, 1.0) \
        + (dissonance ** 1.2) * 0.75 * _rms_norm(clash, 1.0)
    return _rms_norm(sig, 1.0)


def _thermal_cluster(t, root, thermal, dissonance):
    """Layer 2 - the manic star core: three mid-range oscillators, octave-
    folded from the place's root so it stays in tune with it, detuned
    further by the dissonance index, given richer harmonics and a faster
    tremor as things get hotter. Gain-staged by thermal_manic_hum downstream,
    which _reconcile_with_physics already pinned near 0 for cool surfaces -
    so this simply does not sound on a temperate world."""
    mid = _fold_freq(root, 170.0, 340.0)
    freqs = (mid, mid * (1.0 + 0.007 + 0.05 * dissonance), mid * (1.0 + 0.014 + 0.09 * dissonance))
    richness = int(2 + 8 * thermal)
    sig = np.zeros_like(t)
    for f in freqs:
        ph = TWO_PI * f * t
        for k in range(1, richness + 1):
            sig += np.sin(k * ph + 0.9 * k) / k ** (1.4 - 0.6 * thermal)
    tremor = 1.0 + 0.3 * thermal * np.sin(TWO_PI * (3.0 + 5.0 * thermal) * t)
    return _rms_norm(sig * tremor, 1.0)


def _sub_drone(t, root, rng):
    sub_f = _fold_freq(root, 30.0, 62.0)
    swell = 0.65 + 0.35 * np.sin(TWO_PI * 0.045 * t + rng.uniform(0, TWO_PI))
    return _rms_norm((np.sin(TWO_PI * sub_f * t) + 0.3 * np.sin(TWO_PI * 2 * sub_f * t + 1.1)) * swell, 1.0)


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


def synthesize_psychoacoustic_audio(config: CosmicSomaticConfig, duration: float = 10.0,
                                     sample_rate: int = 44100) -> bytes:
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

    # Slow "weather": the whole place breathes over time so it never sits still.
    breathe_rate = 0.03 if tranquil else 0.055
    evolve = 0.9 + 0.1 * np.sin(TWO_PI * breathe_rate * t + rng.uniform(0, TWO_PI))

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
    cluster = _thermal_cluster(t, root, thermal, dissonance) * (
        1.0 + 0.25 * np.sin(TWO_PI * 0.09 * t + rng.uniform(0, TWO_PI)))
    dl, dr = _autopan(drone, t, 0.15 + 0.85 * drift, 0.05, rng.uniform(0, TWO_PI))
    cl, cr = _autopan(cluster, t, 0.15 + 0.85 * drift, 0.031, rng.uniform(0, TWO_PI))

    g_bed = _MATERIAL_BED_GAIN.get(material, 0.15)
    g_drone = 0.15
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
    sub_scale = 1.0 if phys.airborne else 0.35
    sub = _sub_drone(t, root, rng) * (0.01 + 0.05 * hollow) * sub_scale
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

    stereo_int16 = np.int16(stereo * 32767)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo_int16.tobytes())
    return buf.getvalue()


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
    max-width: 560px;
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
.unverified-tag {
    display: inline-block;
    font-size: 0.75rem;
    letter-spacing: 0.02em;
    color: #ff6a4d;
    border: 1px solid #ff6a4d;
    border-radius: 10px;
    padding: 0.05rem 0.5rem;
    margin-left: 0.5rem;
}
.vector-row {
    display: flex;
    justify-content: space-between;
    padding: 0.35rem 0;
    border-bottom: 1px solid #1c1a2b;
    font-size: 0.95rem;
    color: #cfc9de;
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


def _safe_filename(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "atmosphere"


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


def _run_search(raw_query: str) -> None:
    st.session_state.search_error = None
    st.session_state.fallback_notice = None
    st.session_state.gemini_diagnostics = []
    st.session_state.physics_adjustments = []
    st.session_state.model_used = None
    with st.spinner(f"Reasoning about the real physics of {raw_query}..."):
        try:
            query = fetch_astronomical_context(raw_query)
            config = generate_psychoacoustic_schema(query)
            audio_bytes = synthesize_psychoacoustic_audio(config, duration=SOUNDSCAPE_SECONDS)
        except GeminiUnavailableError as e:
            st.session_state.search_error = str(e)
            if e.details and not st.session_state.gemini_diagnostics:
                st.session_state.gemini_diagnostics = list(e.details)
        else:
            st.session_state.selected_query = raw_query
            st.session_state.profile_data = config
            st.session_state.audio_bytes = audio_bytes


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
        "<p class='cosmic-subtitle'>Type any astronomical object, real or hypothetical. Gemini "
        "reasons out its real surface physics live, with no built-in catalogue, and the engine "
        "turns those physics directly into what it would feel like to stand there.</p>",
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1, 3, 1])
    with center:
        current_label = next(
            (label for label, model in MODEL_OPTIONS.items() if model == st.session_state.gemini_model),
            list(MODEL_OPTIONS.keys())[0],
        )
        selected_label = st.selectbox(
            "Synthesis engine",
            list(MODEL_OPTIONS.keys()),
            index=list(MODEL_OPTIONS.keys()).index(current_label),
            help="Choose which Gemini model reasons about the object's physics.",
        )
        st.session_state.gemini_model = MODEL_OPTIONS[selected_label]

        typed_query = _render_search_box()
        search_clicked = st.button("Search", use_container_width=True)

        if search_clicked:
            clean_query = (typed_query or "").strip()
            if not clean_query:
                st.warning("Type or pick the name of an astronomical object first.")
            else:
                _run_search(clean_query)

        if st.session_state.search_error:
            st.error(st.session_state.search_error)
        elif st.session_state.fallback_notice:
            st.info(st.session_state.fallback_notice)

        if st.session_state.gemini_diagnostics or st.session_state.physics_adjustments:
            with st.expander("Technical details (what Gemini stated, and any corrections made)"):
                if st.session_state.gemini_diagnostics:
                    st.code("\n".join(st.session_state.gemini_diagnostics), language=None)
                if st.session_state.physics_adjustments:
                    st.caption("Vectors adjusted to match the stated physics before synthesis:")
                    st.code("\n".join(st.session_state.physics_adjustments), language=None)

        if st.session_state.profile_data is not None:
            config: CosmicSomaticConfig = st.session_state.profile_data
            phys = _surface_physics(config)
            summary = _engine_summary(config)

            unverified_tag = (
                "<span class='unverified-tag'>UNVERIFIED - Gemini's best estimate</span>"
                if summary["unverified"] else ""
            )
            st.markdown(
                f"<h2 class='cosmic-object-name'>{config.object_name}{unverified_tag}</h2>",
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

            st.audio(st.session_state.audio_bytes, format="audio/wav")
            if st.session_state.model_used:
                st.caption(f"Analyzed by {_model_label(st.session_state.model_used)}")
            st.download_button(
                "Download soundscape",
                data=st.session_state.audio_bytes,
                file_name=f"{_safe_filename(config.object_name)}_atmosphere.wav",
                mime="audio/wav",
                use_container_width=True,
            )

            with st.expander("Vector readout"):
                rows = [
                    (name.replace("_", " ").capitalize(), f"{value:.2f}")
                    for name, value in config.vectors.model_dump().items()
                ]
                rows += [
                    ("Grounding tone", f"{config.grounding_tone_hz:.0f} Hz"),
                    ("Engine-derived material", summary["material"]),
                    ("Natural events this place allows", ", ".join(summary["events"])),
                    ("Atmosphere carries sound", "yes" if summary["airborne"] else "no (near-silent, structure-borne only)"),
                ]
                for label, value in rows:
                    st.markdown(
                        f"<div class='vector-row'><span>{label}</span><span>{value}</span></div>",
                        unsafe_allow_html=True,
                    )


if __name__ == "__main__":
    main()
