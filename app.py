"""
Sounds of the Universe
-----------------------
A Universal Psychoacoustic & Emotional Atmosphere Simulator for blind and
visually impaired (BVI) users. Search any astronomical object; the app
resolves its physical characteristics, asks Gemini to translate those
physics into structured psychoacoustic/emotional vectors plus a small
"sonic identity" (root pitch, pulse, dominant material), and then
synthesizes a cinematic, evolving soundscape from them with NumPy/SciPy DSP
so the listener can *feel* what it might be like to stand inside that place.

Failure handling is honest, never generic:
  * The selected Gemini model gets a few retries. Errors are classified from
    the real API response (HTTP code / status), not guessed.
  * Only when the failure is genuinely traffic-related (HTTP 429/5xx,
    timeouts, connection drops) does the app make one real attempt with the
    lighter Gemini 3.5 Flash Lite - and it tells you when that happened and
    why.
  * Any other failure (bad request, missing model, bad key) is shown as-is,
    because retrying or relabelling it as "traffic" would hide the real
    problem. A "Technical details" panel always shows what Gemini returned.
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
            "0.0 to 1.0. High values represent extreme heat/stars. Generates a "
            "dense, buzzing cluster of slightly detuned mid-range waves to "
            "create intense physical and mental agitation."
        )
    )

    # 2. Mental Tension & Threat (Safe/Calm vs. Hostile/Terrifying)
    harmonic_dissonance_index: float = Field(
        description=(
            "0.0 to 1.0. High values represent violent environments (black "
            "holes, supernovae). Introduces sharp, mathematically clashing "
            "intervals (minor seconds/tritones) to trigger physiological and "
            "mental threat/danger."
        )
    )
    rhythmic_unpredictability: float = Field(
        description=(
            "0.0 to 1.0. Controls random, erratic shifts in the timeline (like "
            "solar flares or debris explosions) to keep the listener mentally "
            "on edge."
        )
    )

    # 3. Spatial Scale & Presence (Claustrophobic Enclosure vs. Infinite Vastness)
    claustrophobic_suffocation: float = Field(
        description=(
            "0.0 to 1.0. High values represent dense, high-pressure "
            "atmospheres (like Venus or gas giant cores). Activates a rolling "
            "low-pass filter that swallows all bright frequencies, wrapping "
            "the user in a heavy, muffled mental cage."
        )
    )
    infinite_spatial_drift: float = Field(
        description=(
            "0.0 to 1.0. Controls a slow, deep LFO sweep over a wide stereo "
            "field to make the listener feel completely unanchored, drifting "
            "endlessly through deep space."
        )
    )

    # 4. Fluid Friction (Magnetic Fields, Tearing Winds)
    tearing_friction_index: float = Field(
        description=(
            "0.0 to 1.0. Generates rapid, sweeping phase modulations mimicking "
            "violent solar storms or magnetic shields dragging against vacuum."
        )
    )


SOUND_MATERIALS = ("wind", "plasma", "rumble", "crystal", "metal")


class SonicIdentity(BaseModel):
    """What makes THIS place sound like itself rather than like every other
    place: its tonal centre, its pulse, and the stuff its air is made of."""

    root_frequency_hz: float = Field(
        description=(
            "Fundamental pitch of this environment's drone, 28 to 900 Hz. "
            "Massive, dense, deep or cold places sit very low (28-70 Hz); "
            "medium worlds sit around 80-200 Hz; small, thin, hot or bright "
            "places sit higher (250-900 Hz). Choose from the object's actual "
            "physical scale so different objects get clearly different "
            "tonal centres."
        )
    )
    pulse_rate_hz: float = Field(
        description=(
            "0.0 to 10.0. The rhythmic pulse of the place: a spin, a pressure "
            "throb, an orbital beat, a heartbeat-like thump. 0 means no pulse "
            "at all. Pulsars and fast rotators are high (4-10); crushing "
            "atmospheres are slow and heavy (0.5-1.5); still, empty places "
            "are 0."
        )
    )
    pulse_depth: float = Field(
        description=(
            "0.0 to 1.0. How strongly the pulse thumps and pumps the whole "
            "soundscape. Must be 0 when pulse_rate_hz is 0."
        )
    )
    material: str = Field(
        description=(
            "The dominant substance of this place's sound. Exactly one of: "
            "'wind' (thin gases, drifting clouds, open space), 'plasma' "
            "(stars, flares, ionised gas, crackling energy), 'rumble' "
            "(dense atmospheres, gas giant interiors, deep pressure), "
            "'crystal' (ice, glassy dust, frozen silence, sparkling debris), "
            "'metal' (degenerate matter, magnetars, ringing, resonant, "
            "unearthly places)."
        )
    )


class CosmicSomaticConfig(BaseModel):
    object_name: str
    mental_presence_narrative: str = Field(
        description=(
            "A highly artistic, emotionally vivid 2-sentence description "
            "detailing exactly what it feels like mentally and psychologically "
            "to stand inside this environment."
        )
    )
    vectors: AtmosphereVectors
    identity: SonicIdentity


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


def _sanitize_config(config: CosmicSomaticConfig) -> CosmicSomaticConfig:
    """Gemini occasionally returns 1.2 or an unknown material. Clamp every
    number into its legal range and validate the material so the DSP engine
    can always trust its inputs."""
    v, ident = config.vectors, config.identity
    vectors = AtmosphereVectors(**{
        name: _bounded(getattr(v, name), 0.0, 1.0, 0.0)
        for name in (
            "hollow_isolation_factor", "thermal_manic_hum",
            "harmonic_dissonance_index", "rhythmic_unpredictability",
            "claustrophobic_suffocation", "infinite_spatial_drift",
            "tearing_friction_index",
        )
    })
    material = str(getattr(ident, "material", "wind")).strip().lower()
    if material not in SOUND_MATERIALS:
        material = "wind"
    pulse_rate = _bounded(ident.pulse_rate_hz, 0.0, 10.0, 0.0)
    pulse_depth = _bounded(ident.pulse_depth, 0.0, 1.0, 0.0)
    if pulse_rate < 0.05:
        pulse_rate, pulse_depth = 0.0, 0.0
    identity = SonicIdentity(
        root_frequency_hz=_bounded(ident.root_frequency_hz, 28.0, 900.0, 110.0),
        pulse_rate_hz=pulse_rate,
        pulse_depth=pulse_depth,
        material=material,
    )
    return CosmicSomaticConfig(
        object_name=str(config.object_name).strip() or "Unknown object",
        mental_presence_narrative=str(config.mental_presence_narrative).strip(),
        vectors=vectors,
        identity=identity,
    )


# ---------------------------------------------------------------------------
# 3. DATA INGESTION & GEMINI PROFILER CONNECTOR
# ---------------------------------------------------------------------------
# Every entry carries both descriptive fields (for narrative grounding) and
# continuous 0-1 numeric fields (magnetic_field_strength, atmosphere_density,
# volatility_index, scale_index) so Gemini has concrete, specific numbers to
# reason from instead of vague categories - this is what keeps two different
# objects from being flattened into the same "medium everything" profile.

ASTRO_INDEX = {
    "black hole": {
        "object_type": "black hole",
        "temperature_k": 2.7,
        "mass": "several to billions of solar masses",
        "gravity_g": "effectively infinite at the singularity",
        "atmosphere": "none beyond the event horizon",
        "magnetic_field_strength": 0.7,
        "atmosphere_density": 0.05,
        "volatility_index": 1.0,
        "scale_index": 0.9,
        "notes": (
            "extreme spacetime curvature, silent crushing void at the core, "
            "violent superheated accretion disk screaming at the edge"
        ),
    },
    "neutron star": {
        "object_type": "neutron star",
        "temperature_k": 600000.0,
        "mass": "about 1.4 solar masses compressed into a 20 km sphere",
        "gravity_g": "roughly 200 billion g at the surface",
        "atmosphere": "none, degenerate neutron matter",
        "magnetic_field_strength": 0.95,
        "atmosphere_density": 0.0,
        "volatility_index": 0.65,
        "scale_index": 0.25,
        "notes": "ultra-dense stellar remnant, crushing gravity, rapid rotation",
    },
    "pulsar": {
        "object_type": "pulsar",
        "temperature_k": 1000000.0,
        "mass": "about 1.4 solar masses",
        "gravity_g": "roughly 200 billion g",
        "atmosphere": "none",
        "magnetic_field_strength": 1.0,
        "atmosphere_density": 0.0,
        "volatility_index": 0.55,
        "scale_index": 0.25,
        "notes": (
            "rapidly spinning neutron star sweeping beams of radiation like a "
            "cosmic lighthouse, relentless and metronomic rather than chaotic"
        ),
    },
    "supernova": {
        "object_type": "supernova remnant",
        "temperature_k": 100000.0,
        "mass": "the exploding, disintegrating core of a dying star",
        "gravity_g": "collapsing and rebounding violently",
        "atmosphere": "an expanding shockwave of superheated plasma and debris",
        "magnetic_field_strength": 0.6,
        "atmosphere_density": 0.3,
        "volatility_index": 1.0,
        "scale_index": 0.6,
        "notes": (
            "the explosive death of a star, blinding light, violent shockwave, "
            "chaotic debris flung outward, sudden and catastrophic"
        ),
    },
    "nebula": {
        "object_type": "nebula",
        "temperature_k": 100.0,
        "mass": "thousands of solar masses of gas and dust",
        "gravity_g": "negligible, diffuse",
        "atmosphere": "extremely tenuous gas and dust cloud",
        "magnetic_field_strength": 0.3,
        "atmosphere_density": 0.05,
        "volatility_index": 0.15,
        "scale_index": 0.85,
        "notes": (
            "vast, dark, drifting cloud of gas and dust, cold and hauntingly "
            "empty, a nursery of unborn stars"
        ),
    },
    "venus": {
        "object_type": "terrestrial planet (runaway greenhouse)",
        "temperature_k": 737.0,
        "mass": "0.815 Earth masses",
        "gravity_g": "0.91 g",
        "atmosphere": "96% CO2, crushing dense clouds of sulfuric acid",
        "magnetic_field_strength": 0.05,
        "atmosphere_density": 0.95,
        "volatility_index": 0.2,
        "scale_index": 0.15,
        "notes": "runaway greenhouse effect, thick suffocating atmosphere, hellish pressure",
    },
    "mars": {
        "object_type": "terrestrial planet (cold desert)",
        "temperature_k": 210.0,
        "mass": "0.107 Earth masses",
        "gravity_g": "0.38 g",
        "atmosphere": "thin, mostly CO2, frequent dust storms",
        "magnetic_field_strength": 0.05,
        "atmosphere_density": 0.1,
        "volatility_index": 0.35,
        "scale_index": 0.12,
        "notes": "cold, desolate, rust-red desert world, a thin whistling wind",
    },
    "jupiter": {
        "object_type": "gas giant",
        "temperature_k": 165.0,
        "mass": "318 Earth masses",
        "gravity_g": "2.53 g",
        "atmosphere": "dense hydrogen/helium with storms like the Great Red Spot",
        "magnetic_field_strength": 0.85,
        "atmosphere_density": 0.8,
        "volatility_index": 0.55,
        "scale_index": 0.4,
        "notes": "crushing internal pressure, violent storm bands, an intense magnetosphere",
    },
    "saturn": {
        "object_type": "gas giant (ringed)",
        "temperature_k": 134.0,
        "mass": "95 Earth masses",
        "gravity_g": "1.07 g",
        "atmosphere": "hydrogen/helium with fast winds",
        "magnetic_field_strength": 0.7,
        "atmosphere_density": 0.55,
        "volatility_index": 0.35,
        "scale_index": 0.38,
        "notes": "ringed gas giant, serene and vast from afar, turbulent within",
    },
    "sun": {
        "object_type": "star (main sequence, yellow dwarf)",
        "temperature_k": 5778.0,
        "mass": "about 333,000 Earth masses",
        "gravity_g": "27.9 g at the visible surface",
        "atmosphere": "plasma corona, looping magnetic fields, flares",
        "magnetic_field_strength": 0.75,
        "atmosphere_density": 0.4,
        "volatility_index": 0.6,
        "scale_index": 0.5,
        "notes": "roaring thermonuclear furnace, unpredictable flares and eruptions",
    },
    "star": {
        "object_type": "star",
        "temperature_k": 6000.0,
        "mass": "comparable to, or several times, the Sun",
        "gravity_g": "tens of g at the surface, highly variable",
        "atmosphere": "plasma",
        "magnetic_field_strength": 0.6,
        "atmosphere_density": 0.35,
        "volatility_index": 0.5,
        "scale_index": 0.45,
        "notes": "a nuclear fusion furnace, blinding radiant heat, magnetic turbulence",
    },
    "moon": {
        "object_type": "natural satellite",
        "temperature_k": 220.0,
        "mass": "0.0123 Earth masses",
        "gravity_g": "0.166 g",
        "atmosphere": "virtually none (a trace exosphere)",
        "magnetic_field_strength": 0.02,
        "atmosphere_density": 0.02,
        "volatility_index": 0.05,
        "scale_index": 0.08,
        "notes": (
            "airless and silent, extreme temperature swings between day and "
            "night, vast open plains"
        ),
    },
    "quasar": {
        "object_type": "quasar (active galactic nucleus)",
        "temperature_k": 100000.0,
        "mass": "a host supermassive black hole of millions to billions of solar masses",
        "gravity_g": "extreme near the accretion disk",
        "atmosphere": "superheated accretion plasma",
        "magnetic_field_strength": 0.9,
        "atmosphere_density": 0.2,
        "volatility_index": 1.0,
        "scale_index": 1.0,
        "notes": (
            "one of the most luminous, violent objects in the universe, "
            "blinding radiation, chaotic jets"
        ),
    },
    "white dwarf": {
        "object_type": "white dwarf",
        "temperature_k": 25000.0,
        "mass": "about 0.6 solar masses compressed to Earth-size",
        "gravity_g": "roughly 100,000 g",
        "atmosphere": "none, degenerate matter",
        "magnetic_field_strength": 0.65,
        "atmosphere_density": 0.05,
        "volatility_index": 0.3,
        "scale_index": 0.15,
        "notes": "the collapsed, ultra-dense cooling ember of a dead star",
    },
    "comet": {
        "object_type": "comet",
        "temperature_k": 150.0,
        "mass": "small, often under 10^13 kg",
        "gravity_g": "negligible",
        "atmosphere": "a transient coma of gas and dust when near a star",
        "magnetic_field_strength": 0.05,
        "atmosphere_density": 0.1,
        "volatility_index": 0.4,
        "scale_index": 0.05,
        "notes": "an icy wanderer, a thin trailing tail, fragile and drifting",
    },
    "galaxy": {
        "object_type": "galaxy",
        "temperature_k": 3.0,
        "mass": "billions to trillions of solar masses",
        "gravity_g": "varies enormously across scale",
        "atmosphere": "a near-vacuum interstellar medium",
        "magnetic_field_strength": 0.5,
        "atmosphere_density": 0.0,
        "volatility_index": 0.25,
        "scale_index": 1.0,
        "notes": (
            "a vast spiral or elliptical structure of hundreds of billions of "
            "stars, immense scale, slow rotation over hundreds of millions of "
            "years"
        ),
    },
}


def fetch_astronomical_context(query: str) -> dict:
    """Resolve baseline physical data for an astronomical object.

    Checks a local knowledge index for known keywords (Venus, black hole,
    nebula, star, Mars, etc.) and returns the longest matching entry so more
    specific terms (e.g. 'neutron star') win over generic ones (e.g. 'star').
    Falls back to an open-ended generic template for unknown catalog objects
    so the rest of the pipeline never breaks - Gemini is told explicitly to
    lean on its own knowledge of that specific object in that case, rather
    than trusting the generic placeholder numbers.
    """
    q = (query or "").strip().lower()
    if not q:
        q = "unknown deep space object"

    best_key, best_len = None, 0
    for key in ASTRO_INDEX:
        if key in q and len(key) > best_len:
            best_key, best_len = key, len(key)

    if best_key:
        result = dict(ASTRO_INDEX[best_key])
        result["matched_keyword"] = best_key
        result["is_known"] = True
    else:
        result = {
            "object_type": "unclassified astronomical object",
            "temperature_k": None,
            "mass": "unknown",
            "gravity_g": "unknown",
            "atmosphere": "unknown",
            "magnetic_field_strength": None,
            "atmosphere_density": None,
            "volatility_index": None,
            "scale_index": None,
            "notes": (
                "no local reference data for this specific object - use your own "
                "astronomical knowledge of it if you recognize it by name"
            ),
            "matched_keyword": None,
            "is_known": False,
        }

    result["query"] = query
    return result


SYSTEM_INSTRUCTION = (
    "You are a Cognitive Psychoacoustic Sound Designer working on an "
    "accessibility tool for blind and visually impaired users. Given physical "
    "data about an astronomical object, you decide what it would feel like to "
    "stand inside that environment and encode it as numbers a sound engine "
    "will turn into an immersive soundscape.\n\n"
    "Rules:\n"
    "1. Reason from the object's real physics: temperature, pressure, "
    "gravity, magnetic field, radiation, rotation, scale, emptiness.\n"
    "2. USE THE WHOLE 0.0-1.0 RANGE. Real places are extreme. Most values "
    "should be below 0.25 or above 0.7; do not cluster around 0.4-0.6. Two "
    "physically different objects must end up with clearly different vectors, "
    "root_frequency_hz, pulse and material.\n"
    "3. root_frequency_hz is the place's tonal centre: enormous or crushing "
    "places sit very low, delicate or hot places higher.\n"
    "4. If no local reference data is supplied, rely on your own knowledge of "
    "that specific named object.\n"
    "5. mental_presence_narrative: exactly two vivid sentences, second person, "
    "present tense, built from concrete sensations tied to the physics - "
    "never generic space poetry.\n\n"
    "Calibration anchors. These only show how spread out the scale should be; "
    "never copy them, derive every number from the actual object:\n"
    "- Airless moon: hollow 0.85, thermal 0.03, dissonance 0.05, "
    "unpredictability 0.05, suffocation 0.02, drift 0.35, friction 0.02; "
    "root 90 Hz, pulse 0, material crystal.\n"
    "- Venus surface: hollow 0.05, thermal 0.55, dissonance 0.30, "
    "unpredictability 0.15, suffocation 0.97, drift 0.05, friction 0.25; "
    "root 38 Hz, pulse 0.7 Hz depth 0.8, material rumble.\n"
    "- Edge of a black hole: hollow 0.92, thermal 0.35, dissonance 0.95, "
    "unpredictability 0.80, suffocation 0.35, drift 0.90, friction 0.85; "
    "root 29 Hz, pulse 0.15 Hz depth 0.6, material metal.\n"
    "- Surface of a star: hollow 0.00, thermal 0.98, dissonance 0.60, "
    "unpredictability 0.85, suffocation 0.30, drift 0.20, friction 0.90; "
    "root 180 Hz, pulse 0, material plasma.\n"
    "- Interstellar nebula: hollow 0.90, thermal 0.10, dissonance 0.15, "
    "unpredictability 0.10, suffocation 0.05, drift 0.95, friction 0.20; "
    "root 62 Hz, pulse 0, material wind.\n"
    "Every vector must be a float between 0.0 and 1.0."
)


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


def _parse_gemini_response(response) -> CosmicSomaticConfig:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, CosmicSomaticConfig):
        return parsed
    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini returned an empty response.")
    return CosmicSomaticConfig.model_validate_json(text)


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


def generate_psychoacoustic_schema(query: str, raw_data: dict) -> CosmicSomaticConfig:
    """Asks Gemini to turn the object's physics into a CosmicSomaticConfig.

    The selected model gets its full retries. If (and only if) it failed for
    traffic reasons, one real attempt is made with Gemini 3.5 Flash Lite and
    the reason is recorded for the UI. Non-traffic failures are raised as-is.
    Everything Gemini said on failed attempts is stored in
    st.session_state.gemini_diagnostics. Never returns a fabricated result."""
    model_name = st.session_state.get("gemini_model", GEMINI_FLASH_MODEL)
    client = _get_gemini_client()

    if client is None:
        raise GeminiUnavailableError(
            "No Gemini API key is configured, so a real soundscape can't be "
            "generated. Add GEMINI_API_KEY to .streamlit/secrets.toml or your "
            "environment (and make sure google-genai is installed).",
            reason="not_configured",
        )

    context_note = (
        "Use the physical data below as the primary grounding for your answer."
        if raw_data.get("is_known")
        else (
            "No local reference data exists for this object - rely on your own "
            "astronomical knowledge of it, and estimate only where genuinely "
            "uncertain."
        )
    )
    prompt = (
        f"Astronomical object: {query}\n"
        f"{context_note}\n"
        f"Known physical data: {json.dumps(raw_data, default=str)}\n\n"
        "Design the psychoacoustic profile for standing inside this "
        "environment. Make it unmistakably different from objects of a "
        "different physical character."
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
        return _sanitize_config(config)

    st.session_state.model_used = model_name
    return _sanitize_config(config)


# ---------------------------------------------------------------------------
# 4. THE DYNAMIC PSYCHOACOUSTIC AUDIO ENGINE (NUMPY/SCIPY DSP)
# ---------------------------------------------------------------------------
# Design: the seven vectors decide HOW MUCH of each behaviour a place has;
# the sonic identity decides WHAT the place is made of (its tonal centre, its
# pulse, its material). Because the ingredients themselves differ - not just
# their volume - a crushing greenhouse world, a ringing neutron star, a
# crackling sun and an empty ice moon are different sounds, not one recipe
# at different mix levels.

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


# ---- material beds: the substance the air of this place is made of -------

def _bed_wind(n, sr, rng):
    dur = n / sr
    nb = max(8, int(dur * 2.5))
    pink = _colored_noise(n, rng, 0.5)
    a = _sweeping_bandpass_centers(pink, sr, _sweep_curve(nb, dur, 0.061, rng.uniform(0, TWO_PI), 200, 1100), 300)
    b = _sweeping_bandpass_centers(pink, sr, _sweep_curve(nb, dur, 0.097, rng.uniform(0, TWO_PI), 700, 2600), 500)
    low = _lowpass(pink, sr, 400)
    return _rms_norm(_rms_norm(a, 1.0) * 0.9 + _rms_norm(b, 1.0) * 0.55 + _rms_norm(low, 1.0) * 0.3, 1.0)


def _bed_plasma(n, sr, rng):
    hiss = _highpass(rng.normal(0.0, 1.0, n), sr, 1800)
    flutter = _lowpass(rng.normal(0.0, 1.0, n), sr, 30)
    flutter = np.clip(flutter / (np.std(flutter) + 1e-12), -1.5, 1.5) / 1.5
    sizzle = hiss * np.clip(1.0 + 0.7 * flutter, 0.1, None)
    impulses = (rng.random(n) < 55.0 / sr) * rng.choice([-1.0, 1.0], n) * rng.uniform(0.3, 1.0, n)
    tk = np.arange(int(0.008 * sr)) / sr
    crackle = fftconvolve(impulses, np.exp(-tk / 0.0015))[:n]
    body = _lowpass(_colored_noise(n, rng, 1.0), sr, 500)
    return _rms_norm(_rms_norm(sizzle, 1.0) * 0.55 + _rms_norm(crackle, 1.0) * 0.8
                     + _rms_norm(body, 1.0) * 0.35, 1.0)


def _bed_rumble(n, sr, rng):
    """Dense-pressure texture that stays articulate: a contained bass body
    (45-260 Hz, not sub-boom), a gritty mid growl and fine grain on top."""
    t = np.arange(n) / sr
    body = _highpass(_lowpass(_colored_noise(n, rng, 1.0), sr, 260, passes=2), sr, 45)
    heave = 1.0 + 0.3 * np.sin(TWO_PI * 0.13 * t + rng.uniform(0, TWO_PI))
    pink = _colored_noise(n, rng, 0.5)
    growl = sosfilt(_butter_sos((200.0, 900.0), sr, "bandpass", 2), pink) * (
        0.55 + 0.45 * np.sin(TWO_PI * 0.21 * t + rng.uniform(0, TWO_PI)))
    flicker = np.clip(_lowpass(rng.normal(0.0, 1.0, n), sr, 8) * 3.0, 0.0, None)
    grain = _highpass(pink, sr, 1500) * flicker
    return _rms_norm(0.5 * _rms_norm(body * heave, 1.0) + 0.9 * _rms_norm(growl, 1.0)
                     + 0.35 * _rms_norm(grain, 1.0), 1.0)


def _bed_crystal(n, sr, rng):
    t = np.arange(n) / sr
    out = np.zeros(n)
    partials = rng.uniform(1300.0, 5400.0, 7)
    for f in partials:
        sparkle = np.clip(np.sin(TWO_PI * rng.uniform(0.05, 0.35) * t + rng.uniform(0, TWO_PI)), 0.0, None) ** 2
        out += sparkle * np.sin(TWO_PI * f * t + rng.uniform(0, TWO_PI))
    air = _highpass(_colored_noise(n, rng, 0.5), sr, 3000)
    return _rms_norm(_rms_norm(out, 1.0) + 0.3 * _rms_norm(air, 1.0), 1.0)


def _bed_metal(n, sr, rng, root):
    pink = _colored_noise(n, rng, 0.5)
    f0 = _fold_freq(root, 90.0, 300.0)
    out = np.zeros(n)
    for k, ratio in enumerate((1.0, 2.756, 5.404, 8.933, 13.344)):
        f = f0 * ratio
        if f > 15000.0:
            break
        bw = max(f / 200.0, 1.5)
        band = sosfilt(_butter_sos((f - bw / 2, f + bw / 2), sr, "bandpass", 2), pink)
        out += _rms_norm(band, 1.0) / (k + 1) ** 0.8
    return _rms_norm(out, 1.0)


def _material_bed(material, n, sr, rng, root):
    if material == "plasma":
        return _bed_plasma(n, sr, rng)
    if material == "rumble":
        return _bed_rumble(n, sr, rng)
    if material == "crystal":
        return _bed_crystal(n, sr, rng)
    if material == "metal":
        return _bed_metal(n, sr, rng, root)
    return _bed_wind(n, sr, rng)


# ---- tonal identity ------------------------------------------------------

def _root_drone(t, sr, rng, root, thermal, dissonance):
    """The place's tonal centre: a harmonic stack that gets brighter and more
    chorused with heat, sweetened with a fifth/octave when calm and clashed
    with a tritone/minor second when hostile.

    For very low roots the fundamental is turned down and the interval
    partials are lifted by octaves: the ear still hears the low pitch from
    the harmonics (the "missing fundamental" effect), but the mix stops
    piling energy into the sub-bass where it turns to mud."""
    vibrato = 1.0 + 0.003 * np.sin(TWO_PI * 0.17 * t + rng.uniform(0, TWO_PI))
    phase = TWO_PI * np.cumsum(root * vibrato) / sr
    n_max = int(np.clip(2200.0 / root, 6, 24))
    n_harm = int(4 + (n_max - 4) * (0.3 + 0.7 * thermal))
    slope = 1.1 - 0.5 * thermal
    spread = 0.0035 + 0.02 * thermal
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
    """Layer 2 - the manic star core: three mid-range oscillators (near
    220 / 221.5 / 223 Hz, octave-folded from the place's root so it stays
    in tune with it), detuned further by the dissonance index, given richer
    harmonics and a faster tremor as things get hotter."""
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


# ---- pulse ----------------------------------------------------------------

def _pulse_layers(n, sr, rng, rate, depth, suffocation, unpredictability, root):
    """A rhythmic thump train plus a "pumping" gain curve that ducks the whole
    body on every beat. Returns (thump, duck)."""
    if rate < 0.05 or depth <= 0.02:
        return np.zeros(n), np.ones(n)
    period = sr / rate
    count = int(n / period) + 1
    jitter = rng.normal(0.0, 0.06 * period * unpredictability, count)
    idx = (np.arange(count) * period + jitter).astype(int)
    idx = idx[(idx >= 0) & (idx < n)]
    train = np.zeros(n)
    train[idx] = rng.uniform(0.75, 1.0, len(idx))
    tk = np.arange(int(0.7 * sr)) / sr
    fp = _fold_freq(root, 55.0, 100.0)
    decay = 16.0 - 8.0 * suffocation
    drop = 0.6 * fp / 30.0
    kernel = np.sin(TWO_PI * (fp * tk + drop * (1.0 - np.exp(-30.0 * tk)))) * np.exp(-decay * tk)
    thump = fftconvolve(train, kernel)[:n]
    period_s = 1.0 / rate
    ek = np.arange(int(min(0.6, period_s * 1.5) * sr)) / sr
    env = np.clip(fftconvolve(train, np.exp(-4.0 * ek / period_s))[:n], 0.0, 1.0)
    return thump, 1.0 - 0.6 * depth * env


# ---- discrete events (unpredictability) -----------------------------------

def _event_sound(material, sr, rng, root):
    if material == "plasma":
        if rng.random() < 0.3:
            length = int(rng.uniform(0.8, 1.5) * sr)
            tau = np.arange(length) / length
            return np.diff(rng.normal(0, 1, length + 1)) * np.sin(np.pi * tau) ** 2 * 0.8
        length = int(0.25 * sr)
        tk = np.arange(length) / sr
        return np.diff(rng.normal(0, 1, length + 1)) * np.exp(-tk * rng.uniform(25, 60))
    if material == "metal":
        f = _fold_freq(root * rng.choice([2.0, 3.0, 4.76, 6.7]), 300.0, 3500.0)
        tk = np.arange(int(1.2 * sr)) / sr
        return np.sin(TWO_PI * f * tk) * np.exp(-tk * rng.uniform(2.5, 6.0))
    if material == "rumble":
        tk = np.arange(int(1.0 * sr)) / sr
        f = _fold_freq(root, 35.0, 70.0)
        return np.sin(TWO_PI * f * tk) * np.exp(-tk * rng.uniform(5.0, 9.0)) * 1.2
    if material == "crystal":
        tk = np.arange(int(1.6 * sr)) / sr
        f = rng.uniform(1400.0, 4200.0)
        return (np.sin(TWO_PI * f * tk) + 0.4 * np.sin(TWO_PI * f * 2.76 * tk)) * np.exp(-tk * rng.uniform(1.5, 3.5))
    length = int(rng.uniform(1.0, 2.2) * sr)
    tau = np.arange(length) / length
    noise = rng.normal(0, 1, length)
    center = rng.uniform(500.0, 2500.0)
    band = sosfilt(_butter_sos((center - 300, center + 300), sr, "bandpass", 2), noise)
    return band * np.sin(np.pi * tau) ** 2 * 2.5


def _events(n, sr, rng, material, unpredictability, root):
    left, right = np.zeros(n), np.zeros(n)
    if unpredictability <= 0.02:
        return left, right
    count = int(round((n / sr) * 1.8 * unpredictability ** 1.3))
    for _ in range(count):
        pos = int(rng.uniform(0.02, 0.98) * n)
        amp = rng.uniform(0.35, 1.0) * (0.4 + 0.6 * unpredictability)
        pan = rng.uniform(-1.0, 1.0)
        seg = _event_sound(material, sr, rng, root)
        seg = seg / (np.max(np.abs(seg)) + 1e-9)
        end = min(n, pos + len(seg))
        seg = seg[: end - pos]
        angle = (pan + 1.0) * (np.pi / 4.0)
        left[pos:end] += seg * amp * np.cos(angle)
        right[pos:end] += seg * amp * np.sin(angle)
    return left, right


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


def _distant_calls(n, sr, rng, root, hollow, dissonance):
    """Sparse, swelling tones that call across emptiness and come back as
    long echoes - the audible face of hollow_isolation_factor."""
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


# ---- friction / drift / space ---------------------------------------------

def _shear_layer(n, sr, rng, t, friction):
    """Metallic grinding: swept narrow noise, amplitude-modulated at a grind
    rate that climbs with friction."""
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


# Each material gets a stable timbral colour on top of its own source sound, so
# even places with similar vectors keep a recognisable character:
# (centre Hz, gain dB, Q) peaking filters.
_MATERIAL_EQ = {
    "wind": [],
    "plasma": [(3500.0, 5.0, 0.6)],                       # hot, bright, crackling
    "rumble": [(3500.0, -5.0, 0.6), (900.0, 2.0, 0.8)],   # dense, dark, pressed
    "crystal": [(300.0, -5.0, 0.7), (5000.0, 4.0, 0.7)],  # thin, glassy, cold
    "metal": [(1200.0, 5.0, 1.5)],                        # ringing, resonant
}


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


def synthesize_psychoacoustic_audio(config: CosmicSomaticConfig, duration: float = 10.0,
                                     sample_rate: int = 44100) -> bytes:
    cfg = _sanitize_config(config)
    v, ident = cfg.vectors, cfg.identity
    hollow, thermal = v.hollow_isolation_factor, v.thermal_manic_hum
    dissonance, unpredictability = v.harmonic_dissonance_index, v.rhythmic_unpredictability
    suffocation, drift, friction = v.claustrophobic_suffocation, v.infinite_spatial_drift, v.tearing_friction_index
    root, material = ident.root_frequency_hz, ident.material

    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n, endpoint=False)
    rng = _seeded_rng(cfg.object_name)
    sr = sample_rate

    # Slow "weather": the whole place breathes over time so it never sits still.
    evolve = 0.88 + 0.12 * np.sin(TWO_PI * 0.055 * t + rng.uniform(0, TWO_PI))

    # Material bed, generated independently per ear for natural width.
    bed_l = _material_bed(material, n, sr, rng, root)
    bed_r = _material_bed(material, n, sr, rng, root)
    width = 0.3 + 0.7 * drift
    mid, side = (bed_l + bed_r) / 2.0, (bed_l - bed_r) / 2.0 * width
    bed_l, bed_r = mid + side, mid - side

    # Tonal identity + layer 2 (manic star core) with 0.05 Hz drifting pans.
    drone = _root_drone(t, sr, rng, root, thermal, dissonance)
    cluster = _thermal_cluster(t, root, thermal, dissonance) * (
        1.0 + 0.25 * np.sin(TWO_PI * 0.09 * t + rng.uniform(0, TWO_PI)))
    dl, dr = _autopan(drone, t, 0.15 + 0.85 * drift, 0.05, rng.uniform(0, TWO_PI))
    cl, cr = _autopan(cluster, t, 0.15 + 0.85 * drift, 0.031, rng.uniform(0, TWO_PI))

    g_bed, g_drone = 0.15, 0.15
    g_cluster = 0.20 * thermal ** 1.1
    body_l = (bed_l * g_bed + dl * g_drone + cl * g_cluster) * evolve
    body_r = (bed_r * g_bed + dr * g_drone + cr * g_cluster) * evolve

    # Layer 1 - the cold, isolating void (swept 120-1500 Hz + structured reverb).
    if hollow > 0.02:
        void_l, void_r = _void_layer(n, sr, rng, hollow)
        body_l, body_r = body_l + void_l * 0.15 * hollow ** 0.9, body_r + void_r * 0.15 * hollow ** 0.9

    # Friction: grinding shear texture (amplitude follows friction).
    if friction > 0.02:
        shear = _shear_layer(n, sr, rng, t, friction) * 0.13 * friction
        sl, sr_ = _autopan(shear, t, 0.5 + 0.5 * drift, 0.07, rng.uniform(0, TWO_PI))
        body_l, body_r = body_l + sl, body_r + sr_

    # Unpredictability: discrete flares, cracks, pings, gusts, thuds.
    ev_l, ev_r = _events(n, sr, rng, material, unpredictability, root)
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

    # Pulse: pumping duck on the body (the thump itself is added later).
    thump, duck = _pulse_layers(n, sr, rng, ident.pulse_rate_hz, ident.pulse_depth,
                                suffocation, unpredictability, root)
    body_l, body_r = body_l * duck, body_r * duck

    # Layer 3 - the claustrophobic choke: recursive low-pass swallows treble.
    if suffocation > 0.02:
        cutoff = max(600.0, 9000.0 - 6000.0 * suffocation)
        wet_l = _recursive_lowpass(body_l, sr, cutoff, passes=2)
        wet_r = _recursive_lowpass(body_r, sr, cutoff, passes=2)
        thick = _rms_norm(_lowpass(_colored_noise(n, rng, 1.0), sr, 400), 0.03 * suffocation)
        body_l = (1.0 - suffocation) * body_l + suffocation * wet_l + thick
        body_r = (1.0 - suffocation) * body_r + suffocation * wet_r + thick

    # High-frequency air/pressure texture, present while the place is open enough
    # to let it through; added after the choke so its own cutoff is preserved.
    if suffocation < 0.8:
        openness = 1.0 - suffocation
        air_gain = 0.03 + 0.05 * openness
        body_l = body_l + _air_layer(n, sr, rng, t, suffocation) * air_gain
        body_r = body_r + _air_layer(n, sr, rng, t, suffocation) * air_gain

    # Sub-bass presence kept deliberately small, plus the pressure thump.
    sub = _sub_drone(t, root, rng) * (0.01 + 0.05 * hollow)
    left = body_l + sub + thump * 0.22 * ident.pulse_depth
    right = body_r + sub + thump * 0.22 * ident.pulse_depth

    # Layer 4 - phase-modulation sweeps for magnetic storm fields.
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

    # Clarity EQ: remove sub-rumble and a touch of low-mid mud.
    stereo = sosfilt(_butter_sos(40.0, sr, "high", 2), stereo, axis=0)
    stereo = sosfilt(_peaking_sos(sr, 240.0, -3.5, 0.8), stereo, axis=0)
    for f0, gain_db, q in _MATERIAL_EQ.get(material, []):
        stereo = sosfilt(_peaking_sos(sr, f0, gain_db, q), stereo, axis=0)

    # Master: soft transient compression stops flares from shrinking the body,
    # then peak normalisation and tanh saturation. Loudness and drive follow
    # overall intensity, so calm places stay clean and violent ones push hard.
    intensity = _clamp01((thermal + dissonance + unpredictability + friction) / 4.0)
    knee = 4.0 * _rms(stereo)
    stereo = np.tanh(stereo / knee) * knee
    target_peak = 0.45 + 0.45 * intensity
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

# Suggestions for the search box. It filters live from the first character
# typed, and (where Streamlit supports it) still accepts ANY object name.
SUGGESTION_LIST = sorted(set([key.title() for key in ASTRO_INDEX.keys()] + [
    "Mercury", "Neptune", "Uranus", "Pluto", "Europa", "Titan", "Io", "Ganymede",
    "Andromeda Galaxy", "Milky Way", "Crab Nebula", "Orion Nebula", "Sagittarius A*",
    "Alpha Centauri", "Betelgeuse", "Sirius", "Proxima Centauri", "Halley's Comet",
    "the Kuiper Belt", "the Asteroid Belt", "TRAPPIST-1e", "Kepler-452b", "the Oort Cloud",
]))

SEARCH_PLACEHOLDER = "Try: Venus, a black hole, the Crab Nebula, Europa..."


def _safe_filename(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "atmosphere"


def _render_search_box() -> Optional[str]:
    """Searchable dropdown with live suggestions from the first letter typed
    (fuzzy matching on Streamlit >= 1.56). Degrades gracefully on older
    Streamlit versions instead of crashing."""
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


def _run_search(clean_query: str) -> None:
    st.session_state.search_error = None
    st.session_state.fallback_notice = None
    st.session_state.gemini_diagnostics = []
    st.session_state.model_used = None
    with st.spinner(f"Listening to {clean_query}..."):
        try:
            raw_data = fetch_astronomical_context(clean_query)
            config = generate_psychoacoustic_schema(clean_query, raw_data)
            audio_bytes = synthesize_psychoacoustic_audio(config, duration=SOUNDSCAPE_SECONDS)
        except GeminiUnavailableError as e:
            st.session_state.search_error = str(e)
            if e.details and not st.session_state.gemini_diagnostics:
                st.session_state.gemini_diagnostics = list(e.details)
        else:
            st.session_state.selected_query = clean_query
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
        "<p class='cosmic-subtitle'>Search any astronomical object and hear what it might "
        "feel like to stand inside it.</p>",
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
            help="Choose which Gemini model analyzes the object's physics.",
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

        if st.session_state.gemini_diagnostics:
            with st.expander("Technical details (what Gemini actually returned)"):
                st.code("\n".join(st.session_state.gemini_diagnostics), language=None)

        if st.session_state.profile_data is not None:
            config: CosmicSomaticConfig = st.session_state.profile_data

            st.markdown(f"<h2 class='cosmic-object-name'>{config.object_name}</h2>", unsafe_allow_html=True)
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
                ident = config.identity
                rows += [
                    ("Root frequency", f"{ident.root_frequency_hz:.0f} Hz"),
                    ("Pulse", f"{ident.pulse_rate_hz:.2f} Hz, depth {ident.pulse_depth:.2f}"),
                    ("Material", ident.material),
                ]
                for label, value in rows:
                    st.markdown(
                        f"<div class='vector-row'><span>{label}</span><span>{value}</span></div>",
                        unsafe_allow_html=True,
                    )


if __name__ == "__main__":
    main()
