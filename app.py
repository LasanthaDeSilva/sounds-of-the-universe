"""
Sounds of the Universe
-----------------------
A Universal Psychoacoustic & Emotional Atmosphere Simulator for blind and
visually impaired (BVI) users. Search any astronomical object; the app
resolves its physical characteristics, asks Gemini to translate those
physics into a structured set of psychoacoustic/emotional vectors, and then
synthesizes a unique cinematic soundscape from those vectors with NumPy/SciPy
DSP so the listener can *feel* what it might be like to stand inside that
environment.

If Gemini cannot produce a real, grounded answer (no key configured, a
transient outage, rate limiting), the app does NOT fall back to a generic
canned sound. It retries briefly, and if it still can't get a real answer it
says so plainly instead of pretending.

Requirements:
    pip install streamlit numpy scipy pydantic google-genai

Configuration (optional, all have safe fallbacks):
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
import os
import re
import time
import wave
from typing import Optional

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from pydantic import BaseModel, Field
from scipy.signal import butter, fftconvolve, sosfilt

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - the app still runs without the SDK installed
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

GEMINI_MAX_ATTEMPTS = 3
GEMINI_RETRY_BACKOFF_SECONDS = 1.4


def init_session_state() -> None:
    defaults = {
        "selected_query": None,
        "profile_data": None,
        "audio_bytes": None,
        "gemini_model": GEMINI_FLASH_MODEL,
        "search_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


class GeminiUnavailableError(Exception):
    """Raised when Gemini cannot produce a real psychoacoustic profile."""

    def __init__(self, message: str, reason: str = "error"):
        super().__init__(message)
        self.reason = reason  # "not_configured" or "error"


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
    "accessibility tool for blind and visually impaired users. Given raw "
    "physical data about an astronomical object, translate its environmental "
    "physics into precise psychoacoustic and emotional vectors that a sound "
    "engine will use to synthesize an immersive soundscape. Be scientifically "
    "grounded in the physical data provided, but artistically vivid in the "
    "narrative. Use the full 0.0-1.0 range decisively - do not default to "
    "safe, moderate values for every object. Two physically different objects "
    "(for example a frozen, airless moon and an erupting star) should almost "
    "never receive similar vectors; two physically similar objects may. If no "
    "local reference data is available, rely on your own knowledge of that "
    "specific named object rather than inventing generic numbers. Every "
    "vector must be a float between 0.0 and 1.0."
)


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _get_gemini_client():
    if genai is None:
        return None
    api_key = _secret("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def generate_psychoacoustic_schema(query: str, raw_data: dict) -> CosmicSomaticConfig:
    """Connects to Gemini to map raw physical data onto the CosmicSomaticConfig
    schema. Retries briefly on transient failures. If Gemini genuinely cannot
    produce a real answer, this raises GeminiUnavailableError rather than
    silently substituting a generic, one-size-fits-all sound."""
    model_name = st.session_state.get("gemini_model", GEMINI_FLASH_MODEL)
    client = _get_gemini_client()

    if client is None:
        raise GeminiUnavailableError(
            "No Gemini API key is configured, so a real soundscape can't be generated right now.",
            reason="not_configured",
        )

    context_note = (
        "Use the physical data below as the primary grounding for your answer."
        if raw_data.get("is_known")
        else (
            "Local reference data is unavailable for this specific object - rely "
            "on your own astronomical knowledge of it, and make physically "
            "reasonable estimates only where genuinely uncertain."
        )
    )
    prompt = (
        f"Astronomical object: {query}\n"
        f"{context_note}\n"
        f"Known physical data: {json.dumps(raw_data, default=str)}\n\n"
        "Analyze the physics above and produce a CosmicSomaticConfig capturing "
        "what it would feel like, psychologically and mentally, to stand "
        "inside this environment. Ground every vector in the physical data, "
        "and make sure the result is clearly distinguishable from objects of "
        "a different physical character."
    )

    last_exc: Optional[Exception] = None
    for attempt in range(GEMINI_MAX_ATTEMPTS):
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
            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, CosmicSomaticConfig):
                return parsed
            text = getattr(response, "text", None)
            if text:
                return CosmicSomaticConfig.model_validate_json(text)
            last_exc = ValueError("Gemini returned an empty response.")
        except Exception as exc:
            last_exc = exc
        if attempt < GEMINI_MAX_ATTEMPTS - 1:
            time.sleep(GEMINI_RETRY_BACKOFF_SECONDS * (attempt + 1))

    raise GeminiUnavailableError(
        "Too much traffic right now - please try again in a moment.",
        reason="error",
    ) from last_exc


# ---------------------------------------------------------------------------
# 4. THE DYNAMIC PSYCHOACOUSTIC AUDIO ENGINE (NUMPY/SCIPY DSP)
# ---------------------------------------------------------------------------

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


def _sweeping_bandpass(signal: np.ndarray, sample_rate: int, low_start: float,
                        low_end: float, bandwidth: float, num_blocks: int = 36) -> np.ndarray:
    """A resonant bandpass whose center frequency sweeps slowly over time,
    implemented as overlap-add blocks so the sweep is smooth and click-free."""
    n = len(signal)
    block = max(1, n // num_blocks)
    fade = min(400, block // 4)
    out = np.zeros(n)
    win_sum = np.zeros(n)
    pos = 0
    while pos < n:
        end = min(n, pos + block + fade)
        seg = signal[pos:end]
        if len(seg) == 0:
            break
        frac = min(1.0, pos / max(1, n))
        center = low_start + (low_end - low_start) * frac
        lo = max(20.0, center - bandwidth / 2)
        hi = min(sample_rate / 2 - 50, center + bandwidth / 2)
        if hi <= lo:
            hi = lo + 10.0
        sos = _butter_sos((lo, hi), sample_rate, "bandpass", order=2)
        filtered = sosfilt(sos, seg)
        w = np.ones(len(seg))
        if fade > 0 and len(seg) > 2 * fade:
            ramp = np.linspace(0.0, 1.0, fade)
            w[:fade] = ramp
            w[-fade:] = ramp[::-1]
        out[pos:end] += filtered * w
        win_sum[pos:end] += w
        pos += block
    win_sum[win_sum == 0] = 1.0
    return out / win_sum


def _recursive_lowpass(signal: np.ndarray, sample_rate: int, cutoff_hz: float, passes: int = 3) -> np.ndarray:
    """A digital filter loop: the same low-pass filter applied recursively to
    progressively strip away brightness, simulating a heavy, muffled cage."""
    sos = _butter_sos(cutoff_hz, sample_rate, "low", order=2)
    y = signal
    for _ in range(passes):
        y = sosfilt(sos, y)
    return y


def _dissonance_stabs(n: int, sample_rate: int, rng: np.random.Generator,
                       base_freq: float, amount: float) -> np.ndarray:
    """Sharp, short-lived minor-second / tritone bursts layered on top of the
    mix, density and loudness scaled by amount."""
    if amount <= 0:
        return np.zeros(n)
    t_full = np.arange(n) / sample_rate
    out = np.zeros(n)
    intervals = (16 / 15, 2 ** 0.5)  # minor second, tritone
    num_stabs = int(2 + amount * 10)
    for _ in range(num_stabs):
        ratio = intervals[rng.integers(0, len(intervals))]
        freq = base_freq * ratio * rng.uniform(0.5, 2.0)
        pos = int(rng.integers(0, n))
        dur = int(rng.uniform(0.05, 0.25) * sample_rate)
        end = min(n, pos + dur)
        if end <= pos:
            continue
        seg_t = t_full[pos:end] - t_full[pos]
        local_env = np.exp(-seg_t * 12.0)
        out[pos:end] += amount * local_env * np.sin(2 * np.pi * freq * seg_t)
    return out


def _rhythmic_gate(n: int, sample_rate: int, rng: np.random.Generator, amount: float) -> np.ndarray:
    """Random, erratic amplitude dips across the timeline, simulating solar
    flares or debris explosions punching through the soundscape."""
    if amount <= 0:
        return np.ones(n)
    gate = np.ones(n)
    num_events = int(3 + amount * 20)
    for _ in range(num_events):
        pos = int(rng.integers(0, n))
        length = int(rng.uniform(0.01, 0.08) * sample_rate)
        depth = rng.uniform(0.2, 0.9) * amount
        end = min(n, pos + length)
        gate[pos:end] *= (1.0 - depth)
    kernel_len = min(200, n)
    kernel = np.ones(kernel_len) / kernel_len
    return np.convolve(gate, kernel, mode="same")


def _phase_tearing(signal: np.ndarray, sample_rate: int, rate_hz: float,
                    depth_ms: float, mix: float) -> np.ndarray:
    """Rapid, sweeping phase modulation via a modulated fractional delay line,
    mimicking magnetic fields tearing against vacuum."""
    n = len(signal)
    t = np.arange(n) / sample_rate
    depth_samples = (depth_ms / 1000.0) * sample_rate
    lfo = (np.sin(2 * np.pi * rate_hz * t) + 1.0) / 2.0
    delay_samples = lfo * depth_samples
    src_idx = np.clip(np.arange(n) - delay_samples, 0, n - 1)
    delayed = np.interp(src_idx, np.arange(n), signal)
    return (1.0 - mix) * signal + mix * delayed


def _apply_stereo_drift(mono: np.ndarray, sample_rate: int, drift_amount: float,
                         rate_hz: float = 0.05) -> np.ndarray:
    """Slow equal-power stereo panning LFO, from centered (drift=0) to a wide,
    unanchored sweep across the stereo field (drift=1)."""
    n = len(mono)
    t = np.arange(n) / sample_rate
    pan = np.clip(np.sin(2 * np.pi * rate_hz * t) * drift_amount, -1.0, 1.0)
    angle = (pan + 1.0) * (np.pi / 4.0)
    left = mono * np.cos(angle)
    right = mono * np.sin(angle)
    return np.stack([left, right], axis=-1)


def _fade_envelope(n: int, sample_rate: int, attack_s: float = 0.6, release_s: float = 1.2) -> np.ndarray:
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
    v = config.vectors
    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n, endpoint=False)
    rng = _seeded_rng(config.object_name)

    mix = np.zeros(n)

    # Layer 1 - The Cold, Isolating Void
    if v.hollow_isolation_factor > 0:
        raw_noise = rng.normal(0.0, 1.0, n)
        swept = _sweeping_bandpass(raw_noise, sample_rate, low_start=120.0, low_end=900.0, bandwidth=250.0)
        tail_len = min(n, int(sample_rate * 2.5))
        decay = np.exp(-np.linspace(0.0, 6.0, tail_len))
        impulse = rng.normal(0.0, 1.0, tail_len) * decay
        impulse = impulse / (np.sum(np.abs(impulse)) + 1e-9)
        reverbed = fftconvolve(swept, impulse, mode="full")[:n]
        void_layer = 0.55 * swept + 0.85 * reverbed
        peak = np.max(np.abs(void_layer)) + 1e-9
        void_layer = void_layer / peak
        mix += v.hollow_isolation_factor * 0.5 * void_layer

    # Layer 2 - The Manic Star Core
    if v.thermal_manic_hum > 0:
        base = 220.0
        detune = v.harmonic_dissonance_index
        f1 = base
        f2 = base * (1.0 + 0.007 + 0.05 * detune)
        f3 = base * (1.0 + 0.014 + 0.09 * detune)
        osc = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t) + np.sin(2 * np.pi * f3 * t)
        osc = osc / 3.0
        stabs = _dissonance_stabs(n, sample_rate, rng, base_freq=base * 2, amount=v.harmonic_dissonance_index)
        mix += v.thermal_manic_hum * 0.45 * (osc + stabs)

    # Rhythmic unpredictability applies across whatever has been layered so far
    gate = _rhythmic_gate(n, sample_rate, rng, v.rhythmic_unpredictability)
    mix = mix * gate

    # Layer 3 - The Claustrophobic Choke
    if v.claustrophobic_suffocation > 0:
        cutoff = max(150.0, 9000.0 - 8200.0 * v.claustrophobic_suffocation)
        choked = _recursive_lowpass(mix, sample_rate, cutoff, passes=3)
        rumble = 0.15 * v.claustrophobic_suffocation * np.sin(2 * np.pi * 45.0 * t)
        mix = (1.0 - v.claustrophobic_suffocation) * mix + v.claustrophobic_suffocation * choked + rumble

    # Layer 4 - Infinite Drift & Friction
    if v.tearing_friction_index > 0:
        mix = _phase_tearing(
            mix, sample_rate,
            rate_hz=4.0 + 6.0 * v.tearing_friction_index,
            depth_ms=2.0 + 12.0 * v.tearing_friction_index,
            mix=min(0.85, v.tearing_friction_index),
        )

    stereo = _apply_stereo_drift(mix, sample_rate, drift_amount=v.infinite_spatial_drift, rate_hz=0.05)

    env = _fade_envelope(n, sample_rate)
    stereo = stereo * env[:, None]

    # Master loudness/drive scales with overall "intensity" so a calm moon and
    # an erupting black hole are audibly, not just spectrally, different -
    # violent objects come out louder and more harmonically driven.
    intensity = _clamp01(
        (v.thermal_manic_hum + v.harmonic_dissonance_index
         + v.rhythmic_unpredictability + v.tearing_friction_index) / 4.0
    )
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
div[data-testid="stFormSubmitButton"] button, div.stButton button {
    font-family: 'Space Grotesk', sans-serif;
    background: #ff6a4d;
    color: #07060d;
    border: none;
    border-radius: 4px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    font-size: 1.05rem;
    transition: transform 0.15s ease, background 0.15s ease;
}
div[data-testid="stFormSubmitButton"] button:hover, div.stButton button:hover {
    transform: translateY(-1px);
    background: #ff8266;
    color: #07060d;
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

# A Google-style search bar: a pill-shaped input with instant, client-side
# (zero-latency) autosuggestions filtered from the first letter typed, plus a
# "Search" button - built as a small self-contained HTML/JS widget since a
# native Streamlit text_input only reruns on Enter/blur, not per keystroke.
# Submitting (click, Enter, or picking a suggestion) navigates the page with
# a ?q=... query parameter, which Streamlit reads back out below.
SEARCH_WIDGET_TEMPLATE = """
<div style="font-family:'Space Grotesk',sans-serif;">
  <style>
    #cosmic-search-wrap { position:relative; max-width:640px; margin:0 auto; }
    #cosmic-search-input {
        width:100%; box-sizing:border-box; font-size:1.25rem;
        padding:0.95rem 1.4rem; background:#12111f; color:#f1eef7;
        border:2px solid #2a2740; border-radius:28px;
        font-family:'Space Grotesk',sans-serif;
    }
    #cosmic-search-input:focus { outline:none; border-color:#ff6a4d; }
    #cosmic-suggestions {
        position:absolute; top:calc(100% + 6px); left:12px; right:12px;
        background:#12111f; border:1px solid #2a2740; border-radius:14px;
        z-index:10; max-height:220px; overflow-y:auto; display:none;
        box-shadow:0 12px 24px rgba(0,0,0,0.35);
    }
    .cosmic-suggestion-item { padding:0.65rem 1.1rem; color:#cfc9de; cursor:pointer; }
    .cosmic-suggestion-item:hover { background:#1c1a2b; color:#f1eef7; }
    #cosmic-search-btn {
        display:block; margin:1rem auto 0 auto; background:#ff6a4d; color:#07060d;
        border:none; border-radius:20px; padding:0.65rem 1.8rem; font-weight:600;
        font-size:1rem; font-family:'Space Grotesk',sans-serif; cursor:pointer;
    }
    #cosmic-search-btn:hover { background:#ff8266; }
  </style>
  <div id="cosmic-search-wrap">
    <input id="cosmic-search-input" autocomplete="off"
           placeholder="Try: Venus, a black hole, the Crab Nebula, Europa..."
           value="__CURRENT_VALUE__">
    <div id="cosmic-suggestions"></div>
    <button id="cosmic-search-btn" type="button">Search</button>
  </div>
</div>
<script>
  (function () {
    const ALL_SUGGESTIONS = __SUGGESTIONS_JSON__;
    const input = document.getElementById('cosmic-search-input');
    const box = document.getElementById('cosmic-suggestions');
    const btn = document.getElementById('cosmic-search-btn');

    function runSearch(value) {
      const trimmed = (value || '').trim();
      if (!trimmed) return;
      const url = new URL(window.parent.location.href);
      url.searchParams.set('q', trimmed);
      url.searchParams.set('t', Date.now().toString());
      window.parent.location.href = url.toString();
    }

    function renderSuggestions(list) {
      box.innerHTML = '';
      if (list.length === 0) { box.style.display = 'none'; return; }
      list.slice(0, 8).forEach(function (name) {
        const item = document.createElement('div');
        item.className = 'cosmic-suggestion-item';
        item.textContent = name;
        item.addEventListener('mousedown', function (e) {
          e.preventDefault();
          input.value = name;
          box.style.display = 'none';
          runSearch(name);
        });
        box.appendChild(item);
      });
      box.style.display = 'block';
    }

    input.addEventListener('input', function () {
      const val = input.value.trim().toLowerCase();
      if (!val) { box.style.display = 'none'; return; }
      const matches = ALL_SUGGESTIONS.filter(function (name) {
        return name.toLowerCase().indexOf(val) !== -1;
      });
      renderSuggestions(matches);
    });

    input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') {
        box.style.display = 'none';
        runSearch(input.value);
      }
    });

    input.addEventListener('blur', function () {
      setTimeout(function () { box.style.display = 'none'; }, 150);
    });

    btn.addEventListener('click', function () {
      runSearch(input.value);
    });
  })();
</script>
"""

SUGGESTION_LIST = sorted(set(list(ASTRO_INDEX.keys()) + [
    "Mercury", "Neptune", "Uranus", "Pluto", "Europa", "Titan", "Io", "Ganymede",
    "Andromeda Galaxy", "Milky Way", "Crab Nebula", "Orion Nebula", "Sagittarius A*",
    "Alpha Centauri", "Betelgeuse", "Sirius", "Proxima Centauri", "Halley's Comet",
    "the Kuiper Belt", "the Asteroid Belt", "TRAPPIST-1e", "Kepler-452b", "the Oort Cloud",
]))


def _safe_filename(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "atmosphere"


def _get_query_param(name: str, default: str = "") -> str:
    try:
        value = st.query_params.get(name, default)
    except Exception:
        return default
    if isinstance(value, list):
        return value[0] if value else default
    return value if value is not None else default


def render_search_widget(current_value: str = "") -> None:
    html = (
        SEARCH_WIDGET_TEMPLATE
        .replace("__SUGGESTIONS_JSON__", json.dumps(SUGGESTION_LIST))
        .replace("__CURRENT_VALUE__", current_value.replace('"', "&quot;"))
    )
    components.html(html, height=300, scrolling=False)


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

        incoming_query = _get_query_param("q", "")
        render_search_widget(current_value=incoming_query)

        should_search = bool(incoming_query) and incoming_query != st.session_state.get("selected_query")

        if should_search:
            st.session_state.search_error = None
            with st.spinner(f"Listening to {incoming_query}..."):
                try:
                    raw_data = fetch_astronomical_context(incoming_query)
                    config = generate_psychoacoustic_schema(incoming_query, raw_data)
                    audio_bytes = synthesize_psychoacoustic_audio(config)
                except GeminiUnavailableError as e:
                    st.session_state.search_error = str(e)
                    st.session_state.selected_query = incoming_query
                else:
                    st.session_state.selected_query = incoming_query
                    st.session_state.profile_data = config
                    st.session_state.audio_bytes = audio_bytes

        if st.session_state.search_error:
            st.error(st.session_state.search_error)

        if st.session_state.profile_data is not None:
            config: CosmicSomaticConfig = st.session_state.profile_data

            st.markdown(f"<h2 class='cosmic-object-name'>{config.object_name}</h2>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='narrative-block'>{config.mental_presence_narrative}</div>",
                unsafe_allow_html=True,
            )

            st.audio(st.session_state.audio_bytes, format="audio/wav")
            st.download_button(
                "Download soundscape",
                data=st.session_state.audio_bytes,
                file_name=f"{_safe_filename(config.object_name)}_atmosphere.wav",
                mime="audio/wav",
                use_container_width=True,
            )

            with st.expander("Vector readout"):
                for field_name, field_value in config.vectors.model_dump().items():
                    label = field_name.replace("_", " ").capitalize()
                    st.markdown(
                        f"<div class='vector-row'><span>{label}</span>"
                        f"<span>{field_value:.2f}</span></div>",
                        unsafe_allow_html=True,
                    )


if __name__ == "__main__":
    main()
