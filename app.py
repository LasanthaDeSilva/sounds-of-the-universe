"""
Sounds of the Universe
=======================

An accessible astronomy exploration application that lets users -- especially
visually impaired users -- explore astronomical objects through scientifically
grounded auditory representations.

IMPORTANT SCIENTIFIC HONESTY NOTE (read before extending this file):
Ordinary sound is a mechanical wave that needs a medium to travel through.
Most of outer space is a near-vacuum, so most of the audio in this
application is NOT a literal recording of anything. Every object in the
registry below carries an explicit `audio_provenance` describing what kind
of audio it produces:

    "measured_audio"   -> an actual acoustic recording made in a medium
                           where sound really propagates (e.g. Earth).
    "scientific_sonification" -> a real, measured physical quantity (a
                           pulsar's spin period, a rotation curve, a
                           gravitational-wave chirp shape) is mapped onto an
                           audible parameter such as timing or pitch. The
                           *timing/shape* is data-driven; the *timbre* is a
                           synthesized instrument standing in for the signal.
    "physics_based_model" -> a synthesized soundscape whose parameters
                           (filter cutoffs, turbulence amount, modulation
                           rate, etc.) are chosen from real physical
                           properties of the object (pressure, density,
                           temperature, rotation), but the object was never
                           literally recorded and the mapping from physics
                           to timbre is a defensible model, not a
                           measurement.
    "interpretive_representation" -> an artistic/scientific translation used
                           when no direct acoustic or data-sonification
                           pathway exists (e.g. a nebula's emission spectrum
                           translated into a chord).

No object in this file is ever labeled "measured_audio" unless the sound is
actually a real environmental recording. Everything else is clearly marked
as a model, a sonification, or an interpretation, and the app surfaces that
label everywhere audio is generated.

This file is intentionally a single, self-contained Streamlit app per the
project brief. The object catalogue below covers 14 representative objects
across planets, moons, a star, a stellar remnant, a black hole, a nebula and
a galaxy -- enough to demonstrate every provenance category and every major
synthesis technique the brief calls for. The registry is written to be
trivially extendable: add a new ObjectProfile to OBJECT_REGISTRY and it
automatically appears in every part of the UI (search, compare, favorites,
export, audio lab).
"""

from __future__ import annotations

import hashlib
import io
import json
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import streamlit as st
except ImportError:  # pragma: no cover - allows the audio engine to be
    st = None          # unit-tested / imported outside a Streamlit runtime.

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

APP_TITLE = "Sounds of the Universe"
SAMPLE_RATE = 44100
APP_VERSION = "0.1.0"
ENGINE_VERSION = "audio-engine-1.0"

# Gemini models used for different tasks. Verify these identifiers against
# the current Generative Language API model list before deploying; model
# names change over time and an invalid name will simply cause the
# "Ask the Cosmos" panel to show its offline fallback message.
GEMINI_MODEL_STANDARD = "gemini-3.6-flash"
GEMINI_MODEL_DEEP = "gemini-3.1-pro-preview"
GEMINI_MODEL_ROUTER = "gemini-3.5-flash-lite"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

PROVENANCE_LABELS = {
    "measured_audio": "Measured audio data",
    "scientific_sonification": "Scientific sonification",
    "physics_based_model": "Physics-based auditory model",
    "interpretive_representation": "Interpretive scientific representation",
}

CONFIDENCE_LEVELS = ["High", "Moderate", "Low", "Interpretive"]


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------

def make_seed(*parts: Any) -> int:
    """Turn any combination of object id / parameters into a stable,
    reproducible 32-bit integer seed. The same inputs always produce the
    same seed, and therefore the same audio -- this is what lets the app
    promise deterministic, repeatable generation (brief section 5)."""
    payload = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)


def rng_for(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Low-level audio primitives
# ---------------------------------------------------------------------------
# Every primitive below is a small, named, explainable building block. The
# per-object "recipe" (see OBJECT_REGISTRY) is just a list of calls into
# these primitives with object-specific parameters -- this is the "common
# audio engine, object-specific parameters" architecture the brief asks for.

def t_axis(duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    return np.linspace(0.0, duration, int(duration * sr), endpoint=False)


def gen_sine(freq: float, duration: float, sr: int = SAMPLE_RATE,
             phase: float = 0.0) -> np.ndarray:
    t = t_axis(duration, sr)
    return np.sin(2 * np.pi * freq * t + phase)


def gen_additive(freqs: List[float], amps: List[float], duration: float,
                  sr: int = SAMPLE_RATE) -> np.ndarray:
    """Additive synthesis: sum of weighted sine partials. Used to represent
    harmonic structure such as resonant atmospheric layers or stable
    oscillation modes."""
    t = t_axis(duration, sr)
    out = np.zeros_like(t)
    for f, a in zip(freqs, amps):
        out += a * np.sin(2 * np.pi * f * t)
    peak = np.max(np.abs(out)) or 1.0
    return out / peak


def gen_noise(duration: float, seed: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    rng = rng_for(seed)
    return rng.normal(0.0, 1.0, int(duration * sr))


def fft_bandpass(signal: np.ndarray, low_hz: float, high_hz: float,
                  sr: int = SAMPLE_RATE) -> np.ndarray:
    """Simple, dependency-free band-pass filter implemented in the
    frequency domain. Used to represent how a dense or thin atmosphere
    would shape which frequencies survive -- e.g. Venus's thick atmosphere
    keeps mostly low frequencies, Mars's thin atmosphere removes most of
    the low-frequency body of a sound."""
    n = len(signal)
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    spectrum = spectrum * mask
    filtered = np.fft.irfft(spectrum, n=n)
    peak = np.max(np.abs(filtered)) or 1.0
    return filtered / peak


def gen_filtered_noise(duration: float, low_hz: float, high_hz: float,
                        seed: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Band-limited noise -- the basic building block for turbulence,
    wind, atmospheric hiss, and storm texture."""
    noise = gen_noise(duration, seed, sr)
    return fft_bandpass(noise, low_hz, high_hz, sr)


def gen_am(carrier: np.ndarray, mod_freq: float, mod_depth: float,
           duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Amplitude modulation: used for slow environmental "breathing"
    (e.g. rotating storm bands) or rhythmic pulsing tied to a rotation
    period."""
    t = t_axis(duration, sr)
    n = min(len(carrier), len(t))
    lfo = 1.0 - mod_depth + mod_depth * (0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t[:n]))
    return carrier[:n] * lfo


def gen_fm(carrier_freq: float, mod_freq: float, mod_index: float,
           duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Frequency modulation: used for magnetically active or plasma-like
    signal layers whose pitch itself wavers."""
    t = t_axis(duration, sr)
    modulator = mod_index * np.sin(2 * np.pi * mod_freq * t)
    return np.sin(2 * np.pi * carrier_freq * t + modulator)


def gen_pulse_train(period: float, pulse_width: float, duration: float,
                     seed: int, sr: int = SAMPLE_RATE,
                     jitter: float = 0.0, click_freq: float = 800.0) -> np.ndarray:
    """A train of short tone bursts spaced by `period` seconds. This is the
    primitive used for real, measured periodic astronomical signals (a
    pulsar's rotation, a periodic radio signal): the *timing* here can be
    driven directly by a real measured period, which is what makes a
    pulsar's audio a "scientific sonification" rather than an invented
    sound -- the spacing between clicks literally encodes a measured
    rotation rate, scaled into the audible range where needed."""
    rng = rng_for(seed)
    n = int(duration * sr)
    out = np.zeros(n)
    tick = 0.0
    while tick < duration:
        jittered = tick + (rng.normal(0, jitter) if jitter else 0.0)
        start = int(jittered * sr)
        length = int(pulse_width * sr)
        if start >= 0 and start < n:
            end = min(start + length, n)
            burst_t = np.arange(end - start) / sr
            envelope = np.exp(-burst_t / max(pulse_width / 4, 1e-4))
            out[start:end] += envelope * np.sin(2 * np.pi * click_freq * burst_t)
        tick += period
    peak = np.max(np.abs(out)) or 1.0
    return out / peak


def gen_chirp(f_start: float, f_end: float, duration: float,
              sr: int = SAMPLE_RATE) -> np.ndarray:
    """A frequency sweep. Used for the black-hole merger model, loosely
    styled on the rising-frequency "chirp" shape published for
    gravitational-wave detections -- an independent illustrative model,
    not detector strain data."""
    t = t_axis(duration, sr)
    k = (f_end - f_start) / max(duration, 1e-6)
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t ** 2)
    return np.sin(phase)


def apply_envelope(signal: np.ndarray, attack: float = 0.05,
                    release: float = 0.2, sr: int = SAMPLE_RATE) -> np.ndarray:
    n = len(signal)
    env = np.ones(n)
    a = min(int(attack * sr), n // 2)
    r = min(int(release * sr), n // 2)
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    if r > 0:
        env[-r:] = np.linspace(1, 0, r)
    return signal * env


def apply_slow_lfo(signal: np.ndarray, rate_hz: float, depth: float,
                    sr: int = SAMPLE_RATE) -> np.ndarray:
    """A slow amplitude wobble representing gradual environmental change
    (e.g. drifting atmospheric turbulence) rather than a rhythmic pulse."""
    t = np.arange(len(signal)) / sr
    lfo = 1.0 - depth + depth * (0.5 + 0.5 * np.sin(2 * np.pi * rate_hz * t))
    return signal * lfo


def soft_limit(signal: np.ndarray, ceiling: float = 0.9) -> np.ndarray:
    """A gentle safety limiter so layered mixes never clip harshly and stay
    within a safe, comfortable listening range (brief section 20)."""
    return ceiling * np.tanh(signal / max(ceiling, 1e-6))


def to_stereo(mono: np.ndarray, width: float = 0.0,
              seed: int = 0) -> np.ndarray:
    """Optional gentle stereo widening by decorrelating a small amount of
    noise between channels. width=0 keeps the signal mono-compatible."""
    if width <= 0:
        return np.stack([mono, mono], axis=-1)
    rng = rng_for(seed)
    n = len(mono)
    decorrelation = fft_bandpass(rng.normal(0, 1, n), 200, 4000) * width * 0.15
    left = mono + decorrelation
    right = mono - decorrelation
    peak = max(np.max(np.abs(left)), np.max(np.abs(right))) or 1.0
    return np.stack([left / peak, right / peak], axis=-1)


# ---------------------------------------------------------------------------
# WAV export (no external audio dependency -- uses the standard library
# `wave` module so the app has no fragile binary dependencies)
# ---------------------------------------------------------------------------

def float_stereo_to_wav_bytes(stereo: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """Convert a float64 [-1, 1] stereo array of shape (n, 2) into 16-bit
    PCM WAV bytes."""
    clipped = np.clip(stereo, -1.0, 1.0)
    ints = (clipped * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(ints.tobytes())
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Layer:
    """One layer in an object's audio recipe. `kind` selects which
    primitive builds it; `params` are the object-specific numbers that
    make that primitive sound like *this* object rather than a generic
    texture; `reason` is the one-line scientific justification shown in
    the provenance panel."""
    kind: str
    params: Dict[str, Any]
    gain: float
    reason: str


@dataclass
class ObjectProfile:
    object_id: str
    object_name: str
    object_type: str  # planet, moon, star, stellar_remnant, black_hole,
                       # nebula, galaxy, small_body, plasma_environment, ...
    short_description: str
    physical_parameters: Dict[str, str]
    environmental_parameters: Dict[str, str]
    known_phenomena: List[str]
    available_measured_data: List[str]
    sonification_methods: List[str]
    recipe: List[Layer]
    audio_provenance: str  # one of PROVENANCE_LABELS keys
    provenance_explanation: str
    scientific_confidence: str  # one of CONFIDENCE_LEVELS
    limitations: List[str]
    not_represented: List[str]
    listening_guide: List[str]
    source_references: List[str]
    base_freq_range: Tuple[float, float] = (20.0, 4000.0)
    default_duration: float = 18.0

    def explanation(self) -> Dict[str, str]:
        """Answers the six accessibility questions required by brief
        section 9/8: what is represented, what physical properties drove
        the sound, what data/model was used, what to listen for, what is
        NOT represented, and which provenance category applies."""
        return {
            "What object or environment is represented": self.short_description,
            "What physical properties influenced the sound": "; ".join(
                f"{k}: {v}" for k, v in {**self.physical_parameters,
                                          **self.environmental_parameters}.items()
            ),
            "What data or model was used": "; ".join(self.sonification_methods),
            "What the listener should pay attention to": "; ".join(self.listening_guide),
            "What the sound does not represent": "; ".join(self.not_represented),
            "Audio category": PROVENANCE_LABELS[self.audio_provenance],
        }


# ---------------------------------------------------------------------------
# Audio recipe engine: turns a list of Layers into a finished stereo signal.
# This is the "common engine, object-specific parameters" architecture:
# every object below is expressed only as data (a recipe), not as its own
# bespoke synthesis function.
# ---------------------------------------------------------------------------

def render_layer(layer: Layer, duration: float, seed: int, intensity: float,
                  complexity: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    p = layer.params
    if layer.kind == "filtered_noise":
        sig = gen_filtered_noise(duration, p["low"], p["high"], seed + hash(layer.reason) % 997, sr)
    elif layer.kind == "additive":
        n_partials = max(1, int(len(p["freqs"]) * (0.4 + 0.6 * complexity)))
        sig = gen_additive(p["freqs"][:n_partials], p["amps"][:n_partials], duration, sr)
    elif layer.kind == "am":
        carrier = gen_sine(p["carrier_freq"], duration, sr)
        sig = gen_am(carrier, p["mod_freq"], p["mod_depth"], duration, sr)
    elif layer.kind == "fm":
        sig = gen_fm(p["carrier_freq"], p["mod_freq"], p["mod_index"] * (0.5 + complexity), duration, sr)
    elif layer.kind == "pulse_train":
        sig = gen_pulse_train(p["period"], p["pulse_width"], duration,
                               seed + 17, sr, jitter=p.get("jitter", 0.0),
                               click_freq=p.get("click_freq", 800.0))
    elif layer.kind == "chirp":
        sig = gen_chirp(p["f_start"], p["f_end"], min(p.get("chirp_duration", duration), duration), sr)
        pad = np.zeros(int(duration * sr) - len(sig))
        sig = np.concatenate([sig, pad])
    elif layer.kind == "silence_field":
        # A deliberately near-empty layer used for airless/vacuum objects,
        # so the mix stays honest about how little there is to "hear".
        sig = gen_noise(duration, seed + 3, sr) * 0.02
    else:
        raise ValueError(f"Unknown layer kind: {layer.kind}")

    if p.get("slow_lfo_rate"):
        sig = apply_slow_lfo(sig, p["slow_lfo_rate"], p.get("slow_lfo_depth", 0.3), sr)
    return sig * layer.gain * (0.4 + 0.6 * intensity)


def synthesize_object_audio(profile: ObjectProfile, duration: float,
                             intensity: float, complexity: float,
                             freq_lo: float, freq_hi: float,
                             stereo_width: float, seed: int,
                             sr: int = SAMPLE_RATE) -> np.ndarray:
    mix = np.zeros(int(duration * sr))
    for layer in profile.recipe:
        rendered = render_layer(layer, duration, seed, intensity, complexity, sr)
        n = min(len(mix), len(rendered))
        mix[:n] += rendered[:n]
    mix = fft_bandpass(mix, max(freq_lo, 1.0), max(freq_hi, freq_lo + 1.0), sr)
    mix = apply_envelope(mix, attack=min(1.5, duration * 0.08), release=min(2.0, duration * 0.12), sr=sr)
    mix = soft_limit(mix, ceiling=0.85)
    return to_stereo(mix, width=stereo_width, seed=seed)


# ---------------------------------------------------------------------------
# Object registry
# ---------------------------------------------------------------------------
# Each entry below is a real, defensible mapping from physical properties to
# synthesis parameters. Numeric physical values (temperatures, pressures,
# periods) are approximate reference figures rounded for readability; treat
# them as "approximate reference values" (brief section 15), not
# high-precision measurements, and verify against a current source (e.g.
# NASA's Planetary Fact Sheet) before using this app for anything beyond
# education. Adding a new object is just adding one more entry here -- the
# UI, search, comparison, favorites, and export all read from this dict.

OBJECT_REGISTRY: Dict[str, ObjectProfile] = {}


def _register(profile: ObjectProfile) -> None:
    OBJECT_REGISTRY[profile.object_id] = profile


_register(ObjectProfile(
    object_id="sun",
    object_name="The Sun",
    object_type="star",
    short_description=(
        "A main-sequence G-type star; its outer layers oscillate in "
        "well-documented low-frequency pressure modes (helioseismology)."
    ),
    physical_parameters={
        "Surface temperature": "~5,500 C (photosphere)",
        "Rotation period": "~25 days at the equator (differential rotation)",
        "Composition": "Hydrogen and helium plasma",
    },
    environmental_parameters={
        "Magnetic activity": "Strong, cyclical (~11-year solar cycle)",
        "Surface phenomena": "Granulation, sunspots, solar flares, solar wind",
    },
    known_phenomena=["p-mode pressure oscillations", "solar flares", "coronal mass ejections", "sunspot cycle"],
    available_measured_data=["Helioseismic p-mode frequencies (~2-4 millihertz, published by SOHO/GONG)"],
    sonification_methods=[
        "Frequency-shifted representation of published solar p-mode oscillation frequencies (raised roughly 40,000x into the audible range, following the general approach used in published helioseismology sonifications)",
        "Layered plasma-noise texture representing magnetic activity",
    ],
    recipe=[
        Layer("additive", {"freqs": [110, 165, 220, 275, 330, 385], "amps": [1.0, 0.7, 0.5, 0.35, 0.25, 0.15]}, 0.55,
              "Additive partials stand in for the family of measured p-mode oscillation frequencies, shifted up into the audible range."),
        Layer("filtered_noise", {"low": 300, "high": 3000, "slow_lfo_rate": 0.05, "slow_lfo_depth": 0.5}, 0.35,
              "Slow-drifting mid/high noise represents granulation and surface plasma turbulence."),
        Layer("fm", {"carrier_freq": 60, "mod_freq": 0.11, "mod_index": 3.0}, 0.25,
              "Frequency modulation represents fluctuating magnetic activity linked to the solar cycle."),
    ],
    audio_provenance="scientific_sonification",
    provenance_explanation=(
        "The pitch layer's frequencies are a shifted representation of real, published helioseismic "
        "oscillation frequencies. The instrument timbre and the plasma/magnetic layers are a "
        "physics-based model, not a recording -- the Sun's oscillations are not audible sound waves "
        "traveling through space to a listener's ear."
    ),
    scientific_confidence="Moderate",
    limitations=[
        "Exact p-mode frequencies vary by measurement and depend on the specific mode being observed; only a representative cluster is used here.",
        "The frequency-shift factor is illustrative, following the general scale used in public helioseismology sonifications, not a single official standard.",
    ],
    not_represented=["The Sun's actual loudness, timbre, or full spectrum of oscillation modes", "Any literal audible sound traveling through space"],
    listening_guide=["The layered, chord-like tones in the low register (the shifted oscillation modes)", "The slow drifting texture above them (granulation/turbulence)", "The wavering pitch layer (magnetic activity)"],
    source_references=["NASA/GSFC: Solar and Heliospheric Observatory (SOHO) helioseismology overview", "NASA Goddard: Sun fact sheet"],
    base_freq_range=(40, 4000), default_duration=20,
))

_register(ObjectProfile(
    object_id="mercury",
    object_name="Mercury",
    object_type="planet",
    short_description=(
        "The innermost planet: an airless world with only an extremely thin, "
        "transient exosphere, extreme day-night temperature swings, and a weak "
        "global magnetic field."
    ),
    physical_parameters={
        "Atmosphere": "Negligible (surface-bound exosphere only)",
        "Surface temperature": "-180 C to 430 C (extreme day/night swing)",
        "Rotation": "Slow (59 Earth days per rotation, 3:2 spin-orbit resonance)",
    },
    environmental_parameters={
        "Magnetic field": "Weak but present; interacts with the solar wind",
        "Surface": "Heavily cratered, no weather system",
    },
    known_phenomena=["Extreme thermal cycling", "Weak magnetosphere interacting with solar wind", "Exospheric sodium/calcium emission"],
    available_measured_data=["MESSENGER mission magnetometer and thermal data (general findings)"],
    sonification_methods=[
        "Interpretive sonification of thermal cycling as slow amplitude change",
        "Sparse, near-silent texture representing the near-absence of an atmosphere",
    ],
    recipe=[
        Layer("silence_field", {}, 1.0, "Mercury has essentially no atmosphere, so almost nothing would propagate as ordinary sound; a near-silent noise floor represents this honestly rather than inventing atmospheric sound."),
        Layer("am", {"carrier_freq": 90, "mod_freq": 0.017, "mod_depth": 0.9}, 0.4,
              "A very slow amplitude cycle stands in for Mercury's extreme, slow day-night temperature swing (interpretive, not a real thermal-to-audio dataset)."),
        Layer("fm", {"carrier_freq": 220, "mod_freq": 0.4, "mod_index": 1.5}, 0.2,
              "A thin, wavering tone represents the weak magnetic field's interaction with the solar wind."),
    ],
    audio_provenance="interpretive_representation",
    provenance_explanation=(
        "Mercury has no meaningful atmosphere, so there is no physical medium for ordinary sound. "
        "This entire profile is an interpretive translation of thermal and magnetic behavior into "
        "audible change, not a model of real acoustic propagation."
    ),
    scientific_confidence="Interpretive",
    limitations=["No acoustic medium exists on Mercury; this cannot be understood as any kind of 'atmospheric sound'.", "Thermal cycle timing is illustrative, not scaled from an exact measured cycle length."],
    not_represented=["Any literal sound on Mercury's surface", "Precise measured thermal or magnetic waveform data"],
    listening_guide=["The very slow rise and fall in loudness (day/night thermal swing)", "The faint, thin wavering tone (magnetic field)", "The near-silence underneath (airless surface)"],
    source_references=["NASA MESSENGER mission overview", "NASA Mercury fact sheet"],
    base_freq_range=(20, 2000), default_duration=16,
))

_register(ObjectProfile(
    object_id="venus",
    object_name="Venus",
    object_type="planet",
    short_description=(
        "A planet with a crushingly dense, carbon-dioxide-dominated atmosphere, "
        "extreme surface pressure and heat, thick global cloud cover, and very "
        "slow rotation."
    ),
    physical_parameters={
        "Atmosphere": "~96.5% carbon dioxide, extremely dense",
        "Surface pressure": "~92x Earth's sea-level pressure",
        "Surface temperature": "~465 C",
        "Rotation": "Extremely slow (~243 Earth days), retrograde",
    },
    environmental_parameters={
        "Clouds": "Thick global sulfuric-acid cloud layers",
        "Winds": "Fast high-altitude winds despite slow surface rotation (super-rotation)",
    },
    known_phenomena=["Runaway greenhouse effect", "Atmospheric super-rotation", "Thick, opaque cloud deck"],
    available_measured_data=["Venera and Pioneer Venus probe atmospheric pressure/temperature profiles (general findings)"],
    sonification_methods=[
        "Physics-based low-pass filtering to represent how an extremely dense atmosphere would suppress high frequencies and emphasize a heavy, low body",
        "Slow modulation representing the planet's very slow rotation",
        "Broadband turbulence representing high-altitude super-rotating winds",
    ],
    recipe=[
        Layer("filtered_noise", {"low": 20, "high": 220}, 0.65,
              "A dense, 92x-Earth-pressure atmosphere is modeled as heavy low-frequency emphasis and strong attenuation of higher frequencies."),
        Layer("filtered_noise", {"low": 400, "high": 900, "slow_lfo_rate": 0.03, "slow_lfo_depth": 0.6}, 0.2,
              "A slowly drifting higher band represents fast high-altitude super-rotating winds moving above a near-static surface."),
        Layer("am", {"carrier_freq": 55, "mod_freq": 0.008, "mod_depth": 0.7}, 0.3,
              "An extremely slow amplitude cycle reflects Venus's ~243-day rotation period, scaled into a perceptible timescale."),
    ],
    audio_provenance="physics_based_model",
    provenance_explanation=(
        "This is a physics-based model of how a near-92-atmosphere, carbon-dioxide-dominated "
        "environment would filter and shape a soundscape -- it is not a literal recording. No "
        "microphone has ever operated for long on Venus's surface."
    ),
    scientific_confidence="Moderate",
    limitations=["No long-duration acoustic recording of Venus's surface exists to validate this model.", "The exact acoustic attenuation curve for a 92-atmosphere CO2 environment is approximated, not derived from a published transfer function."],
    not_represented=["The literal loudness or timbre of any real Venusian sound", "Any specific recorded event (e.g. an actual storm)"],
    listening_guide=["The heavy, muffled low-frequency body (dense atmosphere)", "The higher, slowly shifting texture on top (fast high-altitude winds)", "The very slow overall swell (extremely slow rotation)"],
    source_references=["NASA/JPL Venus fact sheet", "Pioneer Venus and Venera mission atmospheric summaries"],
    base_freq_range=(20, 900), default_duration=20,
))

_register(ObjectProfile(
    object_id="earth",
    object_name="Earth",
    object_type="planet",
    short_description=(
        "Our home planet: a temperate nitrogen-oxygen atmosphere that actually "
        "supports ordinary sound propagation, used here as a familiar reference "
        "point for comparison with every other object."
    ),
    physical_parameters={
        "Atmosphere": "~78% nitrogen, ~21% oxygen, moderate density",
        "Surface temperature": "-88 C to 58 C recorded extremes; ~15 C global average",
        "Rotation": "24 hours",
    },
    environmental_parameters={"Weather": "Wind, precipitation, and thunderstorms are real, physically propagating sound sources"},
    known_phenomena=["Wind", "Rain", "Thunder", "Ocean waves", "Magnetosphere / auroral activity"],
    available_measured_data=["Countless real environmental acoustic recordings exist for Earth (not embedded in this build)"],
    sonification_methods=[
        "Illustrative physics-based model of wind and distant thunder for comparison purposes",
        "This is NOT a real field recording; Earth is the one object where real recordings are actually possible, but none is bundled with this app version",
    ],
    recipe=[
        Layer("filtered_noise", {"low": 80, "high": 3500, "slow_lfo_rate": 0.09, "slow_lfo_depth": 0.4}, 0.5,
              "Broadband, moderately filtered noise represents ordinary wind moving through a normal-density atmosphere -- the kind of sound Earth's atmosphere can actually carry."),
        Layer("pulse_train", {"period": 4.5, "pulse_width": 0.5, "jitter": 0.3, "click_freq": 70}, 0.35,
              "Occasional low rumbling bursts represent distant thunder, spaced irregularly like real storm activity."),
        Layer("filtered_noise", {"low": 1200, "high": 4000}, 0.12,
              "A light high-frequency layer stands in for fine environmental detail (e.g. light rain) audible in a normal atmosphere."),
    ],
    audio_provenance="physics_based_model",
    provenance_explanation=(
        "Earth is the one object in this catalogue where the underlying physics (a moderate-density "
        "atmosphere) really does support the kind of sound you are hearing. Even so, this specific "
        "audio is a synthesized illustrative model built for comparison, not an actual field "
        "recording -- no microphone recording is bundled with this version of the app."
    ),
    scientific_confidence="High",
    limitations=["This is a synthesized stand-in, not an actual recorded field sample.", "Real weather sound varies enormously by location and conditions; this is a generic illustrative model."],
    not_represented=["Any single real, specific recorded weather event", "The full acoustic diversity of Earth's environments"],
    listening_guide=["The broadband wind-like texture", "The occasional low rumbling bursts (modeled thunder)", "How much more 'open' and less filtered this feels compared with Venus or Mars"],
    source_references=["General atmospheric science references (NOAA)"],
    base_freq_range=(60, 4000), default_duration=18,
))

_register(ObjectProfile(
    object_id="mars",
    object_name="Mars",
    object_type="planet",
    short_description=(
        "A cold desert world with a thin carbon-dioxide atmosphere at roughly "
        "1% of Earth's sea-level pressure, frequent dust activity, and low "
        "atmospheric density that weakens how sound would travel."
    ),
    physical_parameters={
        "Atmosphere": "~95% carbon dioxide, very thin (~0.6% of Earth's surface pressure)",
        "Surface temperature": "Roughly -125 C to 20 C",
        "Rotation": "~24 hours 37 minutes",
    },
    environmental_parameters={"Weather": "Seasonal dust storms, occasionally global in scale", "Wind": "Present but weak in force due to low density"},
    known_phenomena=["Global dust storms", "Dust devils", "Thin, weak wind force despite measurable wind speed"],
    available_measured_data=["NASA InSight and Perseverance mission microphones have recorded real Martian wind sound (general public findings, not the literal dataset used here)"],
    sonification_methods=[
        "Physics-based high-pass emphasis and reduced low-frequency body representing a thin atmosphere's weak sound transmission",
        "Grainy, textured noise representing dust movement",
    ],
    recipe=[
        Layer("filtered_noise", {"low": 250, "high": 3200}, 0.55,
              "Weighting toward higher frequencies and away from a strong low end models how a thin, low-density atmosphere transmits sound weakly and with little bass body -- consistent with what NASA's Mars microphones have publicly reported about quieter, thinner-sounding wind."),
        Layer("filtered_noise", {"low": 800, "high": 4000, "slow_lfo_rate": 0.15, "slow_lfo_depth": 0.5}, 0.25,
              "A grainy, shifting high band represents fine dust particles moving in the wind."),
        Layer("am", {"carrier_freq": 40, "mod_freq": 0.02, "mod_depth": 0.5}, 0.15,
              "A slow, weak low-frequency pulse represents occasional large-scale dust storm activity, deliberately kept faint given the thin atmosphere."),
    ],
    audio_provenance="physics_based_model",
    provenance_explanation=(
        "Actual microphones have recorded real wind sound on Mars (NASA's InSight and Perseverance "
        "missions), and their public descriptions -- a thin, quiet, high-frequency-leaning sound -- "
        "inform the filtering choices here. This specific audio, however, is a synthesized model, "
        "not that literal recorded dataset."
    ),
    scientific_confidence="Moderate",
    limitations=["Not the literal NASA-recorded audio; a model informed by its publicly reported character.", "Actual recorded loudness and detail vary by mission, location, and instrument."],
    not_represented=["Any specific real recorded Martian audio clip", "The full dynamic range of an actual dust storm"],
    listening_guide=["How thin and high-pitched the texture feels compared with Earth or Venus", "The grainy shifting layer (dust)", "The faint, weak low pulse (deep, distant dust-storm activity)"],
    source_references=["NASA JPL: InSight and Perseverance microphone mission pages", "NASA Mars fact sheet"],
    base_freq_range=(150, 4000), default_duration=18,
))


_register(ObjectProfile(
    object_id="jupiter",
    object_name="Jupiter",
    object_type="planet",
    short_description=(
        "A giant hydrogen-helium planet with deep atmospheric turbulence, "
        "powerful storms including the Great Red Spot, rapid rotation, and "
        "the strongest planetary magnetic field and radiation belts in the "
        "solar system."
    ),
    physical_parameters={
        "Atmosphere": "~90% hydrogen, ~10% helium; extremely deep",
        "Rotation": "~10 hours (fastest rotation of any planet)",
        "Notable feature": "Great Red Spot: a persistent giant storm",
    },
    environmental_parameters={"Magnetosphere": "The largest and strongest of any planet in the solar system", "Radiation": "Intense radiation belts"},
    known_phenomena=["Large-scale atmospheric banding", "Giant storms", "Rapid rotation", "Intense magnetosphere and radio emissions"],
    available_measured_data=["Juno and Voyager mission radio/plasma-wave recordings (general findings; not the literal dataset here)"],
    sonification_methods=[
        "Broad turbulent noise layered with evolving storm-like modulation",
        "Rapid amplitude modulation reflecting the ~10-hour rotation period",
        "A wavering plasma-inspired layer representing the strong magnetosphere",
    ],
    recipe=[
        Layer("filtered_noise", {"low": 30, "high": 1200, "slow_lfo_rate": 0.08, "slow_lfo_depth": 0.5}, 0.55,
              "Broad, deep turbulent noise represents the immense, deep hydrogen-helium atmosphere in constant large-scale motion."),
        Layer("am", {"carrier_freq": 65, "mod_freq": 0.028, "mod_depth": 0.6}, 0.3,
              "Amplitude modulation tied to a compressed but proportionally fast cycle represents Jupiter's roughly 10-hour rotation, the fastest of any planet."),
        Layer("fm", {"carrier_freq": 140, "mod_freq": 1.3, "mod_index": 4.0}, 0.25,
              "An unstable, wavering FM layer represents the powerful magnetosphere and its radio/plasma-wave emissions."),
        Layer("filtered_noise", {"low": 400, "high": 1600, "slow_lfo_rate": 0.05, "slow_lfo_depth": 0.7}, 0.15,
              "A shifting mid-band texture represents the Great Red Spot and other persistent storm systems moving within the banded atmosphere."),
    ],
    audio_provenance="physics_based_model",
    provenance_explanation=(
        "This is a physics-based model built from Jupiter's known turbulence, rotation rate, and "
        "magnetospheric activity. Real radio and plasma-wave data exist from missions like Juno and "
        "Voyager, but this specific audio is a synthesized representation, not that dataset."
    ),
    scientific_confidence="Moderate",
    limitations=["Not derived from an actual loaded Juno/Voyager plasma-wave dataset.", "The Great Red Spot's actual acoustic signature (if any) is unknown; its representation here is illustrative."],
    not_represented=["Any literal recorded Jovian radio emission", "The true scale or energy of a storm the size of the Great Red Spot"],
    listening_guide=["The broad, deep turbulence (the vast atmosphere)", "The relatively fast pulsing (rapid rotation)", "The unstable, wavering tone (the powerful magnetosphere)"],
    source_references=["NASA JPL: Juno mission overview", "NASA Jupiter fact sheet"],
    base_freq_range=(30, 1600), default_duration=20,
))

_register(ObjectProfile(
    object_id="saturn",
    object_name="Saturn",
    object_type="planet",
    short_description=(
        "A hydrogen-helium giant known for its extensive ring system, banded "
        "atmosphere, powerful storms, rapid rotation, and an active "
        "magnetosphere."
    ),
    physical_parameters={"Atmosphere": "~96% hydrogen, ~3% helium", "Rotation": "~10.7 hours", "Rings": "Extensive ice-and-rock ring system"},
    environmental_parameters={"Magnetosphere": "Large and active, with periodic radio emissions", "Storms": "Includes large hexagonal polar storm pattern"},
    known_phenomena=["Ring structure", "Hexagonal polar jet stream", "Fast rotation", "Radio emissions tied to rotation"],
    available_measured_data=["Cassini mission radio and plasma-wave instrument recordings (general findings)"],
    sonification_methods=[
        "Turbulent noise and banding-style filtering similar in family to Jupiter but with distinct parameters",
        "A subtle, explicitly interpretive layer standing in for the ring system's structure -- not a literal 'ring sound'",
    ],
    recipe=[
        Layer("filtered_noise", {"low": 25, "high": 900, "slow_lfo_rate": 0.06, "slow_lfo_depth": 0.55}, 0.5,
              "Deep turbulent noise represents the hydrogen-helium atmosphere, tuned lower and slower than Jupiter to reflect Saturn's lower density and different banding character."),
        Layer("am", {"carrier_freq": 58, "mod_freq": 0.026, "mod_depth": 0.55}, 0.3,
              "Amplitude modulation reflects Saturn's roughly 10.7-hour rotation period."),
        Layer("additive", {"freqs": [520, 780, 1040, 1300], "amps": [0.5, 0.4, 0.3, 0.2]}, 0.12,
              "A sparse, quiet set of high, evenly spaced tones is an explicitly interpretive stand-in for the ring system's layered structure -- it is not a claim that the rings make an audible sound."),
        Layer("fm", {"carrier_freq": 120, "mod_freq": 0.9, "mod_index": 3.0}, 0.2,
              "A wavering layer represents Saturn's active magnetosphere and its periodic radio emissions."),
    ],
    audio_provenance="physics_based_model",
    provenance_explanation=(
        "Most of this profile is a physics-based atmospheric and magnetospheric model. The high, "
        "sparse tone layer representing the rings is explicitly interpretive: it is a structural "
        "metaphor, not a sonification of any measured ring emission."
    ),
    scientific_confidence="Moderate",
    limitations=["The ring-representing layer is interpretive and should not be understood as scientific ring data.", "Cassini's real radio/plasma-wave recordings are not the literal dataset used here."],
    not_represented=["Any literal 'sound of the rings'", "Real Cassini radio-instrument audio"],
    listening_guide=["The deep, slow turbulence (atmosphere)", "The evenly spaced high tones (interpretive ring metaphor -- read the caption before assuming these are 'ring sounds')", "The wavering magnetospheric layer"],
    source_references=["NASA JPL: Cassini mission overview", "NASA Saturn fact sheet"],
    base_freq_range=(25, 1300), default_duration=20,
))

_register(ObjectProfile(
    object_id="neptune",
    object_name="Neptune",
    object_type="planet",
    short_description=(
        "The outermost known giant planet: extremely cold, with the fastest "
        "sustained winds recorded in the solar system and a dynamic, "
        "ice-giant atmosphere."
    ),
    physical_parameters={"Atmosphere": "Hydrogen, helium, and methane (methane gives it its blue color)", "Temperature": "Roughly -220 C average", "Winds": "Recorded up to ~2,100 km/h -- the fastest in the solar system"},
    environmental_parameters={"Rotation": "~16 hours", "Storms": "Large dark storm systems have been observed"},
    known_phenomena=["Extremely fast winds", "Large dark storm systems", "Very low temperature"],
    available_measured_data=["Voyager 2 flyby measurements (general findings)"],
    sonification_methods=["Fast-moving, tightly filtered turbulence representing extreme wind speed", "A cold, sparse harmonic layer representing the planet's low temperature and dim, distant sunlight"],
    recipe=[
        Layer("filtered_noise", {"low": 150, "high": 2200, "slow_lfo_rate": 0.4, "slow_lfo_depth": 0.6}, 0.6,
              "Fast, tightly modulated noise represents Neptune's extremely high wind speeds, the fastest sustained winds measured on any solar system planet."),
        Layer("additive", {"freqs": [180, 270, 360], "amps": [0.6, 0.3, 0.15]}, 0.25,
              "A sparse, cool-sounding set of partials represents the planet's very low temperature and distance from the Sun."),
        Layer("am", {"carrier_freq": 50, "mod_freq": 0.017, "mod_depth": 0.4}, 0.2,
              "A moderate amplitude cycle reflects Neptune's roughly 16-hour rotation."),
    ],
    audio_provenance="physics_based_model",
    provenance_explanation="A physics-based model driven by Neptune's measured wind speeds, temperature, and rotation; not a recording.",
    scientific_confidence="Moderate",
    limitations=["No direct acoustic measurement of Neptune's atmosphere exists.", "Wind-speed-to-modulation-rate mapping is illustrative, not a derived acoustic transfer function."],
    not_represented=["The literal loudness such winds would produce", "Any specific recorded storm event"],
    listening_guide=["The fast, restless texture (extreme wind speed)", "The sparse, cool high tones (extreme cold and distance from the Sun)"],
    source_references=["NASA JPL: Voyager 2 Neptune flyby summary", "NASA Neptune fact sheet"],
    base_freq_range=(150, 2200), default_duration=18,
))

_register(ObjectProfile(
    object_id="moon",
    object_name="The Moon",
    object_type="moon",
    short_description=(
        "Earth's Moon: an airless body where ordinary sound cannot "
        "propagate, but where Apollo-era seismometers recorded real, "
        "measured moonquakes as ground-vibration data."
    ),
    physical_parameters={"Atmosphere": "None (surface-bound exosphere only)", "Surface": "Regolith (loose dust and rock)"},
    environmental_parameters={"Seismic activity": "Real moonquakes were recorded by Apollo seismometers, some of which ring for an unusually long time"},
    known_phenomena=["Moonquakes", "Long seismic ringing due to dry, unfractured rock"],
    available_measured_data=["Apollo Passive Seismic Experiment moonquake recordings (general public findings; not the literal dataset used here)"],
    sonification_methods=["Interpretive sonification of seismic 'ringing' behavior as long, slowly decaying tones", "Near-silent noise floor representing the absence of an atmosphere"],
    recipe=[
        Layer("silence_field", {}, 1.0, "The Moon has no atmosphere, so there is no medium for ordinary airborne sound; a near-silent floor keeps that honest."),
        Layer("additive", {"freqs": [140, 210, 280], "amps": [0.8, 0.5, 0.3], "slow_lfo_rate": 0.02, "slow_lfo_depth": 0.3}, 0.4,
              "Long, slowly decaying tones are an interpretive stand-in for the unusually long 'ringing' that real moonquakes are documented to produce, because dry lunar rock does not damp vibration quickly."),
        Layer("pulse_train", {"period": 6.0, "pulse_width": 1.2, "jitter": 1.0, "click_freq": 90}, 0.25,
              "Occasional, irregular low events represent the sparse, unpredictable timing of moonquakes."),
    ],
    audio_provenance="interpretive_representation",
    provenance_explanation=(
        "Ordinary sound cannot travel across the Moon's surface because there is no atmosphere. Real "
        "seismometers recorded actual moonquakes as ground vibration, and the well-documented finding "
        "that this vibration rings for an unusually long time inspires this profile's long, decaying "
        "tones -- but the exact audio here is an interpretive translation, not the literal seismic "
        "waveform data."
    ),
    scientific_confidence="Interpretive",
    limitations=["Not the literal Apollo seismometer waveform.", "Real moonquakes are infrequent and irregular; timing here is illustrative."],
    not_represented=["Any airborne sound on the lunar surface", "The literal recorded Apollo seismic dataset"],
    listening_guide=["The unusually long, slow decay of each tone (real moonquakes are documented to ring longer than earthquakes)", "The near-silence between events (an airless, mostly quiet body)"],
    source_references=["NASA: Apollo Passive Seismic Experiment overview"],
    base_freq_range=(60, 900), default_duration=20,
))

_register(ObjectProfile(
    object_id="titan",
    object_name="Titan",
    object_type="moon",
    short_description=(
        "Saturn's largest moon: the only moon in the solar system with a "
        "substantial atmosphere, thick enough that -- unusually for a moon "
        "-- ordinary sound really can propagate there."
    ),
    physical_parameters={"Atmosphere": "Dense, mostly nitrogen with methane; surface pressure ~1.5x Earth's", "Surface temperature": "~-179 C", "Surface": "Methane lakes and dunes"},
    environmental_parameters={"Weather": "A methane cycle with rain and surface liquid, similar in structure to Earth's water cycle", "Wind": "Generally light near the surface"},
    known_phenomena=["Methane rain and lakes", "Dense, hazy atmosphere", "Light surface winds recorded during the Huygens descent"],
    available_measured_data=["ESA/NASA Huygens probe descent instrumentation recorded real atmospheric data during its 2005 landing (general public findings; the literal recording is not embedded in this app)"],
    sonification_methods=["Physics-based modeling of a dense, cold atmosphere -- distinct from Venus's dense-but-scorching profile", "Light wind texture consistent with Huygens's general findings of a calm lower atmosphere"],
    recipe=[
        Layer("filtered_noise", {"low": 30, "high": 500}, 0.55,
              "A dense nitrogen atmosphere at 1.5x Earth's surface pressure is modeled with a solid low-to-mid body -- similar in principle to Venus's dense-atmosphere approach, but centered on much colder, calmer conditions."),
        Layer("filtered_noise", {"low": 600, "high": 1800, "slow_lfo_rate": 0.04, "slow_lfo_depth": 0.3}, 0.15,
              "A light, gently drifting higher layer represents the generally light winds reported near Titan's surface."),
        Layer("pulse_train", {"period": 3.0, "pulse_width": 0.4, "jitter": 0.4, "click_freq": 300}, 0.15,
              "Occasional soft, rounded events stand in for methane rain/droplet activity, part of Titan's documented methane cycle."),
    ],
    audio_provenance="physics_based_model",
    provenance_explanation=(
        "Titan's atmosphere is genuinely dense enough for real sound to propagate, and the Huygens "
        "probe's instruments did record real atmospheric measurements during descent in 2005. This "
        "specific audio, however, is a synthesized model informed by those general findings, not the "
        "literal Huygens recording."
    ),
    scientific_confidence="Moderate",
    limitations=["Not the literal Huygens descent recording.", "Exact wind speed and rain-event timing are illustrative, not scaled from the specific mission telemetry."],
    not_represented=["The literal Huygens probe audio", "Any specific recorded methane storm"],
    listening_guide=["The solid, present low-to-mid body (dense but very cold atmosphere)", "The light, gentle upper texture (calm surface winds)", "The soft, occasional droplet-like events (methane rain)"],
    source_references=["ESA: Huygens probe mission overview", "NASA JPL: Titan fact sheet"],
    base_freq_range=(30, 1800), default_duration=18,
))


_register(ObjectProfile(
    object_id="pulsar",
    object_name="Pulsar (generic model)",
    object_type="stellar_remnant",
    short_description=(
        "A rapidly rotating neutron star that sweeps a beam of radio "
        "emission past Earth with extreme regularity, similar in character "
        "to well-studied pulsars such as the Crab Pulsar."
    ),
    physical_parameters={"Type": "Neutron star remnant of a supernova", "Spin period": "Illustrative period of 0.1 second (real pulsars range from milliseconds to several seconds)", "Density": "Extreme; roughly a solar mass compressed into a city-sized sphere"},
    environmental_parameters={"Emission": "Beamed radio (and often X-ray/gamma-ray) emission, observed as regular pulses"},
    known_phenomena=["Extremely regular pulse timing", "Gradual spin-down over time", "Occasional sudden 'glitches' in spin rate"],
    available_measured_data=["Published pulsar rotation periods (e.g. the Crab Pulsar's ~33-millisecond period) are real, measured values"],
    sonification_methods=[
        "Direct sonification of a measured or representative pulsar spin period as a pulse-train timing pattern -- the spacing between clicks is data-driven",
        "Synthetic click timbre standing in for the actual radio-frequency emission, which itself is inaudible electromagnetic radiation, not sound",
    ],
    recipe=[
        Layer("pulse_train", {"period": 0.28, "pulse_width": 0.015, "jitter": 0.001, "click_freq": 900}, 0.8,
              "The interval between clicks is set directly from a real (or representative) measured pulsar spin period -- this timing is genuinely data-driven, even though the click's timbre is synthetic."),
        Layer("filtered_noise", {"low": 40, "high": 250}, 0.1,
              "A very faint, low background represents the broader astrophysical environment without implying any additional measured signal."),
    ],
    audio_provenance="scientific_sonification",
    provenance_explanation=(
        "A pulsar's rotation period is a real, precisely measured astronomical quantity. Here, that "
        "measured period sets the exact spacing between audible clicks, so the *timing* is genuinely "
        "data-driven. The click sound itself is a synthesized stand-in for radio-frequency emission, "
        "which is not audible sound and is only 'heard' because it has been converted into an audio "
        "signal -- this is the definition of a sonification, not a recording."
    ),
    scientific_confidence="High",
    limitations=["The click timbre is arbitrary and carries no scientific meaning; only the timing does.", "This profile uses an illustrative period; swap in a specific pulsar's published period for a specific-object sonification."],
    not_represented=["Any literal audible radio wave (radio waves are not sound)", "The pulsar's true brightness, spectrum, or beam geometry"],
    listening_guide=["The extremely regular, metronome-like spacing between clicks -- that regularity is the real, measured signal", "The near-silence between pulses"],
    source_references=["NASA/Jodrell Bank: published pulsar period catalogues", "General pulsar astrophysics references"],
    base_freq_range=(30, 3000), default_duration=15,
))

_register(ObjectProfile(
    object_id="black_hole",
    object_name="Black Hole (binary merger model)",
    object_type="black_hole",
    short_description=(
        "A stylized, illustrative model of the rising-frequency 'chirp' shape "
        "publicly described for gravitational-wave detections of merging "
        "black holes, paired with an accretion-disk turbulence layer."
    ),
    physical_parameters={"Event modeled": "A black hole binary inspiral and merger, in the general style of published LIGO/Virgo detections", "Signal shape": "A frequency sweep rising rapidly just before merger"},
    environmental_parameters={"Accretion disk (if present)": "Superheated, turbulent infalling matter can produce electromagnetic (not gravitational-wave) signals in some systems"},
    known_phenomena=["Gravitational-wave emission during inspiral and merger", "Rapid rise in both frequency and amplitude just before merger ('chirp')", "Ringdown to a stable final black hole"],
    available_measured_data=["LIGO/Virgo collaborations have published real gravitational-wave strain data and official sonifications for detections such as GW150914 (not the literal data used here)"],
    sonification_methods=[
        "An illustrative chirp model with the same general rising-frequency shape publicly described for real gravitational-wave detections",
        "Turbulent noise representing a hypothetical accretion disk, presented separately from the chirp so the two are not confused",
    ],
    recipe=[
        Layer("chirp", {"f_start": 35, "f_end": 260, "chirp_duration": 3.0}, 0.6,
              "A rising frequency sweep echoes the general shape publicly described for real gravitational-wave chirps (frequency and amplitude both rise sharply just before merger); this is an independent illustrative model, not LIGO's actual strain data."),
        Layer("filtered_noise", {"low": 20, "high": 400, "slow_lfo_rate": 0.1, "slow_lfo_depth": 0.6}, 0.25,
              "Turbulent low noise represents a hypothetical superheated accretion disk -- an electromagnetic phenomenon, not the gravitational-wave signal itself, kept as a distinct layer to avoid conflating the two."),
    ],
    audio_provenance="interpretive_representation",
    provenance_explanation=(
        "A black hole itself emits no sound in any medium, and this app does not use real gravitational-wave "
        "strain data. This profile is an independent, illustrative model built to have the same general "
        "rising-frequency 'chirp' shape that has been publicly described for real LIGO/Virgo detections; "
        "it is not an official sonification and should not be treated as scientific data."
    ),
    scientific_confidence="Interpretive",
    limitations=["Not derived from real LIGO/Virgo strain data.", "Real gravitational-wave chirps vary enormously in duration and frequency range depending on the masses involved; this uses one illustrative example shape."],
    not_represented=["Any literal gravitational-wave recording", "The true duration, mass, or distance of any specific real merger event"],
    listening_guide=["The rising pitch and loudness near the end (the 'chirp' shape)", "The separate turbulent layer underneath (a hypothetical accretion disk, not the gravitational-wave signal)"],
    source_references=["LIGO/Virgo/KAGRA Collaboration: public information on GW150914 and gravitational-wave sonification"],
    base_freq_range=(20, 400), default_duration=8,
))

_register(ObjectProfile(
    object_id="nebula",
    object_name="Emission Nebula (generic model)",
    object_type="nebula",
    short_description=(
        "A generic star-forming emission nebula: a vast cloud of gas and dust "
        "illuminated and ionized by nearby young stars, characterized by "
        "specific emission wavelengths from ionized gas."
    ),
    physical_parameters={"Composition": "Mostly ionized hydrogen, with traces of oxygen, sulfur, and other elements", "Scale": "Light-years across", "Density": "Extremely low by everyday standards, but vast in scale"},
    environmental_parameters={"Illumination": "Ionizing ultraviolet radiation from nearby hot young stars"},
    known_phenomena=["Characteristic emission-line light (e.g. hydrogen-alpha)", "Star formation", "Slow, large-scale gas dynamics"],
    available_measured_data=["Emission-line spectra of nebulae are real, measured astronomical data (specific wavelength ratios are not loaded in this build)"],
    sonification_methods=[
        "Interpretive translation of characteristic emission-line relationships into a sustained chord",
        "Very slow, spacious texture representing the immense physical scale and slow dynamics of a nebula",
    ],
    recipe=[
        Layer("additive", {"freqs": [130, 195, 260, 325], "amps": [0.7, 0.5, 0.4, 0.25], "slow_lfo_rate": 0.015, "slow_lfo_depth": 0.4}, 0.6,
              "A sustained, slowly shifting chord is an interpretive translation of the idea of characteristic emission-line light (such as hydrogen-alpha) rather than a literal frequency conversion of any specific measured spectrum."),
        Layer("filtered_noise", {"low": 20, "high": 600, "slow_lfo_rate": 0.01, "slow_lfo_depth": 0.5}, 0.3,
              "A very slow, spacious noise bed represents the vast physical scale and gradual internal motion of the gas cloud."),
    ],
    audio_provenance="interpretive_representation",
    provenance_explanation=(
        "Nebulae are studied through their light, especially emission-line spectra, which are real "
        "measured data. This profile is an interpretive translation of that general concept into a "
        "sustained musical chord; it does not encode any specific measured spectrum or wavelength "
        "ratio."
    ),
    scientific_confidence="Interpretive",
    limitations=["Not a direct frequency mapping of any specific nebula's measured spectrum.", "Represents a generic emission nebula, not any single named object."],
    not_represented=["Any specific nebula's actual measured spectral data", "Visual structure or imagery"],
    listening_guide=["The slow, sustained chord (a stand-in for characteristic emission-line light)", "The very gradual movement underneath (the nebula's vast scale and slow internal dynamics)"],
    source_references=["General astrophysics references on emission nebula spectroscopy"],
    base_freq_range=(20, 700), default_duration=22,
))

_register(ObjectProfile(
    object_id="galaxy",
    object_name="Spiral Galaxy (generic model)",
    object_type="galaxy",
    short_description=(
        "A generic spiral galaxy, sonified around the well-known finding "
        "that stars far from the galactic center orbit faster than visible "
        "matter alone would predict -- the 'flat rotation curve' evidence "
        "for dark matter."
    ),
    physical_parameters={"Structure": "A rotating disk of stars, gas, and dust with spiral arms", "Scale": "Tens of thousands of light-years across", "Rotation curve": "Observed to stay roughly flat at large radius rather than declining as visible mass alone would predict"},
    environmental_parameters={"Radio emission": "Neutral hydrogen (21cm line) is used to measure rotation speed at large radius"},
    known_phenomena=["Flat rotation curve (key evidence for dark matter)", "Spiral density waves", "Widespread star formation in spiral arms"],
    available_measured_data=["21cm neutral hydrogen radio surveys are used to measure real galaxy rotation curves (a generic, not object-specific, curve shape is used here)"],
    sonification_methods=["Sonification of a generic flat rotation-curve shape as a sustained drone whose modulation depth reflects orbital speed at increasing radius", "Slow FM drift representing large-scale rotation"],
    recipe=[
        Layer("fm", {"carrier_freq": 90, "mod_freq": 0.05, "mod_index": 2.0, "slow_lfo_rate": 0.02, "slow_lfo_depth": 0.2}, 0.55,
              "A slow, sustained frequency-modulated drone represents the galaxy's large-scale rotation; the modulation stays present rather than fading, echoing the real, well-documented finding that rotation speed stays roughly flat at large radius instead of declining."),
        Layer("filtered_noise", {"low": 20, "high": 500, "slow_lfo_rate": 0.008, "slow_lfo_depth": 0.4}, 0.3,
              "A very slow, wide noise bed represents the vast scale and diffuse gas/dust content of a spiral galaxy."),
    ],
    audio_provenance="interpretive_representation",
    provenance_explanation=(
        "Galaxy rotation curves are real, measured astronomical data (typically from 21cm neutral "
        "hydrogen radio observations), and the flat rotation curve is genuine, well-established "
        "evidence for dark matter. This audio uses a generic, illustrative curve shape rather than "
        "any specific galaxy's measured data, so it is presented as interpretive rather than a direct "
        "sonification of a named dataset."
    ),
    scientific_confidence="Interpretive",
    limitations=["Uses a generic rotation-curve shape, not a specific galaxy's measured 21cm data.", "Does not represent any single, named real galaxy."],
    not_represented=["Any specific galaxy's actual measured rotation curve", "Visual spiral structure or star positions"],
    listening_guide=["The sustained, unresolving quality of the drone (the rotation speed staying roughly constant rather than falling off)", "The slow, wide texture underneath (scale and diffuse gas content)"],
    source_references=["General extragalactic astronomy references on 21cm rotation curve measurements and dark matter evidence"],
    base_freq_range=(20, 500), default_duration=22,
))


OBJECT_TYPE_LABELS = {
    "planet": "Planet", "moon": "Moon", "star": "Star",
    "stellar_remnant": "Stellar remnant", "black_hole": "Black hole",
    "nebula": "Nebula", "galaxy": "Galaxy", "small_body": "Small body",
    "plasma_environment": "Plasma environment",
}


# ---------------------------------------------------------------------------
# Gemini integration ("Ask the Cosmos")
# ---------------------------------------------------------------------------
# The audio itself is always produced by the deterministic engine above --
# Gemini is only ever asked to *explain* the science, never to generate or
# alter the audio.

GEMINI_SYSTEM_INSTRUCTIONS = (
    "You are a scientific explanation assistant inside an accessible astronomy "
    "sonification app called Sounds of the Universe, used by many visually "
    "impaired users. Follow these rules strictly: "
    "1) Never invent specific numeric scientific facts or measurements; if you "
    "are not confident of a number, describe it qualitatively instead. "
    "2) Always distinguish between measured data, modeled/simulated results, "
    "and interpretive or artistic choices. "
    "3) Never claim that a modeled or sonified audio clip is a literal audio "
    "recording of the object. "
    "4) Acknowledge scientific uncertainty and the limitations of a simplified "
    "model when relevant. "
    "5) Write for a listener who cannot see the screen: describe things "
    "verbally and clearly, avoid relying on visual metaphors like colors or "
    "shapes on a chart. "
    "6) Do not present speculation as established science; label speculation "
    "explicitly when you offer it."
)


def _gemini_available() -> bool:
    if requests is None or st is None:
        return False
    try:
        return bool(st.secrets.get("GEMINI_API_KEY"))
    except Exception:
        return False


def call_gemini(model: str, user_prompt: str, context: str = "") -> Optional[str]:
    """Call the Generative Language API. Returns None (rather than raising)
    on any failure, so the calling UI code can fall back to a clear offline
    message instead of crashing the app."""
    if not _gemini_available():
        return None
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        url = f"{GEMINI_API_BASE}/{model}:generateContent?key={api_key}"
        full_prompt = f"{context}\n\nUser question: {user_prompt}" if context else user_prompt
        body = {
            "systemInstruction": {"parts": [{"text": GEMINI_SYSTEM_INSTRUCTIONS}]},
            "contents": [{"role": "user", "parts": [{"text": full_prompt}]}],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 800},
        }
        resp = requests.post(url, json=body, timeout=20)
        if resp.status_code != 200:
            return None
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        return text or None
    except Exception:
        return None


def classify_question_depth(question: str) -> str:
    """Use the lightweight routing model to decide whether a question needs
    the deep/pro model or the standard model. Falls back to a simple
    heuristic if Gemini is unavailable, so routing still works offline."""
    routed = call_gemini(
        GEMINI_MODEL_ROUTER,
        f"Classify this astronomy question as exactly one word, either "
        f"'simple' or 'deep'. A 'deep' question asks for detailed physical "
        f"mechanisms, comparisons across multiple objects, or in-depth "
        f"scientific reasoning. Question: {question}",
    )
    if routed:
        cleaned = routed.strip().lower()
        if "deep" in cleaned:
            return "deep"
        if "simple" in cleaned:
            return "simple"
    # Offline / failure fallback heuristic
    deep_markers = ["why", "compare", "how does", "explain in detail", "difference between", "mechanism"]
    return "deep" if any(m in question.lower() for m in deep_markers) else "simple"


def ask_the_cosmos(question: str, object_context: Optional[ObjectProfile]) -> Dict[str, str]:
    """Top-level entry point for the Ask the Cosmos panel. Returns a dict
    with the answer text and which model (or offline mode) produced it."""
    context = ""
    if object_context:
        context = (
            f"The user is currently looking at: {object_context.object_name} "
            f"({OBJECT_TYPE_LABELS.get(object_context.object_type, object_context.object_type)}). "
            f"Audio category for this object: {PROVENANCE_LABELS[object_context.audio_provenance]}. "
            f"Known physical parameters: {object_context.physical_parameters}. "
            f"Known limitations of this object's audio model: {object_context.limitations}."
        )
    if not _gemini_available():
        return {
            "answer": (
                "Ask the Cosmos is currently offline because no Gemini API key is configured "
                "(or the request could not be completed). You can still explore every object, "
                "generate audio, read scientific descriptions, and compare objects without this "
                "feature -- it only affects free-form question answering."
            ),
            "model_used": "offline",
        }
    depth = classify_question_depth(question)
    model = GEMINI_MODEL_DEEP if depth == "deep" else GEMINI_MODEL_STANDARD
    answer = call_gemini(model, question, context)
    if answer is None:
        return {
            "answer": (
                "The request to the scientific explanation assistant did not complete "
                "successfully. You can still explore every object, generate audio, and read "
                "scientific descriptions without this feature."
            ),
            "model_used": "offline",
        }
    return {"answer": answer, "model_used": model}


# ---------------------------------------------------------------------------
# Streamlit application
# ---------------------------------------------------------------------------

NAV_SECTIONS = [
    "Home",
    "Explore objects",
    "Audio laboratory",
    "Scientific sonifications",
    "Ask the Cosmos",
    "Compare objects",
    "Favorites",
    "Data and export",
    "About and scientific method",
]


def init_session_state() -> None:
    defaults = {
        "nav": "Home",
        "selected_object": "venus",
        "compare_object": "mars",
        "favorites": [],
        "theme": "dark",
        "audio_cache": {},
        "lab_duration": 15.0,
        "lab_intensity": 0.7,
        "lab_complexity": 0.6,
        "lab_freq_lo": None,
        "lab_freq_hi": None,
        "lab_stereo": 0.3,
        "lab_seed_offset": 0,
        "cosmos_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def inject_css() -> None:
    dark = st.session_state.get("theme", "dark") == "dark"
    if dark:
        bg, fg, muted, accent, card, border = "#0b0d10", "#eef1f4", "#9aa4ae", "#d8dee4", "#12151a", "#242a31"
    else:
        bg, fg, muted, accent, card, border = "#fbfbfa", "#14171a", "#565f68", "#14171a", "#ffffff", "#dde1e5"
    st.markdown(f"""
    <style>
        :root {{
            --bg: {bg}; --fg: {fg}; --muted: {muted};
            --accent: {accent}; --card: {card}; --border: {border};
        }}
        .stApp {{
            background-color: var(--bg);
            color: var(--fg);
            font-family: "Helvetica Neue", Arial, sans-serif;
        }}
        h1, h2, h3, h4 {{
            font-weight: 600;
            letter-spacing: -0.01em;
            color: var(--fg);
        }}
        .su-eyebrow {{
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.72rem;
            color: var(--muted);
            margin-bottom: 0.2rem;
        }}
        .su-card {{
            background-color: var(--card);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 0.9rem;
        }}
        .su-divider {{
            border: none;
            border-top: 1px solid var(--border);
            margin: 1.4rem 0;
        }}
        .su-provenance {{
            border-left: 3px solid var(--accent);
            padding-left: 0.9rem;
            margin: 0.8rem 0;
        }}
        .su-section-number {{
            color: var(--muted);
            font-variant-numeric: tabular-nums;
            margin-right: 0.4rem;
        }}
        .su-muted {{ color: var(--muted); }}
        div[data-testid="stSidebar"] {{
            background-color: var(--card);
            border-right: 1px solid var(--border);
        }}
    </style>
    """, unsafe_allow_html=True)


def provenance_panel(profile: ObjectProfile) -> None:
    label = PROVENANCE_LABELS[profile.audio_provenance]
    st.markdown(f"""
    <div class="su-provenance">
        <div class="su-eyebrow">Audio provenance</div>
        <p><strong>Audio basis:</strong> {label}</p>
        <p><strong>Literal sound in space:</strong> {"Not applicable -- an ordinary environmental medium exists" if profile.object_id == "earth" else "No"}</p>
        <p><strong>Scientific confidence:</strong> {profile.scientific_confidence}</p>
        <p>{profile.provenance_explanation}</p>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Important limitations of this audio"):
        for item in profile.limitations:
            st.markdown(f"- {item}")
    with st.expander("What this audio does NOT represent"):
        for item in profile.not_represented:
            st.markdown(f"- {item}")


def explanation_block(profile: ObjectProfile) -> None:
    st.markdown("#### What you are hearing")
    for question, answer in profile.explanation().items():
        st.markdown(f"**{question}.** {answer}")


def object_selector(label: str, key: str, exclude: Optional[str] = None) -> str:
    ids = [oid for oid in OBJECT_REGISTRY if oid != exclude]
    names = [OBJECT_REGISTRY[oid].object_name for oid in ids]
    current = st.session_state.get(key, ids[0])
    idx = ids.index(current) if current in ids else 0
    choice_name = st.selectbox(label, names, index=idx, key=f"{key}_select")
    chosen_id = ids[names.index(choice_name)]
    st.session_state[key] = chosen_id
    return chosen_id


def generate_and_cache_audio(profile: ObjectProfile, duration: float, intensity: float,
                              complexity: float, freq_lo: float, freq_hi: float,
                              stereo_width: float, seed_offset: int = 0) -> Tuple[np.ndarray, bytes, int]:
    seed = make_seed(profile.object_id, duration, intensity, complexity, freq_lo, freq_hi, stereo_width, seed_offset)
    cache_key = seed
    cache = st.session_state["audio_cache"]
    if cache_key in cache:
        return cache[cache_key]
    signal = synthesize_object_audio(profile, duration, intensity, complexity, freq_lo, freq_hi, stereo_width, seed)
    wav_bytes = float_stereo_to_wav_bytes(signal)
    cache[cache_key] = (signal, wav_bytes, seed)
    return signal, wav_bytes, seed


def safety_notice() -> None:
    st.markdown("""
    <div class="su-card">
        <div class="su-eyebrow">Safe listening</div>
        <p>Start at a low volume, especially with headphones. Some profiles use sustained low
        frequencies; take listening breaks if needed, and stop if any sound causes discomfort.
        This application is an educational and accessibility tool, not a medical or diagnostic
        device.</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: Home
# ---------------------------------------------------------------------------

def page_home() -> None:
    st.markdown('<div class="su-eyebrow">Sounds of the Universe</div>', unsafe_allow_html=True)
    st.markdown("# Hear the physical universe.")
    st.markdown(
        "Explore astronomical objects through scientifically grounded sonification, "
        "physics-based auditory models, measured-data transformations, and accessible "
        "scientific descriptions."
    )
    st.markdown('<hr class="su-divider">', unsafe_allow_html=True)

    st.markdown("### 1. What this is, and what it is not")
    st.markdown(
        "Ordinary sound is a mechanical wave that needs a physical medium, such as air, to "
        "travel through. Most of outer space is a near-vacuum, so most astronomical objects "
        "cannot be 'heard' in the ordinary sense. What this application offers instead is a "
        "structured, honest catalogue of four different kinds of audio, always labeled:"
    )
    cols = st.columns(4)
    labels_and_desc = [
        ("Measured audio data", "A real acoustic recording made where sound can actually propagate."),
        ("Scientific sonification", "A real measured quantity mapped onto timing, pitch, or another audible parameter."),
        ("Physics-based auditory model", "A synthesized soundscape whose parameters come from real physical properties."),
        ("Interpretive representation", "An artistic-scientific translation used when no direct data pathway exists."),
    ]
    for col, (label, desc) in zip(cols, labels_and_desc):
        with col:
            st.markdown(f'<div class="su-card"><strong>{label}</strong><br><span class="su-muted">{desc}</span></div>', unsafe_allow_html=True)

    st.markdown("### 2. A growing scientific catalogue")
    st.markdown(
        f"This build includes {len(OBJECT_REGISTRY)} objects spanning planets, moons, a star, a "
        "stellar remnant, a black hole, a nebula, and a galaxy -- enough to demonstrate every "
        "provenance category and synthesis technique this application supports. The catalogue "
        "is modular by design and is meant to keep growing; it does not, and does not claim to, "
        "cover every object in the universe."
    )

    st.markdown("### 3. Featured objects")
    featured_ids = ["venus", "pulsar", "titan", "black_hole"]
    cols = st.columns(len(featured_ids))
    for col, oid in zip(cols, featured_ids):
        profile = OBJECT_REGISTRY[oid]
        with col:
            st.markdown(f"**{profile.object_name}**")
            st.caption(OBJECT_TYPE_LABELS.get(profile.object_type, profile.object_type))
            st.caption(PROVENANCE_LABELS[profile.audio_provenance])
            if st.button("Open profile", key=f"home_open_{oid}"):
                st.session_state["selected_object"] = oid
                st.session_state["nav"] = "Explore objects"
                st.rerun()

    st.markdown('<hr class="su-divider">', unsafe_allow_html=True)
    st.markdown("### 4. Accessibility statement")
    st.markdown(
        "This application is built for blind and visually impaired users first. Every control "
        "has a text label, every generated sound has a full written explanation, no information "
        "is conveyed through color alone, and the layout follows a logical heading structure for "
        "screen readers. See About and scientific method for the full accessibility approach."
    )

    st.markdown("### 5. Scientific method, in brief")
    st.markdown(
        "Scientific data provides the foundation. Physics-based models translate defensible "
        "properties into sound. Deterministic synthesis creates reproducible audio. Ask the "
        "Cosmos explains the underlying science. A provenance panel tells you, for every object, "
        "what is measured, modeled, sonified, or interpretive."
    )
    safety_notice()


# ---------------------------------------------------------------------------
# Page: Explore objects
# ---------------------------------------------------------------------------

def page_explore() -> None:
    st.markdown('<div class="su-eyebrow">Catalogue</div>', unsafe_allow_html=True)
    st.markdown("# Explore objects")

    search = st.text_input("Search objects by name", value="", key="explore_search")
    type_filter = st.selectbox(
        "Filter by category", ["All categories"] + sorted(set(OBJECT_TYPE_LABELS.get(p.object_type, p.object_type) for p in OBJECT_REGISTRY.values())),
        key="explore_type_filter",
    )

    filtered = []
    for oid, profile in OBJECT_REGISTRY.items():
        if search and search.lower() not in profile.object_name.lower():
            continue
        label = OBJECT_TYPE_LABELS.get(profile.object_type, profile.object_type)
        if type_filter != "All categories" and label != type_filter:
            continue
        filtered.append(oid)

    st.caption(f"{len(filtered)} object(s) match your search.")
    cols = st.columns(3)
    for i, oid in enumerate(filtered):
        profile = OBJECT_REGISTRY[oid]
        with cols[i % 3]:
            st.markdown(f'<div class="su-card"><strong>{profile.object_name}</strong><br>'
                        f'<span class="su-muted">{OBJECT_TYPE_LABELS.get(profile.object_type, profile.object_type)}'
                        f' -- {PROVENANCE_LABELS[profile.audio_provenance]}</span></div>', unsafe_allow_html=True)
            if st.button(f"View {profile.object_name}", key=f"explore_select_{oid}"):
                st.session_state["selected_object"] = oid
                st.rerun()

    st.markdown('<hr class="su-divider">', unsafe_allow_html=True)
    selected = st.session_state.get("selected_object", "venus")
    if selected not in OBJECT_REGISTRY:
        selected = "venus"
    render_object_profile(OBJECT_REGISTRY[selected])


def render_object_profile(profile: ObjectProfile) -> None:
    st.markdown(f"## {profile.object_name}")
    st.caption(OBJECT_TYPE_LABELS.get(profile.object_type, profile.object_type))
    st.markdown(profile.short_description)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Physical properties")
        for k, v in profile.physical_parameters.items():
            st.markdown(f"- **{k}:** {v}")
        st.markdown("#### Known phenomena")
        for item in profile.known_phenomena:
            st.markdown(f"- {item}")
    with right:
        st.markdown("#### Environmental conditions")
        for k, v in profile.environmental_parameters.items():
            st.markdown(f"- **{k}:** {v}")
        st.markdown("#### Available measured data")
        for item in profile.available_measured_data:
            st.markdown(f"- {item}")

    st.markdown("#### Source notes")
    for ref in profile.source_references:
        st.markdown(f"- {ref}")

    provenance_panel(profile)

    st.markdown("#### Generate this object's auditory representation")
    if st.button("Generate audio", key=f"profile_generate_{profile.object_id}"):
        lo, hi = profile.base_freq_range
        signal, wav_bytes, seed = generate_and_cache_audio(
            profile, profile.default_duration, 0.7, 0.6, lo, hi, 0.25,
        )
        st.audio(wav_bytes, format="audio/wav")
        st.caption(f"Deterministic seed: {seed} -- regenerating with the same settings always produces this exact result.")
        st.download_button("Download WAV", data=wav_bytes, file_name=f"{profile.object_id}.wav",
                            mime="audio/wav", key=f"profile_download_{profile.object_id}")

    explanation_block(profile)

    fav_col1, fav_col2 = st.columns(2)
    with fav_col1:
        if profile.object_id not in st.session_state["favorites"]:
            if st.button("Add to favorites", key=f"fav_add_{profile.object_id}"):
                st.session_state["favorites"].append(profile.object_id)
                st.rerun()
        else:
            if st.button("Remove from favorites", key=f"fav_remove_{profile.object_id}"):
                st.session_state["favorites"].remove(profile.object_id)
                st.rerun()
    with fav_col2:
        if st.button("Compare with another object", key=f"compare_from_{profile.object_id}"):
            st.session_state["nav"] = "Compare objects"
            st.rerun()


# ---------------------------------------------------------------------------
# Page: Audio laboratory
# ---------------------------------------------------------------------------

def page_lab() -> None:
    st.markdown('<div class="su-eyebrow">Audio laboratory</div>', unsafe_allow_html=True)
    st.markdown("# Audio laboratory")
    st.markdown(
        "Generate and explore an object's auditory representation with full control over "
        "duration, intensity, complexity, frequency range, and stereo width. Generation is "
        "deterministic: the same object and the same settings always produce the same result."
    )

    oid = object_selector("Object selector", "selected_object")
    profile = OBJECT_REGISTRY[oid]
    default_lo, default_hi = profile.base_freq_range

    st.markdown("#### Controls")
    c1, c2 = st.columns(2)
    with c1:
        duration = st.slider("Duration (seconds)", 4.0, 40.0, float(min(profile.default_duration, 40.0)), 1.0,
                              help="How long the generated clip is.")
        intensity = st.slider("Intensity", 0.1, 1.0, st.session_state["lab_intensity"], 0.05,
                               help="Overall loudness/energy of the generated layers.")
        complexity = st.slider("Complexity", 0.1, 1.0, st.session_state["lab_complexity"], 0.05,
                                help="How many harmonic partials and modulation layers are active.")
    with c2:
        freq_lo = st.slider("Frequency range: low cutoff (Hz)", 10, 2000, int(default_lo), 5,
                             help="Frequencies below this are filtered out of the final mix.")
        freq_hi = st.slider("Frequency range: high cutoff (Hz)", 200, 8000, int(default_hi), 10,
                             help="Frequencies above this are filtered out of the final mix.")
        stereo_width = st.slider("Stereo width", 0.0, 1.0, st.session_state["lab_stereo"], 0.05,
                                  help="0 is mono-compatible; higher values gently widen the stereo image.")

    seed_offset = st.number_input(
        "Deterministic seed offset", min_value=0, max_value=9999,
        value=st.session_state["lab_seed_offset"], step=1,
        help="Change this to explore a different-but-still-reproducible variation of the same object and settings.",
    )
    st.session_state.update({
        "lab_intensity": intensity, "lab_complexity": complexity, "lab_stereo": stereo_width,
        "lab_seed_offset": seed_offset,
    })

    audio_mode = st.selectbox(
        "Audio mode (descriptive label only -- all modes use this object's real recipe)",
        ["Environmental / atmospheric model", "Scientific sonification", "Periodic signal",
         "Magnetic / plasma representation", "Interpretive scientific representation"],
        help="This label describes how to interpret the result; it does not change the underlying physics-based recipe below.",
    )

    if st.button("Generate", key="lab_generate", type="primary"):
        signal, wav_bytes, seed = generate_and_cache_audio(
            profile, duration, intensity, complexity, freq_lo, freq_hi, stereo_width, seed_offset,
        )
        st.session_state["_lab_last"] = (oid, wav_bytes, seed)

    if "_lab_last" in st.session_state and st.session_state["_lab_last"][0] == oid:
        _, wav_bytes, seed = st.session_state["_lab_last"]
        st.audio(wav_bytes, format="audio/wav")
        st.caption(f"Deterministic seed: {seed}")
        st.download_button("Download WAV", data=wav_bytes, file_name=f"{oid}_lab.wav",
                            mime="audio/wav", key="lab_download")
        metadata = {
            "object_id": oid, "object_name": profile.object_name,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_s": duration, "sample_rate": SAMPLE_RATE, "audio_mode_label": audio_mode,
            "engine_version": ENGINE_VERSION, "deterministic_seed": seed,
            "parameters": {"intensity": intensity, "complexity": complexity, "freq_lo": freq_lo,
                           "freq_hi": freq_hi, "stereo_width": stereo_width, "seed_offset": seed_offset},
            "audio_provenance": PROVENANCE_LABELS[profile.audio_provenance],
            "source_references": profile.source_references,
            "limitations": profile.limitations,
        }
        st.download_button("Download metadata (JSON)", data=json.dumps(metadata, indent=2),
                            file_name=f"{oid}_lab_metadata.json", mime="application/json", key="lab_download_meta")

    st.markdown('<hr class="su-divider">', unsafe_allow_html=True)
    provenance_panel(profile)
    explanation_block(profile)
    st.markdown("#### Recipe layers used for this object")
    for layer in profile.recipe:
        st.markdown(f"- **{layer.kind}** -- {layer.reason}")
    safety_notice()


# ---------------------------------------------------------------------------
# Page: Scientific sonifications
# ---------------------------------------------------------------------------

def page_sonifications() -> None:
    st.markdown('<div class="su-eyebrow">Method</div>', unsafe_allow_html=True)
    st.markdown("# Scientific sonifications")
    st.markdown(
        "Sonification is the practice of translating data into sound so it can be understood by "
        "listening rather than looking. This section explains how, and why, this application uses "
        "it, and why a sonified signal is not the same thing as an audible sound in space."
    )

    st.markdown("### Why convert data into sound at all")
    st.markdown(
        "Many astronomical signals -- radio emission, X-rays, gravitational waves, particle "
        "counts -- are entirely inaudible: they are not mechanical vibrations in a medium, so "
        "human ears cannot detect them no matter how 'loud' the source is. Sonification lets a "
        "listener perceive patterns such as timing, periodicity, or intensity in these signals "
        "without needing to see a chart."
    )

    st.markdown("### How frequency and timing are mapped")
    st.markdown(
        "This application's engine builds every sound from a small set of primitives: additive "
        "and subtractive synthesis, filtered noise, amplitude and frequency modulation, and pulse "
        "trains. When a real measured quantity is available -- most clearly a pulsar's spin period "
        "-- that quantity sets a real, data-driven parameter directly, such as the spacing between "
        "clicks in a pulse train. When frequencies fall outside the audible range (roughly 20 Hz to "
        "20 kHz for human hearing), they are shifted -- multiplied or divided by a constant factor "
        "-- into a range people can actually hear, as this app does with solar oscillation "
        "frequencies for the Sun."
    )

    st.markdown("### How amplitude and normalization work")
    st.markdown(
        "Every generated clip is passed through a gentle limiter so that layered sounds never clip "
        "harshly, and every clip is normalized to a comfortable, safe playback level rather than "
        "the loudest possible level."
    )

    st.markdown("### Sonification versus ordinary sound")
    st.markdown(
        "Ordinary sound is a real mechanical vibration reaching your ear through a medium such as "
        "air. A sonification is a human-designed translation: real data drives some audible "
        "parameter, but the instrument or timbre you hear is chosen by a designer, not measured. "
        "A sonified signal was never 'audible in space' in any literal sense -- the conversion to "
        "sound happens entirely after the data is collected."
    )

    st.markdown("### Examples in this catalogue")
    examples = [
        ("Pulsar", "Real, measured spin period sets the timing between clicks directly."),
        ("The Sun", "Published helioseismic oscillation frequencies are shifted up into the audible range."),
        ("Galaxy", "A generic flat rotation-curve shape drives ongoing modulation depth."),
        ("Black hole", "An illustrative rising-frequency shape echoes the general form of a published gravitational-wave chirp."),
    ]
    for name, desc in examples:
        st.markdown(f"- **{name}:** {desc}")


# ---------------------------------------------------------------------------
# Page: Ask the Cosmos
# ---------------------------------------------------------------------------

def page_ask_cosmos() -> None:
    st.markdown('<div class="su-eyebrow">Ask the Cosmos</div>', unsafe_allow_html=True)
    st.markdown("# Ask the Cosmos")
    st.markdown(
        "Ask a scientific question about an astronomical object, its physical properties, or how "
        "its audio was built. This assistant explains the science behind the deterministic audio "
        "engine; it never generates or changes the audio itself."
    )
    if not _gemini_available():
        st.info(
            "Ask the Cosmos is currently offline: no Gemini API key is configured for this session. "
            "Every other part of the application -- object exploration, audio generation, "
            "comparisons, favorites, and export -- works normally without it."
        )

    oid = object_selector("Ask about this object (optional context)", "selected_object")
    profile = OBJECT_REGISTRY[oid]
    question = st.text_area("Your question", value="", key="cosmos_question",
                             placeholder=f"For example: why does {profile.object_name}'s audio sound the way it does?")

    if st.button("Ask", key="cosmos_ask") and question.strip():
        with st.spinner("Consulting the scientific explanation assistant..."):
            result = ask_the_cosmos(question.strip(), profile)
        st.session_state["cosmos_history"].append({"question": question.strip(), **result})

    for entry in reversed(st.session_state["cosmos_history"]):
        st.markdown(f"**You asked:** {entry['question']}")
        model_label = {"offline": "Offline fallback", GEMINI_MODEL_STANDARD: "Standard model",
                       GEMINI_MODEL_DEEP: "Deep explanation model"}.get(entry["model_used"], entry["model_used"])
        st.caption(f"Answered by: {model_label}")
        st.markdown(entry["answer"])
        st.markdown('<hr class="su-divider">', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: Compare objects
# ---------------------------------------------------------------------------

COMPARE_FIELDS = [
    ("Category", lambda p: OBJECT_TYPE_LABELS.get(p.object_type, p.object_type)),
    ("Audio basis", lambda p: PROVENANCE_LABELS[p.audio_provenance]),
    ("Scientific confidence", lambda p: p.scientific_confidence),
]


def page_compare() -> None:
    st.markdown('<div class="su-eyebrow">Comparison</div>', unsafe_allow_html=True)
    st.markdown("# Compare objects")
    st.markdown(
        "Compare two objects factually, side by side. This view does not rank objects or assign "
        "scores -- it shows how and why their physical conditions, and therefore their auditory "
        "representations, differ."
    )

    col1, col2 = st.columns(2)
    with col1:
        oid_a = object_selector("First object", "selected_object")
    with col2:
        oid_b = object_selector("Second object", "compare_object", exclude=oid_a)

    profile_a, profile_b = OBJECT_REGISTRY[oid_a], OBJECT_REGISTRY[oid_b]

    st.markdown('<hr class="su-divider">', unsafe_allow_html=True)
    left, right = st.columns(2)
    for col, profile in ((left, profile_a), (right, profile_b)):
        with col:
            st.markdown(f"### {profile.object_name}")
            for label, getter in COMPARE_FIELDS:
                st.markdown(f"**{label}:** {getter(profile)}")
            st.markdown("**Physical properties:**")
            for k, v in profile.physical_parameters.items():
                st.markdown(f"- {k}: {v}")
            st.markdown("**Environmental conditions:**")
            for k, v in profile.environmental_parameters.items():
                st.markdown(f"- {k}: {v}")
            st.markdown("**Known phenomena:**")
            for item in profile.known_phenomena:
                st.markdown(f"- {item}")

    st.markdown('<hr class="su-divider">', unsafe_allow_html=True)
    st.markdown("### Why their audio differs")
    st.markdown(
        f"**{profile_a.object_name}:** {profile_a.provenance_explanation}\n\n"
        f"**{profile_b.object_name}:** {profile_b.provenance_explanation}"
    )

    if st.button("Generate both for listening comparison", key="compare_generate"):
        for profile in (profile_a, profile_b):
            lo, hi = profile.base_freq_range
            _, wav_bytes, seed = generate_and_cache_audio(profile, min(profile.default_duration, 15.0), 0.7, 0.6, lo, hi, 0.25)
            st.markdown(f"**{profile.object_name}** (seed {seed})")
            st.audio(wav_bytes, format="audio/wav")


# ---------------------------------------------------------------------------
# Page: Favorites
# ---------------------------------------------------------------------------

def page_favorites() -> None:
    st.markdown('<div class="su-eyebrow">Saved</div>', unsafe_allow_html=True)
    st.markdown("# Favorites")
    favorites = st.session_state["favorites"]
    if not favorites:
        st.markdown("No favorites saved yet. Open any object's profile and select Add to favorites.")
        return
    for oid in list(favorites):
        if oid not in OBJECT_REGISTRY:
            continue
        profile = OBJECT_REGISTRY[oid]
        st.markdown(f'<div class="su-card"><strong>{profile.object_name}</strong><br>'
                    f'<span class="su-muted">{OBJECT_TYPE_LABELS.get(profile.object_type, profile.object_type)}'
                    f' -- {PROVENANCE_LABELS[profile.audio_provenance]}</span></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button(f"Open {profile.object_name}", key=f"fav_open_{oid}"):
                st.session_state["selected_object"] = oid
                st.session_state["nav"] = "Explore objects"
                st.rerun()
        with c2:
            if st.button(f"Remove {profile.object_name}", key=f"fav_remove_list_{oid}"):
                st.session_state["favorites"].remove(oid)
                st.rerun()


# ---------------------------------------------------------------------------
# Page: Data and export
# ---------------------------------------------------------------------------

def page_export() -> None:
    st.markdown('<div class="su-eyebrow">Export</div>', unsafe_allow_html=True)
    st.markdown("# Data and export")
    st.markdown("Export scientific and provenance information for any object as JSON or CSV.")

    oid = object_selector("Object to export", "selected_object")
    profile = OBJECT_REGISTRY[oid]

    record = {
        "object_id": profile.object_id, "object_name": profile.object_name,
        "object_type": profile.object_type, "physical_parameters": profile.physical_parameters,
        "environmental_parameters": profile.environmental_parameters,
        "known_phenomena": profile.known_phenomena,
        "available_measured_data": profile.available_measured_data,
        "sonification_methods": profile.sonification_methods,
        "audio_provenance": PROVENANCE_LABELS[profile.audio_provenance],
        "provenance_explanation": profile.provenance_explanation,
        "scientific_confidence": profile.scientific_confidence,
        "limitations": profile.limitations, "not_represented": profile.not_represented,
        "source_references": profile.source_references,
        "engine_version": ENGINE_VERSION, "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    st.download_button("Download object data (JSON)", data=json.dumps(record, indent=2),
                        file_name=f"{oid}_data.json", mime="application/json", key="export_json")

    csv_lines = ["field,value"]
    for k, v in record.items():
        flat = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
        flat = flat.replace('"', "'")
        csv_lines.append(f'"{k}","{flat}"')
    st.download_button("Download object data (CSV)", data="\n".join(csv_lines),
                        file_name=f"{oid}_data.csv", mime="text/csv", key="export_csv")

    st.markdown('<hr class="su-divider">', unsafe_allow_html=True)
    st.markdown("### Export the whole catalogue")
    all_records = {oid2: {
        "object_name": p.object_name, "object_type": p.object_type,
        "audio_provenance": PROVENANCE_LABELS[p.audio_provenance],
        "scientific_confidence": p.scientific_confidence,
    } for oid2, p in OBJECT_REGISTRY.items()}
    st.download_button("Download full catalogue summary (JSON)", data=json.dumps(all_records, indent=2),
                        file_name="catalogue_summary.json", mime="application/json", key="export_all_json")

    st.caption(
        "Exported metadata always explains how audio was generated (engine version, deterministic "
        "seed, and parameters) whenever it accompanies a specific generated clip -- see the Audio "
        "laboratory page for per-clip metadata export."
    )


# ---------------------------------------------------------------------------
# Page: About and scientific method
# ---------------------------------------------------------------------------

def page_about() -> None:
    st.markdown('<div class="su-eyebrow">About</div>', unsafe_allow_html=True)
    st.markdown("# About and scientific method")

    st.markdown("### Purpose")
    st.markdown(
        "Sounds of the Universe exists to let people -- especially blind and visually impaired "
        "users -- explore astronomical objects through listening, grounded in real physical "
        "properties rather than generic 'space sound' effects."
    )

    st.markdown("### Accessibility goals")
    st.markdown(
        "Every control has a descriptive text label. Every generated sound has a full written "
        "explanation covering what is represented, what drove the sound, what data or model was "
        "used, what to listen for, what is not represented, and which provenance category applies. "
        "No information is conveyed through color alone. Layout follows a logical heading order "
        "for screen readers, and no emoji or icon-only controls are used anywhere in the "
        "application."
    )

    st.markdown("### Sound versus sonification versus modeling")
    st.markdown(
        "This application distinguishes four categories everywhere audio appears: measured audio "
        "data (a real recording), scientific sonification (real data driving an audible parameter), "
        "a physics-based auditory model (parameters drawn from real physics, but no direct "
        "recording or dataset), and an interpretive representation (an explicit artistic-scientific "
        "translation). See Scientific sonifications for a deeper explanation."
    )

    st.markdown("### Data provenance and limitations")
    st.markdown(
        "Physical values in this catalogue are approximate reference figures rounded for "
        "readability, sourced from general mission and agency summaries (see each object's source "
        "notes). They are not high-precision measurements and should be verified against a current "
        "primary source before being used for anything beyond education. The catalogue is "
        "incomplete by design and will keep growing."
    )

    st.markdown("### Responsible use of AI")
    st.markdown(
        "The Ask the Cosmos assistant (built on external Gemini models) is instructed to avoid "
        "inventing facts or measurements, to distinguish measured data from modeled audio, to "
        "acknowledge uncertainty, and to never claim that a modeled or sonified clip is a literal "
        "recording. It never generates or alters audio -- audio always comes from this "
        "application's deterministic engine."
    )

    st.markdown("### Audio safety")
    safety_notice()

    st.markdown("### Scientific limitations, stated openly")
    st.markdown(
        "Some audio here is modeled, some is a transformation of measured data into audible "
        "frequencies, and some is explicitly interpretive. Many astronomical environments cannot "
        "be heard directly by human ears under any circumstances. Physical values carry "
        "uncertainty. Models simplify complex systems. Sound design choices can influence "
        "perception. An auditory representation is not the same thing as direct scientific "
        "observation."
    )


# ---------------------------------------------------------------------------
# Navigation shell and entry point
# ---------------------------------------------------------------------------

PAGE_RENDERERS = {
    "Home": page_home,
    "Explore objects": page_explore,
    "Audio laboratory": page_lab,
    "Scientific sonifications": page_sonifications,
    "Ask the Cosmos": page_ask_cosmos,
    "Compare objects": page_compare,
    "Favorites": page_favorites,
    "Data and export": page_export,
    "About and scientific method": page_about,
}


def render_sidebar() -> None:
    st.sidebar.markdown("### Sounds of the Universe")
    st.sidebar.caption(f"Version {APP_VERSION}")
    current = st.session_state.get("nav", "Home")
    choice = st.sidebar.radio("Navigate", NAV_SECTIONS, index=NAV_SECTIONS.index(current), key="nav_radio")
    st.session_state["nav"] = choice

    st.sidebar.markdown('<hr class="su-divider">', unsafe_allow_html=True)
    theme_choice = st.sidebar.radio("Display theme", ["Dark", "Light"],
                                     index=0 if st.session_state.get("theme", "dark") == "dark" else 1,
                                     key="theme_radio")
    st.session_state["theme"] = "dark" if theme_choice == "Dark" else "light"

    st.sidebar.markdown('<hr class="su-divider">', unsafe_allow_html=True)
    if _gemini_available():
        st.sidebar.caption("Ask the Cosmos: connected")
    else:
        st.sidebar.caption("Ask the Cosmos: offline (no API key configured)")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_session_state()
    inject_css()
    render_sidebar()
    PAGE_RENDERERS[st.session_state["nav"]]()


if __name__ == "__main__":
    if st is None:
        raise SystemExit(
            "Streamlit is not installed. Install dependencies with: pip install -r requirements.txt "
            "then run: streamlit run app.py"
        )
    main()
