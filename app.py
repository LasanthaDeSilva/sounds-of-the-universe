import io
import math
import wave
import hashlib
import csv
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


APP_NAME = "Sounds of the Universe"
APP_VERSION = "0.2.0"

GEMINI_PRO = "gemini-3.1-pro-preview"
GEMINI_FLASH = "gemini-3.6-flash"
GEMINI_LITE = "gemini-3.5-flash-lite"

SAMPLE_RATE = 44100
AUDIO_DURATION = 12.0


st.set_page_config(
    page_title=APP_NAME,
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .main {
        max-width: 1400px;
        margin: auto;
    }
    .stButton > button {
        border-radius: 10px;
        min-height: 42px;
    }
    .provenance {
        padding: 0.9rem 1rem;
        border-radius: 8px;
        background: rgba(128,128,128,0.08);
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@dataclass(frozen=True)
class AstroObject:
    object_id: str
    name: str
    category: str
    subtype: str
    description: str
    temperature_k: Optional[float]
    pressure_bar: Optional[float]
    atmosphere: str
    composition: str
    environment: str
    rotation_period_hours: Optional[float]
    magnetic_environment: str
    known_phenomena: Tuple[str, ...]
    audio_basis: str
    scientific_confidence: str
    source_notes: str


OBJECTS: Dict[str, AstroObject] = {
    "sun": AstroObject(
        "sun", "The Sun", "Star", "G-type main-sequence star",
        "The Sun is the star at the center of our Solar System. Its visible surface is the photosphere, above a much hotter interior and below a dynamic atmosphere.",
        5772, None, "Hydrogen- and helium-dominated plasma",
        "Approximately hydrogen and helium by mass, with heavier elements present",
        "Photosphere, chromosphere, corona, solar wind", 609.6,
        "Strong, dynamic magnetic field",
        ("solar convection", "sunspots", "flares", "coronal mass ejections", "solar wind", "coronal activity"),
        "Physics-informed sonification of solar parameters; not ordinary audible sound recorded in space",
        "High for physical parameters; audio mapping is interpretive",
        "Rounded reference values suitable for educational modeling.",
    ),
    "mercury": AstroObject(
        "mercury", "Mercury", "Planet", "Terrestrial planet",
        "Mercury is the innermost planet. It has an extremely thin exosphere and large temperature differences between illuminated and dark regions.",
        440, 0.0, "Extremely tenuous exosphere",
        "Surface dominated by silicate rock and metal; exosphere contains trace species",
        "Airless rocky surface and exosphere", 1407.6,
        "Weak intrinsic magnetic field",
        ("extreme day-night temperature variation", "impact cratering", "solar-wind interaction"),
        "Physics-based interpretive environmental sonification",
        "High for broad environmental properties",
        "Mercury's exosphere is far too tenuous to behave like an Earth-like atmosphere.",
    ),
    "venus": AstroObject(
        "venus", "Venus", "Planet", "Terrestrial planet",
        "Venus has an exceptionally dense carbon-dioxide atmosphere, very high surface temperature and extreme surface pressure.",
        737, 92.0, "Very dense carbon-dioxide-dominated atmosphere",
        "Mostly CO2 with nitrogen and trace gases",
        "Dense hot atmosphere with sulfuric-acid cloud layers", 5832.5,
        "No strong intrinsic global magnetic field; interacts with solar wind",
        ("super-rotation", "sulfuric-acid clouds", "extreme greenhouse warming", "high surface pressure"),
        "Physics-informed atmospheric sonification",
        "High for physical parameters; exact audible mapping is interpretive",
        "Approximate mean surface temperature and pressure values.",
    ),
    "earth": AstroObject(
        "earth", "Earth", "Planet", "Terrestrial planet",
        "Earth has a nitrogen- and oxygen-rich atmosphere, liquid surface water, weather systems and an active magnetosphere.",
        288, 1.01325, "Nitrogen- and oxygen-dominated atmosphere",
        "Approximately 78% nitrogen and 21% oxygen by volume, plus trace gases",
        "Atmosphere, oceans, land, weather and magnetosphere", 23.934,
        "Strong global magnetosphere",
        ("weather", "lightning", "ocean waves", "aurorae", "earthquakes"),
        "Environmental reference and physics-informed sonification",
        "Very high for broad physical properties",
        "Representative global values.",
    ),
    "mars": AstroObject(
        "mars", "Mars", "Planet", "Terrestrial planet",
        "Mars has a cold, dry surface and a very thin carbon-dioxide-dominated atmosphere. Dust is an important environmental component.",
        210, 0.006, "Very thin carbon-dioxide-dominated atmosphere",
        "Mostly CO2 with nitrogen and argon",
        "Cold dusty surface with thin atmosphere", 24.623,
        "No present global intrinsic field; localized crustal magnetism",
        ("dust devils", "regional dust storms", "global dust storms", "marsquakes", "polar ice"),
        "Thin-atmosphere physics-informed sonification",
        "High for broad physical properties",
        "Representative mean environmental values.",
    ),
    "jupiter": AstroObject(
        "jupiter", "Jupiter", "Planet", "Gas giant",
        "Jupiter is a massive hydrogen- and helium-dominated planet with rapid rotation, powerful atmospheric circulation and a huge magnetosphere.",
        165, None, "Hydrogen- and helium-dominated atmosphere",
        "Mostly molecular hydrogen and helium with trace compounds",
        "Deep atmosphere, clouds, storms and magnetosphere", 9.925,
        "Extremely strong magnetosphere",
        ("Great Red Spot", "powerful jet streams", "lightning", "aurorae", "intense radiation belts"),
        "Atmospheric turbulence and electromagnetic/environmental sonification",
        "High for physical phenomena; audio mapping is interpretive",
        "Upper-atmosphere representative temperature; deeper temperatures rise dramatically.",
    ),
    "saturn": AstroObject(
        "saturn", "Saturn", "Planet", "Gas giant",
        "Saturn is a hydrogen- and helium-rich gas giant famous for its ring system and complex atmospheric and magnetic environment.",
        134, None, "Hydrogen- and helium-dominated",
        "Mostly hydrogen and helium",
        "Cloud layers, storms, rings and magnetosphere", 10.656,
        "Strong magnetosphere",
        ("hexagonal polar jet", "storms", "aurorae", "ring system"),
        "Atmospheric and magnetospheric interpretive sonification",
        "High for broad physical properties",
        "Representative upper-cloud temperature.",
    ),
    "uranus": AstroObject(
        "uranus", "Uranus", "Planet", "Ice giant",
        "Uranus is an ice giant with a hydrogen, helium and methane-rich upper atmosphere and an unusually tilted rotation axis.",
        59, None, "Hydrogen and helium with methane",
        "Hydrogen, helium and methane-rich upper atmosphere",
        "Cold upper atmosphere and magnetosphere", 17.24,
        "Complex tilted/off-center magnetic field",
        ("extreme axial tilt", "methane absorption", "auroral activity"),
        "Cold-atmosphere and magnetospheric sonification",
        "High for broad physical properties",
        "Representative upper-atmosphere temperature.",
    ),
    "neptune": AstroObject(
        "neptune", "Neptune", "Planet", "Ice giant",
        "Neptune has a cold hydrogen, helium and methane-rich atmosphere and some of the fastest measured planetary winds in the Solar System.",
        59, None, "Hydrogen and helium with methane",
        "Hydrogen, helium and methane-rich upper atmosphere",
        "Cold turbulent atmosphere and magnetosphere", 16.11,
        "Strongly tilted and offset magnetic field",
        ("very high winds", "large storms", "methane clouds", "aurorae"),
        "Atmospheric turbulence and magnetospheric sonification",
        "High for broad physical properties",
        "Representative upper-atmosphere temperature.",
    ),
    "io": AstroObject(
        "io", "Io", "Moon", "Volcanically active Galilean moon",
        "Io is the most volcanically active world known in the Solar System. Its activity is driven largely by tidal heating.",
        110, 1e-9, "Extremely tenuous sulfur-dioxide-dominated atmosphere",
        "Silicate and sulfur-rich surface materials",
        "Volcanic surface, tenuous atmosphere and plasma interaction", 42.46,
        "Embedded in Jupiter's magnetosphere",
        ("active volcanoes", "lava flows", "sulfur dioxide plumes", "intense plasma interaction"),
        "Volcanic-event and plasma-environment sonification",
        "High for volcanic activity; exact acoustic mapping is interpretive",
        "Representative surface temperature; volcanic hot spots are much hotter.",
    ),
    "europa": AstroObject(
        "europa", "Europa", "Moon", "Galilean icy moon",
        "Europa is covered by water ice and is believed to possess a global subsurface ocean beneath its icy shell.",
        102, None, "Extremely tenuous oxygen atmosphere",
        "Water ice over a rocky interior; subsurface ocean inferred",
        "Icy surface, subsurface ocean and magnetospheric interaction", 85.22,
        "Induced magnetic response associated with conductive subsurface material",
        ("ice shell", "subsurface ocean", "surface fractures", "magnetospheric interaction"),
        "Icy/surface and magnetospheric interpretive sonification",
        "High for surface properties; subsurface details remain model-dependent",
        "Representative surface temperature.",
    ),
    "titan": AstroObject(
        "titan", "Titan", "Moon", "Saturnian moon",
        "Titan possesses the densest atmosphere of any Solar System moon and has a nitrogen-rich atmosphere with methane-driven weather processes.",
        94, 1.467, "Dense nitrogen-rich atmosphere",
        "Mostly nitrogen with methane and other hydrocarbons",
        "Dense cold atmosphere, haze, clouds, methane weather and lakes", 382.68,
        "Generally embedded in Saturn's magnetosphere",
        ("methane rain", "methane lakes", "organic haze", "clouds", "seasonal weather"),
        "Dense-atmosphere and methane-weather sonification",
        "High for broad atmospheric properties",
        "Representative surface temperature and pressure.",
    ),
    "enceladus": AstroObject(
        "enceladus", "Enceladus", "Moon", "Icy Saturnian moon",
        "Enceladus is an icy moon with active south-polar jets that eject material from a subsurface water-rich environment.",
        75, None, "Extremely tenuous exosphere",
        "Water ice with water-rich plume material and rocky interior",
        "Icy surface, active plumes and subsurface ocean", 32.88,
        "Interacts strongly with Saturn's magnetosphere",
        ("south-polar plumes", "cryovolcanic activity", "subsurface ocean", "ice fractures"),
        "Plume and magnetospheric interpretive sonification",
        "High for observed plume activity",
        "Representative surface temperature.",
    ),
    "neutron_star": AstroObject(
        "neutron_star", "Neutron Star", "Stellar Remnant", "Compact neutron-degenerate object",
        "A neutron star is an extremely compact stellar remnant with enormous density and potentially very strong magnetic fields.",
        1_000_000, None, "Thin, highly exotic plasma atmosphere where present",
        "Primarily neutron-degenerate matter in the interior",
        "Extreme gravity, dense matter, magnetic and radiation environment", None,
        "Can range from strong to extraordinarily strong",
        ("pulsations", "strong magnetic fields", "relativistic effects", "starquakes", "particle emission"),
        "Data-inspired physical sonification; no ordinary atmospheric sound",
        "High for compact-object physics; mapping is interpretive",
        "Temperature is an illustrative order-of-magnitude value and varies strongly between objects.",
    ),
    "pulsar": AstroObject(
        "pulsar", "Pulsar", "Stellar Remnant", "Rotating neutron star",
        "A pulsar is a rotating neutron star whose radiation beam can periodically sweep across Earth, producing highly regular pulses.",
        None, None, "Not applicable as an Earth-like atmosphere",
        "Neutron-degenerate matter and magnetospheric plasma",
        "Rapid rotation and magnetized plasma environment", None,
        "Extremely strong magnetic field",
        ("precise periodic pulses", "radio emission", "high-energy emission", "magnetospheric plasma"),
        "Direct-style sonification of periodic astrophysical timing concepts",
        "Very high for pulse timing as an observable phenomenon",
        "Individual pulsars have widely varying periods and properties.",
    ),
    "magnetar": AstroObject(
        "magnetar", "Magnetar", "Stellar Remnant", "Highly magnetized neutron star",
        "A magnetar is a neutron star with an extraordinarily strong magnetic field. Some produce powerful bursts of high-energy radiation.",
        None, None, "Not an Earth-like atmosphere",
        "Neutron-degenerate matter with magnetospheric plasma",
        "Extreme magnetic and high-energy radiation environment", None,
        "Exceptionally strong magnetic field",
        ("X-ray bursts", "gamma-ray bursts", "magnetospheric activity", "starquakes"),
        "High-energy event and magnetic-field sonification",
        "High for observed phenomena; audio mapping is interpretive",
        "Magnetic-field strengths vary between magnetars.",
    ),
    "black_hole": AstroObject(
        "black_hole", "Black Hole", "Compact Object", "Black hole",
        "A black hole is a region of spacetime whose event horizon prevents light from escaping outward. Observable environments may include accretion disks and jets.",
        None, None, "No ordinary atmosphere",
        "Spacetime geometry; surrounding matter may form plasma structures",
        "Event horizon, accretion flow, relativistic plasma where present", None,
        "Depends on surrounding plasma and black-hole/accretion system",
        ("accretion", "relativistic jets", "gravitational lensing", "tidal disruption", "gravitational waves from mergers"),
        "Gravitational/accretion/environmental sonification",
        "High for general relativistic phenomena; specific environment depends on object",
        "A black hole itself is not an acoustic environment.",
    ),
}


ENVIRONMENTS = {
    "Atmosphere": (
        "Auditory representation of atmospheric density, turbulence, pressure and related phenomena.",
        "Physics-informed synthesis",
    ),
    "Storm": (
        "Representation of turbulent atmospheric motion and energetic weather phenomena.",
        "Physics-informed interpretive synthesis",
    ),
    "Magnetosphere": (
        "Sonification of magnetic/plasma environmental concepts and available measurements.",
        "Scientific sonification where data exist; otherwise model-based",
    ),
    "Surface": (
        "Environmental representation of surface conditions and measured phenomena.",
        "Physics-informed synthesis",
    ),
    "Volcanic": (
        "Representation of volcanic activity, eruptions and associated physical processes.",
        "Event/environment model",
    ),
    "Extreme Physics": (
        "Auditory representation of extreme-density, magnetic, gravitational or high-energy environments.",
        "Physics-informed conceptual sonification",
    ),
}


if "selected_object" not in st.session_state:
    st.session_state.selected_object = "earth"
if "audio_cache" not in st.session_state:
    st.session_state.audio_cache = {}
if "favorites" not in st.session_state:
    st.session_state.favorites = []
if "gemini_history" not in st.session_state:
    st.session_state.gemini_history = []
if "last_audio_metadata" not in st.session_state:
    st.session_state.last_audio_metadata = None


def safe_float(value, default=0.0):
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def normalize(value, low, high):
    if high == low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def object_to_context(obj: AstroObject) -> str:
    data = asdict(obj)
    return "\n".join(
        f"{key}: {', '.join(value) if isinstance(value, tuple) else value}"
        for key, value in data.items()
    )


def format_temperature(kelvin):
    if kelvin is None:
        return "Not specified"
    celsius = kelvin - 273.15
    return f"{kelvin:,.0f} K ({celsius:,.1f} °C)"


def format_pressure(bar):
    if bar is None:
        return "Not specified"
    if bar == 0:
        return "Effectively negligible"
    return f"{bar:g} bar"


def sine_wave(freq, duration, amplitude=0.2, sr=SAMPLE_RATE):
    n = int(duration * sr)
    t = np.arange(n) / sr
    return amplitude * np.sin(2 * np.pi * freq * t)


def noise(duration, amplitude=0.2, sr=SAMPLE_RATE, seed=0):
    n = int(duration * sr)
    rng = np.random.default_rng(seed)
    return amplitude * rng.normal(0, 1, n)


def lowpass(signal, cutoff_hz, sr=SAMPLE_RATE):
    cutoff_hz = max(1.0, min(cutoff_hz, sr / 2 - 1))
    alpha = (2 * math.pi * cutoff_hz) / (
        2 * math.pi * cutoff_hz + sr
    )
    out = np.empty_like(signal)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        out[i] = out[i - 1] + alpha * (signal[i] - out[i - 1])
    return out


def highpass(signal, cutoff_hz, sr=SAMPLE_RATE):
    return signal - lowpass(signal, cutoff_hz, sr)


def normalize_audio(signal):
    peak = np.max(np.abs(signal))
    if peak <= 1e-9:
        return signal
    return signal / peak * 0.92


def apply_fade(signal, sr=SAMPLE_RATE):
    fade_samples = min(int(0.08 * sr), len(signal) // 2)
    if fade_samples <= 0:
        return signal
    signal[:fade_samples] *= np.linspace(0, 1, fade_samples)
    signal[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return signal


def create_wav(signal, sr=SAMPLE_RATE):
    pcm = (np.clip(signal, -1, 1) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def synthesize_environment(obj, environment_name, duration=AUDIO_DURATION, intensity=0.65):
    sr = SAMPLE_RATE
    n = int(duration * sr)
    t = np.arange(n) / sr

    seed = int(
        hashlib.sha256(
            f"{obj.object_id}:{environment_name}".encode()
        ).hexdigest()[:8],
        16,
    )

    pressure = safe_float(obj.pressure_bar, 0.0)
    temperature = safe_float(obj.temperature_k, 250.0)

    pressure_factor = normalize(math.log10(max(pressure, 1e-6)), -6, 2)
    temp_factor = normalize(math.log10(max(temperature, 1)), 1.5, 4.0)

    category_factor = {
        "Planet": 0.50,
        "Moon": 0.38,
        "Star": 0.72,
        "Stellar Remnant": 0.85,
        "Compact Object": 0.90,
    }.get(obj.category, 0.45)

    signal = noise(
        duration,
        amplitude=0.20 + 0.18 * intensity,
        sr=sr,
        seed=seed,
    )
    signal = lowpass(
        signal,
        350 + 2400 * (0.2 + temp_factor * 0.8),
        sr,
    )

    signal *= 0.20 + 0.22 * intensity

    signal += sine_wave(
        35 + 80 * pressure_factor + 45 * category_factor,
        duration,
        amplitude=0.11 + 0.10 * intensity,
        sr=sr,
    )

    modulation_frequency = {
        "Atmosphere": 0.12,
        "Storm": 0.35,
        "Magnetosphere": 0.70,
        "Surface": 0.08,
        "Volcanic": 0.55,
        "Extreme Physics": 1.10,
    }.get(environment_name, 0.2)

    signal *= (
        0.65
        + 0.35 * np.sin(2 * np.pi * modulation_frequency * t)
    )

    if environment_name in {"Atmosphere", "Storm"}:
        wind_noise = highpass(
            noise(duration, 0.30, sr, seed + 1),
            250,
            sr,
        )
        wind_strength = {
            "jupiter": 0.65,
            "neptune": 0.65,
            "venus": 0.42,
            "mars": 0.24,
            "titan": 0.22,
        }.get(obj.object_id, 0.32)
        signal += wind_noise * wind_strength * intensity

    if environment_name == "Storm":
        storm_rate = {
            "jupiter": 0.28,
            "saturn": 0.20,
            "neptune": 0.34,
            "earth": 0.16,
            "venus": 0.10,
            "mars": 0.08,
        }.get(obj.object_id, 0.10)

        storm_mod = (
            np.sin(2 * np.pi * storm_rate * t)
            + 0.35 * np.sin(2 * np.pi * storm_rate * 2.7 * t)
        )

        storm_layer = lowpass(
            noise(duration, 0.22, sr, seed + 2),
            1600,
            sr,
        )

        signal += (
            storm_layer
            * (0.12 + 0.25 * intensity)
            * (0.5 + 0.5 * storm_mod)
        )

    if environment_name == "Magnetosphere":
        plasma = (
            0.12 * np.sin(2 * np.pi * 220 * t)
            + 0.07 * np.sin(2 * np.pi * 440 * t)
            + 0.04 * np.sin(2 * np.pi * 880 * t)
        )

        plasma_noise = highpass(
            noise(duration, 0.12, sr, seed + 3),
            100,
            sr,
        )

        signal += (plasma + plasma_noise) * intensity

        if obj.object_id in {
            "jupiter", "saturn", "uranus", "neptune",
            "pulsar", "magnetar"
        }:
            pulse_frequency = {
                "jupiter": 0.9,
                "saturn": 0.7,
                "uranus": 0.55,
                "neptune": 0.65,
                "pulsar": 3.2,
                "magnetar": 1.6,
            }.get(obj.object_id, 1.0)

            signal += 0.30 * (
                np.sin(2 * np.pi * pulse_frequency * t) ** 18
            )

    if environment_name == "Surface":
        surface_noise = lowpass(
            noise(duration, 0.12, sr, seed + 4),
            700,
            sr,
        )
        signal += surface_noise * (0.20 + intensity * 0.15)

        if obj.object_id == "mars":
            dust = highpass(
                noise(duration, 0.16, sr, seed + 5),
                700,
                sr,
            )
            signal += dust * 0.18

    if environment_name == "Volcanic":
        eruption_frequency = {
            "io": 0.10,
            "enceladus": 0.16,
        }.get(obj.object_id, 0.08)

        eruption = np.sin(2 * np.pi * eruption_frequency * t)

        volcanic_noise = lowpass(
            noise(duration, 0.28, sr, seed + 6),
            1100,
            sr,
        )

        signal += (
            volcanic_noise
            * (0.25 + 0.25 * intensity)
            * (0.5 + 0.5 * eruption)
        )

        signal += sine_wave(
            48,
            duration,
            amplitude=0.15 * intensity,
            sr=sr,
        )

    if environment_name == "Extreme Physics":
        signal += sine_wave(
            28,
            duration,
            amplitude=0.16,
            sr=sr,
        )

        signal += (
            0.08 * np.sin(2 * np.pi * 9 * t)
            + 0.05 * np.sin(2 * np.pi * 17 * t)
            + 0.03 * np.sin(2 * np.pi * 31 * t)
        ) * intensity

        high_energy = highpass(
            noise(duration, 0.16, sr, seed + 7),
            900,
            sr,
        )
        signal += high_energy * 0.20 * intensity

    signal *= max(0.05, min(1.0, intensity))
    return create_wav(apply_fade(normalize_audio(signal), sr))



def category_objects(category):
    return [
        obj for obj in OBJECTS.values()
        if obj.category == category
    ]


def get_categories():
    return sorted({obj.category for obj in OBJECTS.values()})


def object_names(objects):
    return {obj.name: obj.object_id for obj in objects}


def render_provenance(obj):
    st.markdown(
        f"""
        <div class="provenance">
        <strong>Audio basis:</strong> {obj.audio_basis}<br>
        <strong>Scientific confidence:</strong> {obj.scientific_confidence}<br>
        <strong>Source notes:</strong> {obj.source_notes}<br>
        <strong>Scientific limitation:</strong>
        Generated audio is an auditory representation, not automatically
        a literal recording of sound travelling through space.
        </div>
        """,
        unsafe_allow_html=True,
    )


def object_record_for_export(obj):
    record = asdict(obj)
    record["known_phenomena"] = list(obj.known_phenomena)
    return record


def registry_csv(objects):
    if not objects:
        return b""
    output = io.StringIO()
    fieldnames = list(object_record_for_export(objects[0]).keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for obj in objects:
        row = object_record_for_export(obj)
        row["known_phenomena"] = "; ".join(row["known_phenomena"])
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def compare_prompt(objects, focus):
    records = "\n\n".join(object_to_context(obj) for obj in objects)
    return f"""
Compare these astronomical objects scientifically and neutrally.

RECORDS:
{records}

FOCUS:
{focus}

Rules:
- Do not rank, score, or declare a winner.
- Distinguish documented properties from interpretation.
- Do not invent values.
- Explain which differences matter for auditory representation.
- State important uncertainty and limitations.
"""


def lightweight_gemini_intent(query):
    prompt = f"""
Interpret this astronomy search phrase for a user interface:
{query}

Return a short description of the astronomical concept the user appears
to be seeking. Do not invent an object, observation, measurement, or source.
"""
    return gemini_generate(prompt, GEMINI_LITE, 0.1)


@st.cache_resource(show_spinner=False)
def get_gemini_client():
    if genai is None:
        return None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            return None
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def gemini_generate(prompt, model=GEMINI_FLASH, temperature=0.2):
    client = get_gemini_client()

    if client is None:
        return False, (
            "Gemini is not configured. Check that GEMINI_API_KEY "
            "exists in Streamlit Secrets and that google-genai is installed."
        )

    system_instruction = """
You are the scientific intelligence layer of Sounds of the Universe.

Explain astronomy accurately and conservatively.

Rules:
- Never invent measurements.
- Never claim that a planet, vacuum, or other object has a literal
  audible sound when ordinary sound propagation is not supported.
- Clearly distinguish measured data, scientific sonification,
  physics-based modeling, and interpretation.
- If a value is uncertain or object-dependent, say so.
- Do not turn artistic interpretation into scientific fact.
- Prefer established astronomy over speculation.
- Explanations must be understandable without visual information.
"""

    try:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=1800,
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        text = getattr(response, "text", None)

        if not text:
            return False, "Gemini returned no text."

        return True, text

    except Exception as exc:
        return False, (
            f"Gemini request failed safely: {type(exc).__name__}: {exc}"
        )


def explain_object_with_gemini(obj):
    prompt = f"""
Explain this astronomical object to a visually impaired learner.

SCIENTIFIC RECORD:
{object_to_context(obj)}

Provide:
1. What the object is.
2. Its most important physical conditions.
3. Genuinely observed or strongly established phenomena.
4. What the user is hearing in this application.
5. What the audio does NOT represent.
6. Which parts are measured/sonified versus modeled/interpretive.

Do not invent numerical values absent from the record.
"""
    return gemini_generate(prompt, GEMINI_PRO, 0.15)


def ask_gemini_about_object(obj, question):
    prompt = f"""
The user is exploring:

{object_to_context(obj)}

Question:
{question}

Base the answer primarily on the supplied scientific record.
If it does not contain enough information, explicitly say that
additional data would be required.

Be scientifically conservative.
"""
    return gemini_generate(prompt, GEMINI_FLASH, 0.2)


def search_objects(query):
    query = query.strip().lower()

    if not query:
        return list(OBJECTS.values())

    matches = []

    for obj in OBJECTS.values():
        searchable = " ".join([
            obj.name,
            obj.category,
            obj.subtype,
            obj.description,
            obj.atmosphere,
            obj.composition,
            obj.environment,
            " ".join(obj.known_phenomena),
        ]).lower()

        if query in searchable:
            matches.append(obj)

    return matches


def category_objects(category):
    return [
        obj for obj in OBJECTS.values()
        if obj.category == category
    ]


st.sidebar.title("🌌 Explore")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Explore Objects",
        "Compare Objects",
        "Audio Environments",
        "Scientific Sonifications",
        "Ask the Cosmos",
        "My Favorites",
        "Data & Export",
        "About & Scientific Method",
    ],
)

st.sidebar.divider()

st.sidebar.caption(f"{APP_NAME} · v{APP_VERSION}")

st.sidebar.caption(
    "Audio is an auditory representation, not a claim that every "
    "object literally produces an audible sound."
)


if page == "Home":
    st.title("🌌 Sounds of the Universe")
    st.subheader("Experience the cosmos through scientifically grounded sound.")

    st.write(
        """
        Sounds of the Universe is an accessible astronomy application
        that converts astronomical measurements, physical conditions
        and documented phenomena into auditory experiences.
        """
    )

    st.info(
        "Important: most space is not an ordinary acoustic environment. "
        "Many experiences here are scientific sonifications or "
        "physics-informed models rather than microphone recordings."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Objects in current registry", len(OBJECTS))
    with col2:
        st.metric("Environment types", len(ENVIRONMENTS))
    with col3:
        st.metric("Gemini model roles", 3)

    st.divider()
    st.header("Start exploring")

    cols = st.columns(3)

    with cols[0]:
        if st.button("Explore Earth", use_container_width=True):
            st.session_state.selected_object = "earth"
            st.rerun()

    with cols[1]:
        if st.button("Explore Venus", use_container_width=True):
            st.session_state.selected_object = "venus"
            st.rerun()

    with cols[2]:
        if st.button("Explore Jupiter", use_container_width=True):
            st.session_state.selected_object = "jupiter"
            st.rerun()

    st.divider()
    st.header("How the audio works")

    st.markdown(
        """
        **Measured data** — real observations or measurements where available.

        **Scientific sonification** — scientific data mathematically mapped
        into an audible representation.

        **Physics-based model** — a reproducible audio model controlled by
        physical parameters.

        **Interpretive representation** — an educational auditory
        interpretation where science does not uniquely determine a literal sound.
        """
    )


elif page == "Explore Objects":
    st.title("🔭 Explore astronomical objects")

    query = st.text_input(
        "Search the scientific registry",
        placeholder="Try Venus, magnetar, storm, methane, volcanic...",
    )

    results = search_objects(query)
    st.caption(f"{len(results)} object(s) found in the current registry.")

    if not results:
        st.warning("No matching object is currently in the local registry.")
        if query.strip() and st.button(
            "Interpret search phrase with Gemini",
            key="interpret_search_phrase",
        ):
            with st.spinner("Interpreting the search phrase..."):
                ok, answer = lightweight_gemini_intent(query)
            if ok:
                st.info(answer)
            else:
                st.error(answer)
    else:
        names = {obj.name: obj.object_id for obj in results}

        default_name = OBJECTS.get(
            st.session_state.selected_object,
            results[0],
        ).name

        default_index = (
            list(names.keys()).index(default_name)
            if default_name in names
            else 0
        )

        selected_name = st.selectbox(
            "Choose an object",
            list(names.keys()),
            index=default_index,
        )

        selected_id = names[selected_name]
        st.session_state.selected_object = selected_id
        obj = OBJECTS[selected_id]

        favorite_label = (
            "★ Remove from favorites"
            if obj.object_id in st.session_state.favorites
            else "☆ Add to favorites"
        )

        if st.button(
            favorite_label,
            key=f"favorite_{obj.object_id}",
        ):
            if obj.object_id in st.session_state.favorites:
                st.session_state.favorites.remove(obj.object_id)
            else:
                st.session_state.favorites.append(obj.object_id)
            st.rerun()

        st.divider()
        st.header(obj.name)
        st.write(obj.description)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Category", obj.category)
        with c2:
            st.metric("Subtype", obj.subtype)
        with c3:
            st.metric("Scientific confidence", obj.scientific_confidence)

        st.divider()

        left, right = st.columns(2)

        with left:
            st.subheader("Physical conditions")
            st.write(f"**Temperature:** {format_temperature(obj.temperature_k)}")
            st.write(f"**Pressure:** {format_pressure(obj.pressure_bar)}")
            st.write(f"**Atmosphere:** {obj.atmosphere}")
            st.write(f"**Composition:** {obj.composition}")

            if obj.rotation_period_hours:
                st.write(
                    f"**Rotation period:** "
                    f"{obj.rotation_period_hours:.3f} hours"
                )

        with right:
            st.subheader("Environment")
            st.write(obj.environment)
            st.write(
                f"**Magnetic environment:** {obj.magnetic_environment}"
            )
            st.write("**Known phenomena:**")
            for phenomenon in obj.known_phenomena:
                st.write(f"- {phenomenon}")

        st.divider()
        st.subheader("Scientific provenance")

        st.markdown(
            f"""
            <div class="provenance">
            <strong>Audio basis:</strong><br>
            {obj.audio_basis}
            <br><br>
            <strong>Source notes:</strong><br>
            {obj.source_notes}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.subheader("Scientific explanation")

        if st.button(
            "Explain this object with Gemini",
            type="primary",
        ):
            with st.spinner("Preparing a scientifically conservative explanation..."):
                ok, answer = explain_object_with_gemini(obj)

            if ok:
                st.markdown(answer)
            else:
                st.error(answer)


elif page == "Audio Environments":
    st.title("🎧 Audio environments")

    object_names = {
        obj.name: obj.object_id for obj in OBJECTS.values()
    }

    selected_name = st.selectbox(
        "Astronomical object",
        list(object_names.keys()),
    )

    obj = OBJECTS[object_names[selected_name]]

    environment_name = st.selectbox(
        "Environment",
        list(ENVIRONMENTS.keys()),
    )

    description, method = ENVIRONMENTS[environment_name]

    st.write(description)
    st.caption(f"Method: {method}")

    intensity = st.slider(
        "Auditory intensity",
        min_value=0.1,
        max_value=1.0,
        value=0.65,
        step=0.05,
    )

    duration = st.slider(
        "Duration",
        min_value=5,
        max_value=30,
        value=12,
        step=1,
    )

    st.divider()

    if st.button(
        "Generate environment",
        type="primary",
        use_container_width=True,
    ):
        cache_key = (
            f"{obj.object_id}|{environment_name}|"
            f"{intensity}|{duration}"
        )

        with st.spinner(
            "Synthesizing the scientific auditory representation..."
        ):
            if cache_key not in st.session_state.audio_cache:
                st.session_state.audio_cache[cache_key] = (
                    synthesize_environment(
                        obj,
                        environment_name,
                        float(duration),
                        float(intensity),
                    )
                )

            audio = st.session_state.audio_cache[cache_key]

        st.audio(audio, format="audio/wav")
        st.session_state.last_audio_metadata = {
            "object": obj.name,
            "environment": environment_name,
            "duration_seconds": duration,
            "intensity": intensity,
            "sample_rate_hz": SAMPLE_RATE,
            "audio_basis": obj.audio_basis,
            "scientific_confidence": obj.scientific_confidence,
            "interpretation": (
                "Deterministic educational auditory representation; "
                "not a claim of literal sound propagation through space."
            ),
        }
        st.success("Auditory environment generated.")

        st.markdown(
            f"""
            **What you are hearing**

            This is a deterministic, physics-informed auditory
            representation of **{obj.name} — {environment_name}**.

            It is **not** presented as a literal microphone recording
            of sound traveling through the astronomical environment.

            **Scientific basis:** {obj.audio_basis}

            **Confidence:** {obj.scientific_confidence}
            """
        )


elif page == "Scientific Sonifications":
    st.title("📡 Scientific sonifications")

    st.info(
        "This area is reserved for direct scientific datasets and their "
        "documented sonifications. The current prototype provides "
        "physics-informed models while external-data ingestion is expanded."
    )

    st.header("Provenance categories")

    categories = [
        ("Measured data", "Values or signals directly obtained from scientific instruments."),
        ("Sonification", "Scientific data mathematically mapped into audible parameters."),
        ("Physics-based model", "A reproducible audio representation driven by physical parameters."),
        ("Interpretive", "An educational auditory interpretation where no unique literal sound exists."),
    ]

    for title, description in categories:
        st.markdown(f"**{title}**\n\n{description}")

    st.divider()

    st.header("Why this distinction matters")

    st.write(
        """
        Sound waves require a material medium. Empty interplanetary and
        interstellar space is not an ordinary acoustic environment.

        Astronomical instruments can nevertheless detect electromagnetic
        waves, plasma waves, vibrations, timing signals and other physical
        measurements. Those measurements can be transformed into audible
        frequencies.

        This application therefore avoids presenting every generated sound
        as though a microphone literally recorded a planet.
        """
    )


elif page == "Ask the Cosmos":
    st.title("🧠 Ask the Cosmos")

    object_names = {
        obj.name: obj.object_id for obj in OBJECTS.values()
    }

    selected_name = st.selectbox(
        "Context object",
        list(object_names.keys()),
    )

    obj = OBJECTS[object_names[selected_name]]

    st.write(f"Current context: **{obj.name}**")

    question = st.text_area(
        "Ask an astronomy question",
        placeholder="Why does this environment sound different from Earth?",
        height=120,
    )

    if st.button("Ask Gemini", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Consulting the scientific reasoning layer..."):
                ok, answer = ask_gemini_about_object(
                    obj,
                    question.strip(),
                )

            if ok:
                st.markdown(answer)
                st.session_state.gemini_history.append(
                    {
                        "type": "question",
                        "object": obj.name,
                        "question": question.strip(),
                        "answer": answer,
                    }
                )
            else:
                st.error(answer)

    if st.session_state.gemini_history:
        with st.expander("Current-session Gemini history"):
            for item in reversed(st.session_state.gemini_history[-10:]):
                st.markdown(
                    f"**{item.get('type', 'interaction')} — "
                    f"{item.get('object', '')}**"
                )
                if item.get("question"):
                    st.write(f"Question: {item['question']}")
                st.write(item.get("answer", ""))

    st.divider()
    st.subheader("Gemini model roles")

    st.write(
        f"""
        **Gemini 3.1 Pro** — deeper scientific explanations and complex interpretation.

        **Gemini 3.6 Flash** — interactive astronomy questions and exploration.

        **Gemini 3.5 Flash-Lite** — reserved for lightweight/high-volume
        operations as the catalogue expands.
        """
    )

    st.caption(
        "The scientific registry remains the source of application-specific "
        "facts. Gemini is not treated as the application's measurement database."
    )



elif page == "Compare Objects":
    st.title("⚖️ Compare astronomical objects")
    st.write(
        "Compare documented properties without turning the comparison "
        "into a ranking or score."
    )

    names = list(object_names(OBJECTS.values()).keys())
    defaults = ["Earth", "Venus"] if "Earth" in names and "Venus" in names else names[:2]

    selected_names = st.multiselect(
        "Select objects",
        names,
        default=defaults,
        max_selections=4,
    )

    if len(selected_names) < 2:
        st.info("Select at least two objects.")
    else:
        selected_objects = [
            OBJECTS[object_names(OBJECTS.values())[name]]
            for name in selected_names
        ]

        rows = []
        for obj in selected_objects:
            rows.append(
                {
                    "Object": obj.name,
                    "Category": obj.category,
                    "Subtype": obj.subtype,
                    "Temperature": format_temperature(obj.temperature_k),
                    "Pressure": format_pressure(obj.pressure_bar),
                    "Atmosphere": obj.atmosphere,
                    "Rotation": (
                        f"{obj.rotation_period_hours:.3f} h"
                        if obj.rotation_period_hours is not None
                        else "Not specified"
                    ),
                    "Magnetic environment": obj.magnetic_environment,
                    "Audio basis": obj.audio_basis,
                }
            )

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )

        focus = st.text_input(
            "Comparison focus",
            value="environmental and atmospheric differences",
        )

        if st.button(
            "Explain comparison with Gemini",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Preparing a neutral scientific comparison..."):
                ok, answer = gemini_generate(
                    compare_prompt(selected_objects, focus),
                    GEMINI_PRO,
                    0.15,
                )
            if ok:
                st.markdown(answer)
            else:
                st.error(answer)


elif page == "My Favorites":
    st.title("★ My Favorites")

    if not st.session_state.favorites:
        st.info(
            "No favorites yet. Add an object from Explore Objects."
        )
    else:
        for object_id in list(st.session_state.favorites):
            if object_id not in OBJECTS:
                continue

            obj = OBJECTS[object_id]

            with st.expander(
                f"{obj.name} · {obj.category} · {obj.subtype}"
            ):
                st.write(obj.description)
                st.write(
                    f"Temperature: {format_temperature(obj.temperature_k)}"
                )
                st.write(
                    f"Pressure: {format_pressure(obj.pressure_bar)}"
                )
                render_provenance(obj)

                if st.button(
                    f"Remove {obj.name}",
                    key=f"remove_favorite_{obj.object_id}",
                ):
                    st.session_state.favorites.remove(obj.object_id)
                    st.rerun()


elif page == "Data & Export":
    st.title("💾 Data & Export")

    st.info(
        "These exports contain the application's current local registry. "
        "They are not a claim of a complete astronomical catalogue."
    )

    selected_category = st.selectbox(
        "Category",
        ["All"] + get_categories(),
    )

    export_objects = (
        list(OBJECTS.values())
        if selected_category == "All"
        else category_objects(selected_category)
    )

    st.metric("Objects", len(export_objects))

    rows = []
    for obj in export_objects:
        rows.append(
            {
                "Object": obj.name,
                "Category": obj.category,
                "Subtype": obj.subtype,
                "Temperature": format_temperature(obj.temperature_k),
                "Pressure": format_pressure(obj.pressure_bar),
                "Provenance": obj.audio_basis,
                "Confidence": obj.scientific_confidence,
            }
        )

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
    )

    json_bytes = json.dumps(
        [object_record_for_export(obj) for obj in export_objects],
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")

    c1, c2 = st.columns(2)

    with c1:
        st.download_button(
            "Download JSON",
            data=json_bytes,
            file_name="sounds_of_the_universe_registry.json",
            mime="application/json",
            use_container_width=True,
        )

    with c2:
        st.download_button(
            "Download CSV",
            data=registry_csv(export_objects),
            file_name="sounds_of_the_universe_registry.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()
    st.subheader("Last generated audio")

    if st.session_state.last_audio_metadata:
        st.json(st.session_state.last_audio_metadata)
    else:
        st.caption("No audio generated in this session yet.")


elif page == "About & Scientific Method":
    st.title("About Sounds of the Universe")

    st.subheader("Purpose")

    st.write(
        """
        Sounds of the Universe is designed to make astronomical exploration
        more accessible through auditory information.

        The project is especially intended to support people who are blind
        or visually impaired, while remaining scientifically transparent
        about what each sound represents.
        """
    )

    st.divider()
    st.subheader("Scientific method")

    st.write(
        """
        The application follows a simple principle:

        Scientific observation → physical interpretation → reproducible auditory mapping.
        """
    )

    st.subheader("What the application does not claim")

    st.write(
        """
        It does not claim that every planet, star or black hole has a
        naturally audible sound that could be heard by a human listener
        in space.

        It also does not allow an AI-generated artistic sound to be
        presented as an observational measurement.
        """
    )

    st.divider()
    st.subheader("Current registry")

    for category in sorted(
        set(obj.category for obj in OBJECTS.values())
    ):
        objects = category_objects(category)

        with st.expander(f"{category} · {len(objects)} object(s)"):
            for obj in objects:
                st.write(f"**{obj.name}** — {obj.subtype}")

    st.divider()
    st.subheader("Future expansion")

    st.write(
        """
        The registry is designed to expand into much larger astronomical
        catalogues, including Solar System catalogues, exoplanets, stars,
        galaxies, spacecraft measurements, plasma-wave data, solar
        observations, pulsar timing, gravitational-wave data, planetary
        atmospheres and magnetospheric measurements.
        """
    )

    st.caption(
        "The current application is an early scientific prototype, "
        "not a complete catalogue of the observable universe."
    )


st.divider()

st.caption(
    "Sounds of the Universe — scientifically grounded auditory astronomy."
)
