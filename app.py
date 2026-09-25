import io
import math
import wave
import hashlib
import csv
import json
import html
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


APP_NAV_LABEL = "Explore"

st.set_page_config(
    page_title=APP_NAME,
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Swiss-modern / NASA-Science-inspired visual system:
# strong typographic hierarchy, generous grid, black/white surfaces,
# one restrained blue accent, thin rules, numbered sections, and no decorative imagery.
st.markdown(
    """
    <style>
    :root {
        --su-ink: #111216;
        --su-ink-2: #1a1b20;
        --su-paper: #f6f6f3;
        --su-white: #ffffff;
        --su-muted: #777b84;
        --su-rule: #d9d9d5;
        --su-blue: #1479ff;
        --su-blue-dark: #075fcf;
        --su-red: #d71920;
        --su-max: 1440px;
    }

    html, body, [class*="css"] {
        font-family: Inter, Helvetica, Arial, sans-serif;
    }

    .stApp {
        background: var(--su-paper);
        color: var(--su-ink);
    }

    [data-testid="stHeader"] {
        background: rgba(246,246,243,0.92);
        border-bottom: 1px solid var(--su-rule);
    }

    [data-testid="stSidebar"] {
        background: #121317;
        border-right: 1px solid #2a2b31;
    }

    [data-testid="stSidebar"] * {
        color: #f4f4f1;
    }

    [data-testid="stSidebar"] .stRadio label {
        padding: 0.25rem 0;
    }

    .block-container {
        max-width: var(--su-max);
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    /* Make Streamlit's default widget chrome fit the system. */
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 0 !important;
        min-height: 44px;
        border: 1px solid #15161a;
        background: #15161a;
        color: #fff;
        font-weight: 650;
        letter-spacing: 0.01em;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: var(--su-blue);
        background: var(--su-blue);
        color: #fff;
    }

    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {
        border-radius: 0 !important;
        border-color: #bdbfbd !important;
    }

    .stSlider [data-baseweb="slider"] {
        accent-color: var(--su-blue);
    }

    .su-shell {
        width: 100%;
    }

    .su-kicker {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin: 0 0 0.7rem 0;
        color: #676b73;
        font-size: 0.70rem;
        line-height: 1;
        letter-spacing: 0.24em;
        text-transform: uppercase;
        font-weight: 800;
    }

    .su-kicker::before {
        content: "";
        width: 24px;
        height: 2px;
        background: var(--su-blue);
        display: inline-block;
    }

    .su-hero {
        position: relative;
        overflow: hidden;
        min-height: 430px;
        padding: 4.2rem 4.5rem 3.3rem 4.5rem;
        background: #050506;
        color: #fff;
        border-bottom: 8px solid var(--su-blue);
        margin-bottom: 1.5rem;
    }

    .su-hero::before,
    .su-hero::after {
        content: "";
        position: absolute;
        border: 1px solid rgba(255,255,255,0.13);
        border-radius: 50%;
        pointer-events: none;
    }

    .su-hero::before {
        width: 460px;
        height: 460px;
        right: -130px;
        top: -140px;
    }

    .su-hero::after {
        width: 270px;
        height: 270px;
        right: 35px;
        top: -45px;
        border-color: rgba(20,121,255,0.34);
    }

    .su-hero-grid {
        position: relative;
        z-index: 2;
        display: grid;
        grid-template-columns: minmax(0, 1.8fr) minmax(240px, 0.8fr);
        gap: 3rem;
        align-items: end;
    }

    .su-hero-title {
        margin: 0;
        font-size: clamp(3rem, 7vw, 7rem);
        line-height: 0.88;
        letter-spacing: -0.055em;
        font-weight: 800;
        max-width: 900px;
    }

    .su-hero-subtitle {
        margin: 1.4rem 0 0;
        max-width: 700px;
        font-size: clamp(1rem, 1.6vw, 1.35rem);
        line-height: 1.45;
        color: #d7d7d2;
    }

    .su-hero-index {
        justify-self: end;
        min-width: 230px;
        border-top: 1px solid rgba(255,255,255,0.35);
        padding-top: 1rem;
        text-align: right;
    }

    .su-hero-index-number {
        display: block;
        color: var(--su-blue);
        font-size: 4.5rem;
        line-height: 0.9;
        font-weight: 300;
        letter-spacing: -0.05em;
    }

    .su-hero-index-label {
        display: block;
        margin-top: 0.7rem;
        font-size: 0.72rem;
        letter-spacing: 0.20em;
        text-transform: uppercase;
        color: #bdbdb8;
    }

    .su-rule {
        height: 1px;
        background: var(--su-rule);
        margin: 2.2rem 0;
    }

    .su-section-head {
        display: flex;
        justify-content: space-between;
        align-items: end;
        gap: 1.5rem;
        border-bottom: 1px solid #bfc0bd;
        padding-bottom: 0.85rem;
        margin: 2.1rem 0 1.25rem;
    }

    .su-section-title {
        margin: 0;
        font-size: clamp(1.6rem, 3vw, 2.6rem);
        line-height: 1;
        letter-spacing: -0.035em;
        font-weight: 780;
    }

    .su-section-note {
        color: #6d7078;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        text-align: right;
    }

    .su-number-card {
        background: #111216;
        color: #fff;
        min-height: 180px;
        padding: 1.2rem 1.3rem 1.4rem;
        border-top: 3px solid var(--su-blue);
    }

    .su-number {
        font-size: 3.8rem;
        line-height: 0.95;
        font-weight: 250;
        color: var(--su-blue);
        letter-spacing: -0.05em;
    }

    .su-number-label {
        margin-top: 0.75rem;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #d7d7d2;
    }

    .su-card {
        background: #fff;
        border: 1px solid #d5d5d1;
        padding: 1.35rem;
        min-height: 100%;
    }

    .su-card-dark {
        background: #15161a;
        color: #fff;
        border-color: #15161a;
    }

    .su-card-index {
        color: var(--su-blue);
        font-size: 0.72rem;
        letter-spacing: 0.16em;
        font-weight: 800;
        text-transform: uppercase;
    }

    .su-card h3 {
        margin: 0.55rem 0 0.55rem;
        font-size: 1.35rem;
        line-height: 1.05;
        letter-spacing: -0.025em;
    }

    .su-card p {
        color: #60636b;
        line-height: 1.55;
    }

    .su-card-dark p {
        color: #c7c7c2;
    }

    .su-object-glyph {
        width: 76px;
        height: 76px;
        border-radius: 50%;
        border: 1px solid #c8c9c5;
        background:
            radial-gradient(circle at 35% 30%, #ffffff 0 5%, transparent 6%),
            radial-gradient(circle at 60% 65%, #c6c8cc 0 12%, transparent 13%),
            #25262b;
        margin-bottom: 1rem;
    }

    .su-object-glyph.star {
        background:
            radial-gradient(circle, #fff 0 5%, #1479ff 6% 8%, transparent 9%),
            #101115;
    }

    .su-object-glyph.moon {
        background:
            radial-gradient(circle at 38% 32%, #ececea 0 6%, transparent 7%),
            radial-gradient(circle at 65% 58%, #aaa 0 10%, transparent 11%),
            #60636a;
    }

    .su-object-glyph.compact {
        background:
            radial-gradient(circle, #fff 0 3%, #1479ff 4% 9%, #0b0c0e 10% 100%);
    }

    .su-meta {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0;
        border-top: 1px solid #cfd0cc;
        border-left: 1px solid #cfd0cc;
        margin-top: 1rem;
    }

    .su-meta-cell {
        padding: 0.75rem;
        border-right: 1px solid #cfd0cc;
        border-bottom: 1px solid #cfd0cc;
    }

    .su-meta-label {
        display: block;
        color: #74777e;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.25rem;
    }

    .su-meta-value {
        display: block;
        font-size: 0.86rem;
        line-height: 1.35;
        font-weight: 650;
    }

    .su-provenance {
        border-left: 3px solid var(--su-blue);
        background: #fff;
        border-top: 1px solid #ddd;
        border-right: 1px solid #ddd;
        border-bottom: 1px solid #ddd;
        padding: 1rem 1.1rem;
        margin-top: 0.8rem;
    }

    .su-provenance strong {
        color: #111216;
    }

    .su-limit {
        color: #62656c;
        font-size: 0.86rem;
        line-height: 1.5;
    }

    .su-toc {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        border-top: 1px solid #c7c8c4;
        border-left: 1px solid #c7c8c4;
        margin: 1.5rem 0 2.5rem;
    }

    .su-toc-item {
        padding: 1rem;
        min-height: 94px;
        border-right: 1px solid #c7c8c4;
        border-bottom: 1px solid #c7c8c4;
        background: #fff;
    }

    .su-toc-item .n {
        display: block;
        color: var(--su-blue);
        font-size: 0.7rem;
        letter-spacing: 0.14em;
        font-weight: 800;
        margin-bottom: 0.65rem;
    }

    .su-toc-item .t {
        font-weight: 720;
        font-size: 0.9rem;
    }

    .su-footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #c7c8c4;
        color: #74777e;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
    }

    @media (max-width: 900px) {
        .su-hero {
            min-height: 390px;
            padding: 2.5rem 1.4rem 2.2rem;
        }
        .su-hero-grid {
            grid-template-columns: 1fr;
            gap: 2rem;
        }
        .su-hero-index {
            justify-self: start;
            text-align: left;
        }
        .su-toc {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 600px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .su-hero-title {
            font-size: 3.25rem;
        }
        .su-toc {
            grid-template-columns: 1fr;
        }
        .su-meta {
            grid-template-columns: 1fr;
        }
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


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def render_kicker(text: str):
    st.markdown(f'<div class="su-kicker">{esc(text)}</div>', unsafe_allow_html=True)


def render_section(title: str, note: str = ""):
    note_html = f'<div class="su-section-note">{esc(note)}</div>' if note else ""
    st.markdown(
        f"""
        <div class="su-section-head">
            <h2 class="su-section-title">{esc(title)}</h2>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, index_number: str, index_label: str):
    st.markdown(
        f"""
        <section class="su-hero">
            <div class="su-hero-grid">
                <div>
                    <div class="su-kicker" style="color:#bcbdb9;">{esc(APP_NAME)}</div>
                    <h1 class="su-hero-title">{esc(title)}</h1>
                    <p class="su-hero-subtitle">{esc(subtitle)}</p>
                </div>
                <div class="su-hero-index">
                    <span class="su-hero-index-number">{esc(index_number)}</span>
                    <span class="su-hero-index-label">{esc(index_label)}</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_number_card(number: str, label: str):
    st.markdown(
        f"""
        <div class="su-number-card">
            <div class="su-number">{esc(number)}</div>
            <div class="su-number-label">{esc(label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_object_glyph(obj: AstroObject):
    cls = {
        "Star": "star",
        "Moon": "moon",
        "Stellar Remnant": "compact",
        "Compact Object": "compact",
    }.get(obj.category, "")
    st.markdown(f'<div class="su-object-glyph {cls}"></div>', unsafe_allow_html=True)


def render_provenance(obj: AstroObject):
    st.markdown(
        f"""
        <div class="su-provenance">
            <strong>Audio basis</strong><br>{esc(obj.audio_basis)}
            <br><br>
            <strong>Scientific confidence</strong><br>{esc(obj.scientific_confidence)}
            <br><br>
            <strong>Source notes</strong><br>{esc(obj.source_notes)}
            <br><br>
            <span class="su-limit">
                <strong>Scientific limitation</strong><br>
                Generated audio is an auditory representation. It is not
                automatically a literal recording of sound travelling through space.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def object_card(obj: AstroObject, index: int):
    phenomena = ", ".join(obj.known_phenomena[:3])
    st.markdown(
        f"""
        <div class="su-card">
            <div class="su-card-index">{index:02d} / {esc(obj.category)}</div>
            <h3>{esc(obj.name)}</h3>
            <p>{esc(obj.description)}</p>
            <div class="su-meta">
                <div class="su-meta-cell">
                    <span class="su-meta-label">Subtype</span>
                    <span class="su-meta-value">{esc(obj.subtype)}</span>
                </div>
                <div class="su-meta-cell">
                    <span class="su-meta-label">Temperature</span>
                    <span class="su-meta-value">{esc(format_temperature(obj.temperature_k))}</span>
                </div>
                <div class="su-meta-cell">
                    <span class="su-meta-label">Pressure</span>
                    <span class="su-meta-value">{esc(format_pressure(obj.pressure_bar))}</span>
                </div>
                <div class="su-meta-cell">
                    <span class="su-meta-label">Phenomena</span>
                    <span class="su-meta-value">{esc(phenomena)}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def set_selected(object_id: str):
    st.session_state.selected_object = object_id


# ---------- Global navigation ----------
if "main_navigation" not in st.session_state:
    st.session_state.main_navigation = "Home"
if st.session_state.pop("pending_navigation", None):
    st.session_state.main_navigation = "Explore Objects"

st.sidebar.markdown(
    """
    <div style="padding:0.4rem 0 1.1rem;">
        <div style="font-size:0.68rem;letter-spacing:0.22em;text-transform:uppercase;color:#8d9098;">
            Scientific audio atlas
        </div>
        <div style="font-size:1.45rem;font-weight:800;letter-spacing:-0.04em;margin-top:0.35rem;">
            Sounds of the Universe
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigate",
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
    key="main_navigation",
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption(f"{APP_NAME}  /  {APP_VERSION}")
st.sidebar.caption(
    "Auditory representations are not automatically literal recordings of sound in space."
)


# ---------- Home ----------
if page == "Home":
    render_hero(
        "Sounds of the Universe",
        "Explore astronomical environments through reproducible auditory representations, documented phenomena, and a transparent scientific method.",
        str(len(OBJECTS)).zfill(2),
        "objects in the current registry",
    )

    st.markdown(
        """
        <div class="su-toc">
            <div class="su-toc-item"><span class="n">01</span><span class="t">Object registry</span></div>
            <div class="su-toc-item"><span class="n">02</span><span class="t">Audio environments</span></div>
            <div class="su-toc-item"><span class="n">03</span><span class="t">Scientific sonification</span></div>
            <div class="su-toc-item"><span class="n">04</span><span class="t">Accessible astronomy</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_kicker("The premise")
    st.markdown(
        """
        <div style="max-width:900px;font-size:1.22rem;line-height:1.55;">
        Sounds of the Universe turns selected astronomical properties,
        physical conditions, and documented phenomena into audible representations.
        The application separates measured information from modeling and interpretation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='su-rule'></div>", unsafe_allow_html=True)

    render_section("The registry", f"{len(OBJECTS)} objects")
    cols = st.columns(4)
    counts = {}
    for obj in OBJECTS.values():
        counts[obj.category] = counts.get(obj.category, 0) + 1
    for col, (category, count) in zip(cols, sorted(counts.items())):
        with col:
            render_number_card(str(count).zfill(2), category)

    render_section("Begin an exploration", "Direct entry points")
    quick_ids = ["earth", "venus", "jupiter", "mars"]
    cols = st.columns(4)
    for col, object_id in zip(cols, quick_ids):
        obj = OBJECTS[object_id]
        with col:
            render_object_glyph(obj)
            st.markdown(f"### {esc(obj.name)}")
            st.caption(obj.subtype)
            if st.button("Open object", key=f"home_open_{object_id}", use_container_width=True):
                set_selected(object_id)
                st.session_state.pending_navigation = True
                st.rerun()

    render_section("How the audio is constructed", "Four provenance levels")
    method_cols = st.columns(4)
    methods = [
        ("01", "Measured data", "Observed or instrument-derived values where available."),
        ("02", "Scientific sonification", "A documented mathematical mapping from data to audible parameters."),
        ("03", "Physics-based model", "A reproducible synthesis driven by physical parameters."),
        ("04", "Interpretive", "An educational representation where science does not uniquely specify a literal sound."),
    ]
    for col, (num, title, desc) in zip(method_cols, methods):
        with col:
            st.markdown(
                f"""
                <div class="su-card">
                    <div class="su-card-index">{num}</div>
                    <h3>{esc(title)}</h3>
                    <p>{esc(desc)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="su-card su-card-dark" style="margin-top:1.5rem;">
            <div class="su-card-index">Important distinction</div>
            <h3>Space is not automatically an acoustic environment.</h3>
            <p>
            Sound requires a material medium. Many astronomical measurements are
            electromagnetic, plasma, timing, or other physical signals. This application
            can transform such information into sound without claiming that a microphone
            literally recorded a planet in empty space.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- Explore Objects ----------
elif page == "Explore Objects":
    render_kicker("01 / Object registry")
    render_section("Explore astronomical objects", f"{len(OBJECTS)} records")

    q_col, cat_col = st.columns([2.3, 1])
    with q_col:
        query = st.text_input(
            "Search",
            placeholder="Search by object, category, atmosphere, phenomenon, or environment",
            label_visibility="collapsed",
        )
    with cat_col:
        category_filter = st.selectbox(
            "Category",
            ["All"] + get_categories(),
            label_visibility="collapsed",
        )

    results = search_objects(query)
    if category_filter != "All":
        results = [obj for obj in results if obj.category == category_filter]

    st.caption(f"{len(results)} matching object(s) in the local registry.")

    if not results:
        st.warning("No matching object is currently in the local registry.")
        if query.strip() and st.button("Interpret search phrase with Gemini"):
            with st.spinner("Interpreting the search phrase..."):
                ok, answer = lightweight_gemini_intent(query)
            if ok:
                st.info(answer)
            else:
                st.error(answer)
    else:
        names = {obj.name: obj.object_id for obj in results}
        current = OBJECTS.get(st.session_state.selected_object)
        default_name = current.name if current and current.name in names else results[0].name
        selected_name = st.selectbox(
            "Selected object",
            list(names.keys()),
            index=list(names.keys()).index(default_name),
        )
        selected_id = names[selected_name]
        st.session_state.selected_object = selected_id
        obj = OBJECTS[selected_id]

        render_section(obj.name, obj.category)
        left, right = st.columns([1.25, 2.4])

        with left:
            render_object_glyph(obj)
            st.markdown(f"### {esc(obj.subtype)}")
            st.write(obj.description)
            favorite_label = (
                "Remove from favorites"
                if obj.object_id in st.session_state.favorites
                else "Add to favorites"
            )
            if st.button(favorite_label, key=f"favorite_{obj.object_id}", use_container_width=True):
                if obj.object_id in st.session_state.favorites:
                    st.session_state.favorites.remove(obj.object_id)
                else:
                    st.session_state.favorites.append(obj.object_id)
                st.rerun()

        with right:
            st.markdown(
                f"""
                <div class="su-meta">
                    <div class="su-meta-cell"><span class="su-meta-label">Category</span><span class="su-meta-value">{esc(obj.category)}</span></div>
                    <div class="su-meta-cell"><span class="su-meta-label">Subtype</span><span class="su-meta-value">{esc(obj.subtype)}</span></div>
                    <div class="su-meta-cell"><span class="su-meta-label">Temperature</span><span class="su-meta-value">{esc(format_temperature(obj.temperature_k))}</span></div>
                    <div class="su-meta-cell"><span class="su-meta-label">Pressure</span><span class="su-meta-value">{esc(format_pressure(obj.pressure_bar))}</span></div>
                    <div class="su-meta-cell"><span class="su-meta-label">Rotation</span><span class="su-meta-value">{esc(f"{obj.rotation_period_hours:.3f} hours" if obj.rotation_period_hours is not None else "Not specified")}</span></div>
                    <div class="su-meta-cell"><span class="su-meta-label">Magnetic environment</span><span class="su-meta-value">{esc(obj.magnetic_environment)}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        render_section("Physical environment", "Documented properties")
        p1, p2 = st.columns(2)
        with p1:
            st.markdown("**Atmosphere**")
            st.write(obj.atmosphere)
            st.markdown("**Composition**")
            st.write(obj.composition)
        with p2:
            st.markdown("**Environment**")
            st.write(obj.environment)
            st.markdown("**Known phenomena**")
            for phenomenon in obj.known_phenomena:
                st.write(f"- {phenomenon}")

        render_section("Scientific provenance", "Do not confuse representation with observation")
        render_provenance(obj)

        render_section("Scientific explanation", "Gemini is an explanation layer, not the registry")
        if st.button("Explain this object with Gemini", type="primary"):
            with st.spinner("Preparing a scientifically conservative explanation..."):
                ok, answer = explain_object_with_gemini(obj)
            if ok:
                st.markdown(answer)
            else:
                st.error(answer)

        render_section("Related registry entries", "Context")
        related = [
            candidate for candidate in OBJECTS.values()
            if candidate.object_id != obj.object_id and (
                candidate.category == obj.category
                or candidate.environment == obj.environment
            )
        ][:4]
        cols = st.columns(max(1, min(4, len(related))))
        for col, related_obj in zip(cols, related):
            with col:
                object_card(related_obj, list(OBJECTS).index(related_obj.object_id) + 1)


# ---------- Compare ----------
elif page == "Compare Objects":
    render_kicker("02 / Comparative view")
    render_section("Compare astronomical objects", "Documented differences, no ranking")

    names = list(object_names(OBJECTS.values()).keys())
    defaults = ["Earth", "Venus"] if "Earth" in names and "Venus" in names else names[:2]
    selected_names = st.multiselect(
        "Objects",
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
        st.dataframe(rows, use_container_width=True, hide_index=True)

        focus = st.text_input(
            "Comparison focus",
            value="environmental and atmospheric differences",
        )
        if st.button("Explain comparison with Gemini", type="primary", use_container_width=True):
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


# ---------- Audio environments ----------
elif page == "Audio Environments":
    render_kicker("03 / Auditory laboratory")
    render_section("Audio environments", "Deterministic synthesis")

    left, right = st.columns([1, 1.4])
    object_names_map = {obj.name: obj.object_id for obj in OBJECTS.values()}

    with left:
        selected_name = st.selectbox("Astronomical object", list(object_names_map.keys()))
        obj = OBJECTS[object_names_map[selected_name]]
        environment_name = st.selectbox("Environment", list(ENVIRONMENTS.keys()))
        description, method = ENVIRONMENTS[environment_name]
        st.markdown(f"**{esc(environment_name)}**")
        st.write(description)
        st.caption(f"Method: {method}")

    with right:
        intensity = st.slider("Auditory intensity", 0.1, 1.0, 0.65, 0.05)
        duration = st.slider("Duration (seconds)", 5, 30, 12, 1)

        st.markdown(
            f"""
            <div class="su-card">
                <div class="su-card-index">Output specification</div>
                <h3>{esc(obj.name)} / {esc(environment_name)}</h3>
                <p>
                    {duration} seconds at {intensity:.2f} intensity, generated at
                    {SAMPLE_RATE:,} Hz. The signal is deterministic for the same
                    object, environment, duration, and intensity.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("Generate environment", type="primary", use_container_width=True):
        cache_key = f"{obj.object_id}|{environment_name}|{intensity}|{duration}"
        with st.spinner("Synthesizing the auditory representation..."):
            if cache_key not in st.session_state.audio_cache:
                st.session_state.audio_cache[cache_key] = synthesize_environment(
                    obj, environment_name, float(duration), float(intensity)
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
        render_provenance(obj)


# ---------- Scientific sonifications ----------
elif page == "Scientific Sonifications":
    render_kicker("04 / Provenance")
    render_section("Scientific sonifications", "Measurement versus representation")

    st.markdown(
        """
        <div class="su-card su-card-dark">
            <div class="su-card-index">Current status</div>
            <h3>The application currently provides physics-informed models.</h3>
            <p>
            Direct ingestion of external scientific datasets and documented
            instrument sonifications is a separate expansion path. Until such
            datasets are connected, generated audio is not presented as an
            observational recording.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section("Provenance categories", "Four distinct meanings")
    categories = [
        ("01", "Measured data", "Values or signals directly obtained from scientific instruments."),
        ("02", "Sonification", "Scientific data mathematically mapped into audible parameters."),
        ("03", "Physics-based model", "A reproducible audio representation driven by physical parameters."),
        ("04", "Interpretive", "An educational auditory interpretation where no unique literal sound exists."),
    ]
    cols = st.columns(4)
    for col, (num, title, description) in zip(cols, categories):
        with col:
            st.markdown(
                f"""
                <div class="su-card">
                    <div class="su-card-index">{num}</div>
                    <h3>{esc(title)}</h3>
                    <p>{esc(description)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_section("Why the distinction matters", "Scientific communication")
    st.write(
        """
        Sound waves require a material medium. Empty interplanetary and
        interstellar space is not an ordinary acoustic environment.

        Astronomical instruments can nevertheless detect electromagnetic waves,
        plasma waves, vibrations, timing signals, and other physical measurements.
        Those measurements can be transformed into audible frequencies.

        This application therefore avoids presenting every generated sound as
        though a microphone literally recorded a planet.
        """
    )


# ---------- Ask the Cosmos ----------
elif page == "Ask the Cosmos":
    render_kicker("05 / Scientific dialogue")
    render_section("Ask the Cosmos", "Gemini-assisted explanation")

    object_names_map = {obj.name: obj.object_id for obj in OBJECTS.values()}
    selected_name = st.selectbox("Context object", list(object_names_map.keys()))
    obj = OBJECTS[object_names_map[selected_name]]

    st.markdown(
        f"""
        <div class="su-card">
            <div class="su-card-index">Current context</div>
            <h3>{esc(obj.name)}</h3>
            <p>{esc(obj.description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    question = st.text_area(
        "Ask an astronomy question",
        placeholder="Why does this environment sound different from Earth?",
        height=130,
    )

    if st.button("Ask Gemini", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Consulting the scientific reasoning layer..."):
                ok, answer = ask_gemini_about_object(obj, question.strip())
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
        render_section("Current-session history", "Latest 10 interactions")
        for item in reversed(st.session_state.gemini_history[-10:]):
            with st.expander(
                f"{item.get('object', 'Interaction')} / {item.get('question', '')}"
            ):
                st.write(item.get("answer", ""))

    render_section("Model roles", "Application architecture")
    model_cols = st.columns(3)
    model_data = [
        ("Pro", GEMINI_PRO, "Deeper scientific explanations and complex interpretation."),
        ("Flash", GEMINI_FLASH, "Interactive astronomy questions and exploration."),
        ("Lite", GEMINI_LITE, "Lightweight operations as the catalogue expands."),
    ]
    for col, (label, model, role) in zip(model_cols, model_data):
        with col:
            st.markdown(
                f"""
                <div class="su-card">
                    <div class="su-card-index">{esc(label)}</div>
                    <h3>{esc(model)}</h3>
                    <p>{esc(role)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        "The local scientific registry remains the application's source of application-specific facts. "
        "Gemini is not treated as the measurement database."
    )


# ---------- Favorites ----------
elif page == "My Favorites":
    render_kicker("06 / Saved objects")
    render_section("My favorites", f"{len(st.session_state.favorites)} saved")

    if not st.session_state.favorites:
        st.info("No favorites yet. Add an object from Explore Objects.")
    else:
        favorite_objects = [
            OBJECTS[object_id]
            for object_id in st.session_state.favorites
            if object_id in OBJECTS
        ]
        cols = st.columns(min(3, len(favorite_objects)))
        for col, obj in zip(cols, favorite_objects):
            with col:
                object_card(obj, list(OBJECTS).index(obj.object_id) + 1)
                if st.button(
                    f"Remove {obj.name}",
                    key=f"remove_favorite_{obj.object_id}",
                    use_container_width=True,
                ):
                    st.session_state.favorites.remove(obj.object_id)
                    st.rerun()


# ---------- Data & export ----------
elif page == "Data & Export":
    render_kicker("07 / Data")
    render_section("Data & export", "Local registry only")

    st.info(
        "These exports contain the application's current local registry. "
        "They are not a claim of a complete astronomical catalogue."
    )

    selected_category = st.selectbox("Category", ["All"] + get_categories())
    export_objects = (
        list(OBJECTS.values())
        if selected_category == "All"
        else category_objects(selected_category)
    )

    render_number_card(str(len(export_objects)).zfill(2), "objects in this export")
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
    st.dataframe(rows, use_container_width=True, hide_index=True)

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

    render_section("Last generated audio", "Session metadata")
    if st.session_state.last_audio_metadata:
        st.json(st.session_state.last_audio_metadata)
    else:
        st.caption("No audio generated in this session yet.")


# ---------- About ----------
elif page == "About & Scientific Method":
    render_hero(
        "Scientific method",
        "A transparent interface for exploring what is measured, what is modeled, and what is interpreted.",
        "01",
        "method statement",
    )

    render_kicker("Purpose")
    st.markdown(
        """
        <div style="max-width:900px;font-size:1.15rem;line-height:1.6;">
        Sounds of the Universe is designed to make astronomical exploration
        more accessible through auditory information. It is especially intended
        to support people who are blind or visually impaired while remaining
        explicit about what each sound represents.
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section("Scientific method", "Core pipeline")
    st.markdown(
        """
        <div class="su-card su-card-dark">
            <div style="font-size:1.8rem;line-height:1.2;font-weight:750;">
                Scientific observation → physical interpretation → reproducible auditory mapping
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section("What the application does not claim", "Scope boundary")
    st.write(
        """
        It does not claim that every planet, star, or black hole has a naturally
        audible sound that could be heard by a human listener in space.

        It also does not allow an AI-generated artistic sound to be presented as
        an observational measurement.
        """
    )

    render_section("Current registry", f"{len(OBJECTS)} objects")
    for category in sorted(set(obj.category for obj in OBJECTS.values())):
        objects = category_objects(category)
        with st.expander(f"{category} / {len(objects)} object(s)"):
            for obj in objects:
                st.write(f"{obj.name} — {obj.subtype}")

    render_section("Future expansion", "Designed for growth")
    st.write(
        """
        The registry is designed to expand into larger astronomical catalogues,
        including Solar System catalogues, exoplanets, stars, galaxies,
        spacecraft measurements, plasma-wave data, solar observations,
        pulsar timing, gravitational-wave data, planetary atmospheres,
        and magnetospheric measurements.
        """
    )

    st.caption(
        "The current application is an early scientific prototype, not a complete catalogue of the observable universe."
    )


st.markdown(
    f'<div class="su-footer">{esc(APP_NAME)} / {esc(APP_VERSION)} / scientifically grounded auditory astronomy</div>',
    unsafe_allow_html=True,
)
