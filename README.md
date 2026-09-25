# Sounds of the Universe

An accessible astronomy exploration app: explore astronomical objects through
scientifically grounded, deterministic sonification and physics-based audio
models, with a full provenance panel on every object explaining exactly what
kind of audio you're hearing (a real recording, a sonification of measured
data, a physics-based model, or an explicit interpretation).

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

## Search the universe (live internet lookup)

The "Search the universe" page looks up **any** named astronomical object in
real time -- not just the 14 in the curated catalogue -- using two free,
public, no-API-key services:

- **Wikidata** (structured facts: mass, radius, temperature, rotation
  period, and object category, discovered by their live property *labels*
  rather than hard-coded property numbers, so it keeps working even if a
  property ID assumption turns out wrong).
- **Wikipedia** (a short description paragraph only -- never used as a
  source of numeric facts).

This needs outbound internet access from wherever the app runs (it works out
of the box on Streamlit Community Cloud and on a normal local machine; it
will silently fall back to an on-screen "not found" message if the network
is unavailable, rather than crashing).

**Before deploying, set a real contact string** in the `HTTP_HEADERS`
User-Agent near the top of `app.py` -- Wikimedia's API etiquette asks for
one, and requests without it are more likely to be rate-limited.

Every object found this way gets the same honesty treatment as the curated
catalogue: its provenance panel states exactly which physical properties
were actually found on Wikidata and which parts of the sound are a generic,
type-based fallback because a property was missing. Results are cached for
the session and also appear in Explore objects, the Audio laboratory,
Compare objects, Favorites, and Data and export.

## Gemini setup (optional -- "Ask the Cosmos" only)

Everything except the "Ask the Cosmos" question-answering panel works with no
configuration at all: object exploration, the audio laboratory, comparisons,
favorites, and export all run entirely locally with no API key.

To enable "Ask the Cosmos":

1. Get a Gemini API key from Google AI Studio.
2. Locally: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
   and put your key in it. That file is meant to stay out of source control.
3. On Streamlit Community Cloud: open your app's Settings -> Secrets, and
   paste the same `GEMINI_API_KEY = "..."` line there.

If the key is missing or a request fails for any reason, the app never
crashes -- it shows a clear "offline" message in Ask the Cosmos and every
other feature keeps working.

**Verify the model names before deploying.** This build requests
`gemini-3.6-flash`, `gemini-3.1-pro-preview`, and `gemini-3.5-flash-lite`
(defined near the top of `app.py`). Check these against the current
Generative Language API model list -- if a name is wrong or retired, Ask the
Cosmos will simply show its offline fallback rather than breaking anything
else.

## Deploying to Streamlit Community Cloud

1. Push `app.py` and `requirements.txt` to a GitHub repository (do **not**
   push a real `secrets.toml`).
2. Create a new app on Streamlit Community Cloud pointing at that repo and
   `app.py`.
3. Add `GEMINI_API_KEY` under the app's Secrets settings if you want Ask the
   Cosmos enabled.

## What's in this first version

The catalogue ships with 14 objects chosen to demonstrate every provenance
category and every synthesis technique described in the app itself: the Sun,
Mercury, Venus, Earth, Mars, Jupiter, Saturn, Neptune, the Moon, Titan, a
generic pulsar, a generic black-hole merger model, a generic emission nebula,
and a generic spiral galaxy. The app is explicit in its own UI that this is a
growing, incomplete catalogue -- it never claims to cover every object in the
universe.

### Adding a new object

Everything about an object -- its description, physical properties, audio
recipe, provenance label, limitations, and source notes -- lives in one
`ObjectProfile(...)` entry inside `OBJECT_REGISTRY` in `app.py`. Add a new
entry there (see any existing one as a template) and it automatically shows
up in search, the audio laboratory, comparisons, favorites, and export --
no other code changes are needed. The `recipe` field is a list of `Layer`
objects built from the shared primitives (`filtered_noise`, `additive`,
`am`, `fm`, `pulse_train`, `chirp`, `silence_field`) -- reuse those rather
than writing a bespoke synthesis function per object.

### Honesty by construction

Every `ObjectProfile` requires an `audio_provenance` label
(`measured_audio`, `scientific_sonification`, `physics_based_model`, or
`interpretive_representation`), a `provenance_explanation`, a
`scientific_confidence` level, and lists of `limitations` and
`not_represented` facts. These are surfaced automatically in the UI's
provenance panel -- there is no code path that generates audio without
also being able to show where it came from.

## Known limitations of this build

- Numeric physical values (temperatures, pressures, periods) are rounded,
  approximate reference figures for readability and accessibility, not
  precision measurements -- verify against a primary source (e.g. NASA's
  Planetary Fact Sheet) for anything beyond education.
- No real recorded or measured datasets are bundled (no actual Huygens,
  Voyager, Juno, or LIGO data files) -- every object's `available_measured_data`
  and `source_references` fields point to what *exists* publicly, but the
  audio itself is always generated by this app's own deterministic engine,
  as each object's provenance panel explains.
- Accessibility support here relies on Streamlit's own semantic HTML output
  plus this app's text labels and alt-text-equivalent explanations; if you
  need guaranteed WCAG conformance, test with your organization's screen
  reader stack before shipping.
- Live search quality depends entirely on what a given object's Wikidata
  entry actually has filled in. Well-documented objects (planets, bright
  stars, famous exoplanets) usually have several usable numeric properties;
  obscure ones may only have a name and category, in which case the app is
  explicit that it fell back to a generic, type-based model.
- Object-category detection matches Wikidata's "instance of" labels against
  a fixed keyword list (star, planet, moon, asteroid, comet, galaxy,
  nebula, black hole, pulsar/neutron star/white dwarf, star cluster). An
  object classified under an unlisted or unusual category falls back to a
  generic texture rather than guessing.
