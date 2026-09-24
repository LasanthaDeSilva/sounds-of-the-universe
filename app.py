import streamlit as st


# ---------------------------------------------------------
# SOUNDS OF THE UNIVERSE
# Initial Streamlit foundation
# ---------------------------------------------------------

st.set_page_config(
    page_title="Sounds of the Universe",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("Sounds of the Universe")

st.subheader("An auditory exploration of the cosmos")

st.write(
    """
    Explore astronomical objects and environments through
    scientifically grounded sound and sonification.
    """
)


# ---------------------------------------------------------
# Initial navigation
# ---------------------------------------------------------

st.sidebar.title("Explore")

section = st.sidebar.radio(
    "Choose a section",
    [
        "Universe",
        "Astronomical Objects",
        "Environments",
        "Scientific Sonifications",
    ],
)


# ---------------------------------------------------------
# Main content
# ---------------------------------------------------------

if section == "Universe":

    st.header("The Universe")

    st.info(
        "The scientific exploration engine is being prepared. "
        "Astronomical objects, environments, measurements, "
        "and sonifications will be added in later stages."
    )


elif section == "Astronomical Objects":

    st.header("Astronomical Objects")

    st.write(
        """
        This section will contain planets, moons, stars,
        stellar remnants, galaxies, black holes, and other
        astronomical objects.
        """
    )


elif section == "Environments":

    st.header("Astronomical Environments")

    st.write(
        """
        This section will contain scientifically modeled
        environments such as planetary atmospheres,
        magnetospheres, stellar environments, nebulae,
        accretion disks, and other cosmic environments.
        """
    )


elif section == "Scientific Sonifications":

    st.header("Scientific Sonifications")

    st.write(
        """
        This section will contain measured-data
        sonifications and scientifically documented
        audio representations.
        """
    )


# ---------------------------------------------------------
# Accessibility note
# ---------------------------------------------------------

st.divider()

st.caption(
    "Designed with accessibility and auditory exploration "
    "in mind. Scientific provenance will be provided for "
    "audio representations as the system develops."
)
