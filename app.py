"""
app.py - Streamlit front end for the mullion analysis tool.

Run with:   streamlit run app.py
Keep this file in the same folder as Loads.py, Interpolation.py, Integration.py and myquad.py.
It only calls your existing functions, so none of those files need to change.
Units: N, mm (pressures are typed in kPa and converted: 1 kPa = 0.001 N/mm^2).
"""
import io
from contextlib import redirect_stdout

import matplotlib
matplotlib.use("Agg")            # must be set BEFORE Integration imports pyplot (Streamlit runs off the main thread)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from Loads import rectangle, triangle, trapezoid, tributary, custom_load
from Interpolation import natural_spline
from Integration import analyse

KPA = 0.001                      # N/mm^2 per kPa
NAVY, TEAL, AMBER, GREY = "#1B2A41", "#2A7F8E", "#D9822B", "#8A94A3"

st.set_page_config(page_title="Mullion analysis", layout="wide")
st.markdown("<style>[data-testid='stSidebar']{min-width:380px;max-width:380px}</style>", unsafe_allow_html=True)  # roomier sidebar for the table


# ---------------------------------------------------------------- calculation
@st.cache_data(show_spinner="Integrating with the refined trapezoidal rule...")
def compute(H, E, I, npts, cases):
    """Build each load case, add them, and run the project's analyse()."""
    x = np.linspace(0, H, npts)
    parts = []

    c = cases["rect"]
    if c["on"]:
        parts.append(("Rectangle", rectangle(x, c["width"], c["p"] * KPA)))

    c = cases["tri"]
    if c["on"]:
        a, w_peak = tributary(c["p"] * KPA, c["pw"], c["ph"])
        parts.append(("Triangle", triangle(x, H, w_peak)))

    c = cases["trap"]
    if c["on"]:
        a, w_peak = tributary(c["p"] * KPA, c["pw"], c["ph"])
        parts.append(("Trapezoid", trapezoid(x, H, a, w_peak)))

    c = cases["spline"]
    if c["on"]:
        pressures = [p * KPA for p in c["pressures"]]
        parts.append(("Spline profile", custom_load(x, c["heights"], pressures, c["width"])))

    y = np.sum([load for _, load in parts], axis=0)

    log = io.StringIO()                              # analyse() prints and plots; capture both
    with redirect_stdout(log):
        R_A, S, BM, SL, DFL = analyse(x, y, H, E, I)
    plt.close("all")

    return dict(x=x, parts=parts, y=y, R_A=R_A, S=S, BM=BM, SL=SL, DFL=DFL, log=log.getvalue())


# ---------------------------------------------------------------- sidebar inputs
st.title("Mullion analysis")
st.caption("Simply supported mullion under superimposed wind loads. Shear, moment, slope and deflection come from "
           "repeated refined trapezoidal-rule integration; curved pressure profiles use a natural cubic spline.")

with st.sidebar.form("inputs"):
    st.subheader("Beam")
    H = st.number_input("Span H (mm)", min_value=100.0, max_value=20000.0, value=1200.0, step=50.0)
    E = st.number_input("Elastic modulus E (MPa)", min_value=1000.0, max_value=300000.0, value=70000.0, step=1000.0)
    I_e6 = st.number_input("Second moment of area I (x10^6 mm^4)", min_value=0.001, max_value=5000.0,
                           value=2.06, step=0.1, format="%.3f")
    npts = st.slider("Points along the span", 50, 400, 200, step=10)

    st.subheader("Load cases (added together)")

    with st.expander("Rectangle (uniform)"):
        rect_on = st.checkbox("Include", key="rect_on")
        rect_p = st.number_input("Pressure (kPa)", value=1.0, step=0.1, key="rect_p")
        rect_w = st.number_input("Tributary width (mm)", value=1000.0, step=50.0, key="rect_w")

    with st.expander("Triangle"):
        tri_on = st.checkbox("Include", key="tri_on")
        tri_p = st.number_input("Pressure (kPa)", value=1.0, step=0.1, key="tri_p")
        tri_pw = st.number_input("Panel width (mm)", value=1000.0, step=50.0, key="tri_pw")
        tri_ph = st.number_input("Panel height (mm)", value=1200.0, step=50.0, key="tri_ph")

    with st.expander("Trapezoid"):
        trap_on = st.checkbox("Include", key="trap_on")
        trap_p = st.number_input("Pressure (kPa)", value=1.0, step=0.1, key="trap_p")
        trap_pw = st.number_input("Panel width (mm)", value=1000.0, step=50.0, key="trap_pw")
        trap_ph = st.number_input("Panel height (mm)", value=1200.0, step=50.0, key="trap_ph")

    with st.expander("Spline pressure profile", expanded=True):
        spl_on = st.checkbox("Include", value=True, key="spl_on")
        table = st.data_editor(
            pd.DataFrame({"Height (mm)": [0.0, 200.0, 500.0, 800.0, 1200.0],
                          "Pressure (kPa)": [0.0, 0.8, 1.5, 1.2, 0.0]}),
            num_rows="dynamic", hide_index=True, key="spl_table",
            column_config={
                "Height (mm)": st.column_config.NumberColumn("Height (mm)", width="small", format="%.0f"),
                "Pressure (kPa)": st.column_config.NumberColumn("kPa", width="small", format="%.2f"),
            },
        )
        spl_w = st.number_input("Tributary width (mm)", value=1000.0, step=50.0, key="spl_w")
        st.caption("Enter suction as a positive magnitude: custom_load sets negative pressures to zero.")

    st.form_submit_button("Run analysis", type="primary")


# ---------------------------------------------------------------- validation
problems, warnings = [], []

data = table.dropna().sort_values("Height (mm)")
heights = data["Height (mm)"].tolist()
pressures = data["Pressure (kPa)"].tolist()

if spl_on:
    if len(heights) < 2:
        problems.append("The spline table needs at least two rows.")
    elif len(set(heights)) != len(heights):
        problems.append("Two rows in the spline table have the same height.")
    else:
        if heights[0] > 0 or heights[-1] < H:
            warnings.append("Spline heights should run from 0 to H, otherwise the curve is extrapolated.")
        if min(pressures) < 0:
            warnings.append("Negative pressures in the spline table are set to zero by custom_load.")

if trap_on and min(trap_pw, trap_ph) / 2 > H / 2:
    warnings.append("Trapezoid: half the panel's short side is larger than half the span, so the shape is not valid.")

if not any([rect_on, tri_on, trap_on, spl_on]):
    problems.append("Include at least one load case.")

for msg in problems:
    st.error(msg)
for msg in warnings:
    st.warning(msg)
if problems:
    st.stop()

cases = dict(
    rect=dict(on=rect_on, p=rect_p, width=rect_w),
    tri=dict(on=tri_on, p=tri_p, pw=tri_pw, ph=tri_ph),
    trap=dict(on=trap_on, p=trap_p, pw=trap_pw, ph=trap_ph),
    spline=dict(on=spl_on, heights=heights, pressures=pressures, width=spl_w),
)
res = compute(H, E, I_e6 * 1e6, npts, cases)
x, y, S, BM, DFL = res["x"], res["y"], res["S"], res["BM"], res["DFL"]
R_A, R_B = res["R_A"], -S[-1]                  # shear at x = H is minus the right reaction

# ---------------------------------------------------------------- results
iS, iM, iD = np.abs(S).argmax(), np.abs(BM).argmax(), np.abs(DFL).argmax()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Reaction R_A (left)", f"{R_A:,.0f} N")
m2.metric("Reaction R_B (right)", f"{R_B:,.0f} N")
m3.metric("Max shear", f"{abs(S[iS]):,.0f} N", f"at x = {x[iS]:,.0f} mm", delta_color="off", delta_arrow="off")
m4.metric("Max moment", f"{abs(BM[iM]) / 1e6:,.3f} kNm", f"{abs(BM[iM]):,.0f} N·mm at x = {x[iM]:,.0f} mm", delta_color="off", delta_arrow="off")
m5.metric("Max deflection", f"{abs(DFL[iD]):,.3f} mm", f"span/{H / abs(DFL[iD]):,.0f} at x = {x[iD]:,.0f} mm", delta_color="off", delta_arrow="off")

fig, ax = plt.subplots(4, 1, sharex=True, figsize=(9, 9.5))

if len(res["parts"]) > 1:
    for label, load in res["parts"]:
        ax[0].plot(x, load, lw=1.2, ls="--", label=label)
    ax[0].legend(loc="upper right", fontsize=8, frameon=False)
ax[0].plot(x, y, color=NAVY, lw=2)
ax[0].fill_between(x, y, color=NAVY, alpha=0.08)
ax[0].set_ylabel("Load (N/mm)")
ax[0].set_title("Total line load", loc="left", fontsize=10)

ax[1].plot(x, S, color=NAVY, lw=2)
ax[1].axhline(0, color=GREY, lw=0.8)
ax[1].plot(x[iS], S[iS], "o", color=AMBER)
ax[1].set_ylabel("Shear (N)")
ax[1].set_title("Shear force", loc="left", fontsize=10)

ax[2].plot(x, BM / 1e6, color=NAVY, lw=2)
ax[2].axhline(0, color=GREY, lw=0.8)
ax[2].plot(x[iM], BM[iM] / 1e6, "o", color=AMBER)
ax[2].set_ylabel("Moment (kNm)")
ax[2].set_title("Bending moment", loc="left", fontsize=10)

ax[3].plot(x, DFL, color=NAVY, lw=2)
ax[3].axhline(0, color=GREY, lw=0.8)
ax[3].plot(x[iD], DFL[iD], "o", color=AMBER)
ax[3].invert_yaxis()                          # positive deflection is downward
ax[3].set_ylabel("Deflection (mm)")
ax[3].set_title("Deflection (downward shown down)", loc="left", fontsize=10)
ax[3].set_xlabel("Position along mullion (mm)")

for a in ax:
    a.grid(alpha=0.25)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------------- extras
if spl_on:
    with st.expander("Spline pressure profile"):
        xf = np.linspace(0, H, 300)
        spline = natural_spline(heights, pressures)
        pf, ax2 = plt.subplots(figsize=(8, 2.8))
        ax2.plot(xf, spline(xf), color=TEAL, lw=2, label="natural cubic spline")
        ax2.plot(heights, pressures, "o", color=NAVY, label="data points")
        ax2.axhline(0, color=GREY, lw=0.8)
        ax2.set_xlabel("Height (mm)")
        ax2.set_ylabel("Pressure (kPa)")
        ax2.legend(frameon=False, fontsize=8)
        ax2.grid(alpha=0.25)
        for side in ("top", "right"):
            ax2.spines[side].set_visible(False)
        pf.tight_layout()
        st.pyplot(pf)
        plt.close(pf)

with st.expander("Printout from analyse()"):
    st.code(res["log"] or "(nothing printed)")

with st.expander("Numbers and CSV download"):
    df = pd.DataFrame({
        "x (mm)": x, "Load (N/mm)": y, "Shear (N)": S, "Moment (N.mm)": BM,
        "Slope (rad)": res["SL"], "Deflection (mm)": DFL,
    })
    st.dataframe(df, hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False), "mullion_results.csv", "text/csv")

st.caption("Assumptions: simply supported, constant EI, Euler-Bernoulli beam, small deflection, static transverse load, span = H.")
