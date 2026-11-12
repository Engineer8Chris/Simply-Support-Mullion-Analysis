"""
app.py - Streamlit front end for the mullion analysis tool.

Run with:   streamlit run app.py
Keep this file in the same folder as Loads.py, Interpolation.py, Integration.py and myquad.py.
It only calls your existing functions (plus analyse_fixed for the fully fixed case).
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
from Integration import analyse, analyse_fixed

KPA = 0.001                      # N/mm^2 per kPa
NAVY, TEAL, AMBER, GREY = "#1B2A41", "#2A7F8E", "#D9822B", "#8A94A3"

TYPES = ["Off", "Rectangle", "Triangle", "Trapezoid", "Spline"]
N_LOADS = 3                      # up to three loads superimposed (same or different types)
SUPPORTS = ["Simply supported (pin + roller)", "Fully fixed (both ends)"]
DEFAULT_TABLE = {"Height (mm)": [0.0, 200.0, 500.0, 800.0, 1200.0],
                 "Pressure (kPa)": [0.0, 0.8, 1.5, 1.2, 0.0]}

st.set_page_config(page_title="Mullion analysis", layout="wide")
st.markdown("<style>[data-testid='stSidebar']{min-width:380px;max-width:380px}</style>", unsafe_allow_html=True)  # roomier sidebar for the table


# ---------------------------------------------------------------- calculation
@st.cache_data(show_spinner="Integrating with the refined trapezoidal rule...")
def compute(H, E, I, npts, support, loads):
    """Build each load case, add them, and run the project's analyse() (or analyse_fixed())."""
    x = np.linspace(0, H, npts)
    parts, rows, notes = [], [], []

    for k, c in enumerate(loads, start=1):
        kind = c["type"]
        if kind == "Off":
            continue
        label = f"Load {k}: {kind}"
        a = None

        if kind == "Rectangle":
            load = rectangle(x, c["width"], c["p"] * KPA)

        elif kind == "Triangle":
            a_, w_peak = tributary(c["p"] * KPA, c["width"], c["height"])
            load = triangle(x, H, w_peak)

        elif kind == "Trapezoid":
            a, w_peak = tributary(c["p"] * KPA, c["width"], c["height"])
            load = trapezoid(x, H, a, w_peak)
            notes.append(f"Load {k} (Trapezoid): a = {a:.1f} mm, w_peak = {w_peak:.4f} N/mm")

        else:  # Spline
            pressures = [p * KPA for p in c["pressures"]]
            load = custom_load(x, c["heights"], pressures, c["width"])

        parts.append((label, load))
        rows.append({"Load": label, "Peak (N/mm)": float(load.max()),
                     "a (mm)": a, "Total force (N)": float(np.sum((load[1:] + load[:-1]) / 2 * np.diff(x)))})

    y = np.sum([load for _, load in parts], axis=0)

    solver = analyse if support == SUPPORTS[0] else analyse_fixed
    log = io.StringIO()                              # analyse() prints and plots; capture both
    with redirect_stdout(log):
        R_A, S, BM, SL, DFL = solver(x, y, H, E, I)
    plt.close("all")

    return dict(x=x, parts=parts, rows=rows, y=y, R_A=R_A, S=S, BM=BM, SL=SL, DFL=DFL,
                log="\n".join(notes + [""] * bool(notes)) + log.getvalue())


# ---------------------------------------------------------------- sidebar inputs
st.title("Mullion analysis")
st.caption("Mullion under superimposed wind loads. Shear, moment, slope and deflection come from "
           "repeated refined trapezoidal-rule integration; curved pressure profiles use a natural cubic spline.")

# Load types sit OUTSIDE the form so choosing "Spline" shows its table straight away
st.sidebar.subheader("Load types")
st.sidebar.caption("Up to three loads, added together. Any type can be used more than once, e.g. two trapezoids.")
kinds = [st.sidebar.selectbox(f"Load {k} type", TYPES, index=TYPES.index("Spline") if k == 1 else 0, key=f"l{k}_type")
         for k in range(1, N_LOADS + 1)]

with st.sidebar.form("inputs"):
    st.subheader("Beam")
    H = st.number_input("Span H (mm)", min_value=100.0, max_value=20000.0, value=1200.0, step=50.0)
    E = st.number_input("Elastic modulus E (MPa)", min_value=1000.0, max_value=300000.0, value=70000.0, step=1000.0)
    I_e6 = st.number_input("Second moment of area I (x10^6 mm^4)", min_value=0.001, max_value=5000.0,
                           value=2.06, step=0.1, format="%.3f")
    npts = st.slider("Points along the span", 50, 400, 200, step=10)
    support = st.radio("Restraints", SUPPORTS, key="support")

    st.subheader("Load inputs")

    slots = []
    for k in range(1, N_LOADS + 1):
        with st.expander(f"Load {k}", expanded=(k == 1)):
            kind = kinds[k - 1]
            st.caption(f"Type: {kind}")
            p = st.number_input("Pressure (kPa)", value=1.0, step=0.1, key=f"l{k}_p")
            width = st.number_input("Width (mm)", value=1000.0, step=50.0, key=f"l{k}_w")
            st.caption("Width: tributary width for rectangle and spline, panel width for triangle and trapezoid.")
            height = st.number_input("Panel height (mm)", value=1200.0, step=50.0, key=f"l{k}_h")
            st.caption("Panel height is only used by triangle and trapezoid.")
            table = pd.DataFrame(DEFAULT_TABLE)           # placeholder when this load is not a spline
            if kind == "Spline":                           # pressure table only appears for a spline load
                table = st.data_editor(
                    table,
                    num_rows="dynamic", hide_index=True, key=f"l{k}_table",
                    column_config={
                        "Height (mm)": st.column_config.NumberColumn("Height (mm)", width="small", format="%.0f"),
                        "Pressure (kPa)": st.column_config.NumberColumn("kPa", width="small", format="%.2f"),
                    },
                )
                st.caption("Enter suction as a positive magnitude: custom_load sets negative pressures to zero.")
        slots.append(dict(type=kind, p=p, width=width, height=height, table=table))

    st.form_submit_button("Run analysis", type="primary")


# ---------------------------------------------------------------- validation
problems, warnings = [], []
loads = []

for k, s in enumerate(slots, start=1):
    data = s["table"].dropna().sort_values("Height (mm)")
    heights = data["Height (mm)"].tolist()
    pressures = data["Pressure (kPa)"].tolist()

    if s["type"] == "Spline":
        if len(heights) < 2:
            problems.append(f"Load {k}: the spline table needs at least two rows.")
        elif len(set(heights)) != len(heights):
            problems.append(f"Load {k}: two rows in the spline table have the same height.")
        else:
            if heights[0] > 0 or heights[-1] < H:
                warnings.append(f"Load {k}: spline heights should run from 0 to H, otherwise the curve is extrapolated.")
            if min(pressures) < 0:
                warnings.append(f"Load {k}: negative pressures in the spline table are set to zero by custom_load.")

    if s["type"] == "Trapezoid" and min(s["width"], s["height"]) / 2 > H / 2:
        warnings.append(f"Load {k}: half the panel's short side is larger than half the span, so the trapezoid is not valid.")

    loads.append(dict(type=s["type"], p=s["p"], width=s["width"], height=s["height"],
                      heights=heights, pressures=pressures))

if all(s["type"] == "Off" for s in slots):
    problems.append("Set at least one load's type (not Off).")

for msg in problems:
    st.error(msg)
for msg in warnings:
    st.warning(msg)
if problems:
    st.stop()

# the cache key only needs the values each load type really uses
for c in loads:
    if c["type"] != "Spline":
        c["heights"], c["pressures"] = [], []
res = compute(H, E, I_e6 * 1e6, npts, support, loads)
x, y, S, BM, DFL = res["x"], res["y"], res["S"], res["BM"], res["DFL"]
R_A, R_B = res["R_A"], -S[-1]                  # shear at x = H is minus the right reaction
fixed = support == SUPPORTS[1]

# ---------------------------------------------------------------- results
iS, iM, iD = np.abs(S).argmax(), np.abs(BM).argmax(), np.abs(DFL).argmax()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Reaction R_A (left)", f"{R_A:,.0f} N")
m2.metric("Reaction R_B (right)", f"{R_B:,.0f} N")
iSp, iSn = S.argmax(), S.argmin()               # largest positive and largest negative shear (one at each support)
m3.metric("Max shear", f"{abs(S[iS]):,.0f} N", f"+{S[iSp]:,.0f} N at x = {x[iSp]:,.0f} mm, {S[iSn]:,.0f} N at x = {x[iSn]:,.0f} mm",
          delta_color="off", delta_arrow="off")
m4.metric("Max moment", f"{abs(BM[iM]) / 1e6:,.3f} kNm", f"{abs(BM[iM]):,.0f} N·mm at x = {x[iM]:,.0f} mm", delta_color="off", delta_arrow="off")
m5.metric("Max deflection", f"{abs(DFL[iD]):,.3f} mm", f"span/{H / abs(DFL[iD]):,.0f} at x = {x[iD]:,.0f} mm", delta_color="off", delta_arrow="off")

if fixed:
    f1, f2, f3, _, _ = st.columns(5)
    f1.metric("Fixed-end moment A", f"{abs(BM[0]) / 1e6:,.3f} kNm", "hogging", delta_color="off", delta_arrow="off")
    f2.metric("Fixed-end moment B", f"{abs(BM[-1]) / 1e6:,.3f} kNm", "hogging", delta_color="off", delta_arrow="off")
    f3.metric("Max span moment", f"{max(BM.max(), 0) / 1e6:,.3f} kNm",
              f"sagging at x = {x[BM.argmax()]:,.0f} mm", delta_color="off", delta_arrow="off")

traps = [r for r in res["rows"] if r["a (mm)"] is not None]
if traps:                                          # 'a' for every trapezoid in use
    cols = st.columns(max(len(traps), 3))
    for col, r in zip(cols, traps):
        col.metric(f"a, {r['Load']}", f"{r['a (mm)']:,.1f} mm", f"w_peak = {r['Peak (N/mm)']:.4f} N/mm",
                   delta_color="off", delta_arrow="off")

if len(res["rows"]) > 1 or traps:
    summary = pd.DataFrame(res["rows"])
    summary["a (mm)"] = summary["a (mm)"].map(lambda v: "" if v is None or pd.isna(v) else f"{v:,.1f}")
    st.dataframe(summary, hide_index=True)

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
ax[1].plot([x[iSp], x[iSn]], [S[iSp], S[iSn]], "o", color=AMBER)   # mark the peak on each side
ax[1].set_ylabel("Shear (N)")
ax[1].set_title("Shear force", loc="left", fontsize=10)

ax[2].plot(x, BM / 1e6, color=NAVY, lw=2)
ax[2].axhline(0, color=GREY, lw=0.8)
ax[2].plot(x[iM], BM[iM] / 1e6, "o", color=AMBER)
ax[2].set_ylabel("Moment (kNm)")
ax[2].set_title("Bending moment" + (" (hogging negative)" if fixed else ""), loc="left", fontsize=10)

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
spline_loads = [(k, c) for k, c in enumerate(loads, start=1) if c["type"] == "Spline"]
if spline_loads:
    with st.expander("Spline pressure profile"):
        for k, c in spline_loads:
            xf = np.linspace(0, H, 300)
            spline = natural_spline(c["heights"], c["pressures"])
            pf, ax2 = plt.subplots(figsize=(8, 2.8))
            ax2.plot(xf, spline(xf), color=TEAL, lw=2, label="natural cubic spline")
            ax2.plot(c["heights"], c["pressures"], "o", color=NAVY, label="data points")
            ax2.axhline(0, color=GREY, lw=0.8)
            ax2.set_title(f"Load {k}", loc="left", fontsize=10)
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

st.caption("Assumptions: " + ("simply supported (pin and roller)" if not fixed else "fully fixed at both ends (no rotation or deflection)")
           + ", constant EI, Euler-Bernoulli beam, small deflection, static transverse load, span = H.")
