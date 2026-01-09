# import os
# import tempfile
# from pathlib import Path

# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go

# # Import your existing functions
# from fibonacci_simulator import (
#     load_and_resample,
#     simulate_live_fibonacci_professional,
#     export_fib_summary_pdf,
#     FibState,
#     LABEL_LEVELS
# )


# # -----------------------------
# # Plotting (Candles + Fib levels + markers)
# # -----------------------------
# def plot_fibonacci_chart(df: pd.DataFrame, symbol: str, legs: list, min_gap: int):
#     """Generate interactive Plotly chart for Fibonacci analysis"""
#     sym_df = df[df["symbol"] == symbol].copy().reset_index(drop=True)

#     # Filter legs (keep only non-quality-rejected)
#     display_legs = []
#     for leg in legs:
#         if getattr(leg, "quality_rejected", False):
#             continue
#         display_legs.append(leg)

#     display_legs = sorted(display_legs, key=lambda x: x.anchor_seq)

#     if not display_legs:
#         st.warning(f"⚠️ No legs to plot for {symbol}")
#         return None

#     # seq -> datetime mapping
#     seq_to_dt = {}
#     if "seq" in sym_df.columns and "datetime" in sym_df.columns:
#         for _, r in sym_df.iterrows():
#             seq_to_dt[int(r["seq"])] = pd.to_datetime(r["datetime"])

#     def seq_to_datetime(seq_val):
#         return seq_to_dt.get(int(seq_val), pd.NaT)

#     # Create figure
#     fig = go.Figure()

#     # Add candlestick
#     fig.add_trace(
#         go.Candlestick(
#             x=sym_df["datetime"],
#             open=sym_df["open"],
#             high=sym_df["high"],
#             low=sym_df["low"],
#             close=sym_df["close"],
#             increasing_line_color="#00ff5f",
#             decreasing_line_color="#ff4d4d",
#             increasing_fillcolor="#00ff5f",
#             decreasing_fillcolor="#ff4d4d",
#             name=symbol,
#         )
#     )

#     palette = ["cyan", "magenta", "yellow", "orange", "lime", "pink", "white", "deepskyblue", "gold"]

#     def leg_color(leg):
#         return palette[(leg.leg_id - 1) % len(palette)]

#     y_min = float(sym_df["low"].min())
#     y_max = float(sym_df["high"].max())

#     for leg in display_legs:
#         color = leg_color(leg)

#         opacity = 0.55
#         dash = "dot"
#         if leg.state == FibState.TARGET_HIT:
#             opacity = 0.90
#             dash = "solid"
#         elif leg.state == FibState.VALIDATED:
#             opacity = 0.80
#             dash = "dash"
#         elif leg.state == FibState.INVALIDATED:
#             opacity = 0.35
#             dash = "dot"

#         # Min-gap shading
#         fig.add_shape(
#             type="rect",
#             x0=seq_to_datetime(leg.anchor_seq),
#             x1=seq_to_datetime(leg.min_gap_satisfied_seq),
#             y0=y_min,
#             y1=y_max,
#             fillcolor="rgba(255, 0, 0, 0.05)",
#             line=dict(width=0),
#             layer="below",
#         )

#         # Anchor marker
#         fig.add_trace(
#             go.Scatter(
#                 x=[seq_to_datetime(leg.anchor_seq)],
#                 y=[leg.anchor_price],
#                 mode="markers+text",
#                 marker=dict(symbol="star", size=18, color=color, line=dict(color="black", width=2)),
#                 text=[f"Fib-{leg.leg_id}"],
#                 textposition="bottom center",
#                 textfont=dict(size=11, color=color, family="Arial Black"),
#                 showlegend=False,
#             )
#         )

#         fib = leg.get_fib_levels()
#         if not fib:
#             continue

#         # end_seq
#         end_seq = int(sym_df["seq"].max())
#         if leg.state == FibState.TARGET_HIT and leg.target_seq is not None:
#             end_seq = int(leg.target_seq)
#         elif leg.state == FibState.STOPLOSS_HIT and leg.stoploss_seq is not None:
#             end_seq = int(leg.stoploss_seq)
#         elif leg.state == FibState.INVALIDATED and leg.invalidation_seq is not None:
#             end_seq = int(leg.invalidation_seq)

#         idx_leg = display_legs.index(leg)
#         if idx_leg < len(display_legs) - 1:
#             next_anchor = int(display_legs[idx_leg + 1].anchor_seq)
#             if next_anchor > leg.anchor_seq:
#                 end_seq = min(end_seq, next_anchor - 1)

#         anchor_dt = seq_to_datetime(leg.anchor_seq)
#         end_dt = seq_to_datetime(end_seq)

#         # Levels + labels
#         for lv, price in fib.items():
#             width = 1
#             if lv in (0.382, 0.5, 0.618):
#                 width = 2
#             elif lv in (0, 1.0):
#                 width = 2.5
#             elif lv in (1.5, 1.6):
#                 width = 2

#             fig.add_shape(
#                 type="line",
#                 x0=anchor_dt,
#                 x1=end_dt,
#                 y0=float(price),
#                 y1=float(price),
#                 line=dict(color=color, width=width, dash=dash),
#                 opacity=opacity,
#             )

#             if lv in LABEL_LEVELS:
#                 fig.add_annotation(
#                     x=end_dt,
#                     y=float(price),
#                     text=f"Fib-{leg.leg_id}|{lv:.3f}",
#                     showarrow=False,
#                     xanchor="left",
#                     font=dict(size=12, color=color),
#                     bgcolor="rgba(0,0,0,0.0)",
#                     opacity=0.95,
#                 )

#         # Golden zone shading
#         if 0.382 in fib and 0.618 in fib:
#             gz_low = min(float(fib[0.382]), float(fib[0.618]))
#             gz_high = max(float(fib[0.382]), float(fib[0.618]))
#             fig.add_shape(
#                 type="rect",
#                 x0=anchor_dt,
#                 x1=end_dt,
#                 y0=gz_low,
#                 y1=gz_high,
#                 fillcolor="rgba(0, 255, 255, 0.08)",
#                 line=dict(width=0),
#                 layer="below",
#             )

#         # Stoploss zone shading
#         if 0 in fib and 0.382 in fib:
#             sl_low = min(float(fib[0]), float(fib[0.382]))
#             sl_high = max(float(fib[0]), float(fib[0.382]))
#             fig.add_shape(
#                 type="rect",
#                 x0=anchor_dt,
#                 x1=end_dt,
#                 y0=sl_low,
#                 y1=sl_high,
#                 fillcolor="rgba(255, 0, 0, 0.08)",
#                 line=dict(width=0),
#                 layer="below",
#             )

#         # Target zone shading
#         if 1.5 in fib and 1.6 in fib:
#             t_low = min(float(fib[1.5]), float(fib[1.6]))
#             t_high = max(float(fib[1.5]), float(fib[1.6]))
#             fig.add_shape(
#                 type="rect",
#                 x0=anchor_dt,
#                 x1=end_dt,
#                 y0=t_low,
#                 y1=t_high,
#                 fillcolor="rgba(0, 255, 0, 0.08)",
#                 line=dict(width=0),
#                 layer="below",
#             )

#         # Extreme marker
#         if leg.current_extreme_seq is not None and leg.current_extreme_price is not None:
#             text = (leg.locked_extreme_label + " (locked)") if (leg.locked_extreme and leg.locked_extreme_label) else ""
#             fig.add_trace(
#                 go.Scatter(
#                     x=[seq_to_datetime(leg.current_extreme_seq)],
#                     y=[leg.current_extreme_price],
#                     mode="markers+text" if text else "markers",
#                     marker=dict(symbol="diamond", size=14, color=color, line=dict(color="white", width=1)),
#                     text=[text] if text else None,
#                     textposition="top center",
#                     textfont=dict(size=11, color=color),
#                     showlegend=False,
#                 )
#             )

#         # ✅ GZ marker (seq is not dataframe index)
#         gz_seq = leg.validation_info.get("golden_zone_entry_seq")
#         if gz_seq is not None:
#             candle_rows = sym_df[sym_df["seq"] == int(gz_seq)]
#             if not candle_rows.empty:
#                 candle = candle_rows.iloc[0]
#                 candle_low = float(candle["low"])
#                 candle_high = float(candle["high"])
#                 candle_dt = pd.to_datetime(candle["datetime"])

#                 if leg.trend == "UPTREND":
#                     gz_y = candle_low
#                     textpos = "bottom right"
#                 else:
#                     gz_y = candle_high
#                     textpos = "top right"

#                 fig.add_trace(
#                     go.Scatter(
#                         x=[candle_dt],
#                         y=[gz_y],
#                         mode="markers+text",
#                         marker=dict(symbol="circle", size=18, color="lime", line=dict(color="white", width=2)),
#                         text=["GZ"],
#                         textposition=textpos,
#                         textfont=dict(size=12, color="white"),
#                         showlegend=False,
#                     )
#                 )

#         # ENTRY marker
#         if leg.entry_seq is not None:
#             fig.add_trace(
#                 go.Scatter(
#                     x=[seq_to_datetime(leg.entry_seq)],
#                     y=[leg.entry_price],
#                     mode="markers+text",
#                     marker=dict(symbol="triangle-up", size=16, color="deepskyblue", line=dict(color="white", width=1)),
#                     text=["ENTRY"],
#                     textfont=dict(size=11, color="white"),
#                     showlegend=False,
#                 )
#             )

#         # STOPLOSS marker
#         if leg.state == FibState.STOPLOSS_HIT and leg.stoploss_seq is not None:
#             fig.add_trace(
#                 go.Scatter(
#                     x=[seq_to_datetime(leg.stoploss_seq)],
#                     y=[leg.stoploss_price],
#                     mode="markers+text",
#                     marker=dict(symbol="x", size=20, color="red", line=dict(color="white", width=2)),
#                     text=["SL"],
#                     textfont=dict(size=12, color="white"),
#                     showlegend=False,
#                 )
#             )

#         # TARGET marker
#         if leg.state == FibState.TARGET_HIT and leg.target_seq is not None:
#             fig.add_trace(
#                 go.Scatter(
#                     x=[seq_to_datetime(leg.target_seq)],
#                     y=[leg.target_price],
#                     mode="markers+text",
#                     marker=dict(symbol="circle", size=20, color="lime", line=dict(color="white", width=2)),
#                     text=["TGT"],
#                     textfont=dict(size=12, color="white"),
#                     showlegend=False,
#                 )
#             )

#     # Summary
#     valid = sum(1 for x in display_legs if x.validation_info.get("is_valid_fib"))
#     invalidated = sum(1 for x in display_legs if x.state == FibState.INVALIDATED and not getattr(x, "quality_rejected", False))
#     sl_hits = sum(1 for x in display_legs if x.state == FibState.STOPLOSS_HIT)
#     tgt_hits = sum(1 for x in display_legs if x.state == FibState.TARGET_HIT)
#     active = sum(1 for x in display_legs if x.state in (FibState.ACTIVE, FibState.VALIDATED))

#     fig.update_layout(
#         title=dict(
#             text=f"{symbol} | Legs:{len(display_legs)} | Valid:{valid} | SL:{sl_hits} | TGT:{tgt_hits} | Invalid:{invalidated} | Active:{active} | MinGap={min_gap}",
#             font=dict(color="white", size=16),
#         ),
#         xaxis=dict(
#             title="Time",
#             showgrid=True,
#             gridcolor="rgba(128,128,128,0.15)",
#             color="white",
#             type="date",
#         ),
#         yaxis=dict(title="Price", showgrid=True, gridcolor="rgba(128,128,128,0.15)", color="white"),
#         plot_bgcolor="black",
#         paper_bgcolor="black",
#         font=dict(color="white"),
#         xaxis_rangeslider_visible=False,
#         height=800,
#         hovermode="x unified",
#     )

#     return fig


# def main():
#     st.set_page_config(
#         page_title="Fibonacci Retracement Simulator",
#         page_icon="📊",
#         layout="wide",
#         initial_sidebar_state="expanded"
#     )

#     st.markdown("""
#         <style>
#         .main { background-color: #0e1117; }
#         .stButton>button {
#             width: 100%;
#             background-color: #4CAF50;
#             color: white;
#             font-weight: bold;
#             border-radius: 5px;
#             padding: 10px;
#         }
#         .stButton>button:hover { background-color: #45a049; }
#         h1 { color: #4CAF50; }
#         </style>
#     """, unsafe_allow_html=True)

#     st.title("📊 Fibonacci Retracement Simulator")
#     st.markdown("### Professional Trading Analysis Tool")
#     st.markdown("---")

#     # Sidebar
#     with st.sidebar:
#         st.header("⚙️ Configuration")

#         st.subheader("1️⃣ Data Input")
#         uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

#         st.markdown("---")

#         st.subheader("2️⃣ Basic Parameters")
#         resample_timeframe = st.selectbox("Resample Timeframe", ["5min", "10min", "15min", "30min", "1H"], index=0)
#         trend = st.selectbox("Trend Direction", ["UPTREND", "DOWNTREND"], index=0)
#         min_candles = st.number_input("Min Candles (Low to High)", min_value=1, max_value=100, value=2)
#         pivot_period = st.number_input("Pivot Period", min_value=1, max_value=20, value=1)

#         st.markdown("---")

#         st.subheader("3️⃣ Advanced (Optional)")
#         use_manual_anchor = st.checkbox("Use Manual First Anchor")
#         manual_anchor_seq = None
#         if use_manual_anchor:
#             manual_anchor_seq = st.number_input("Manual First Anchor Seq", min_value=0, value=0)

#         use_debug_extreme = st.checkbox("Use Debug Extreme Lock")
#         debug_extreme_index = None
#         if use_debug_extreme:
#             debug_extreme_index = st.number_input("Debug Extreme Pivot Index", min_value=1, value=1)

#     # Small top status (optional, not "File Status section")
#     if uploaded_file is None:
#         st.info("👈 Upload a CSV from the sidebar to begin.")
#     else:
#         st.success(f"✅ File uploaded: **{uploaded_file.name}**")

#     st.markdown("---")

#     # Run
#     if uploaded_file is not None:
#         if st.button("🚀 Run Analysis & Generate PDF", type="primary"):
#             with st.spinner("🔄 Processing..."):
#                 try:
#                     temp_dir = tempfile.mkdtemp()
#                     csv_path = os.path.join(temp_dir, uploaded_file.name)
#                     with open(csv_path, "wb") as f:
#                         f.write(uploaded_file.getbuffer())

#                     # Update simulator globals (no change to simulator file)
#                     import fibonacci_simulator as fib_sim
#                     fib_sim.CSV_FILE = csv_path
#                     fib_sim.RESAMPLE_TIMEFRAME = resample_timeframe
#                     fib_sim.TREND = trend
#                     fib_sim.MIN_CANDLES_LOW_TO_HIGH = int(min_candles)
#                     fib_sim.PIVOT_PERIOD = int(pivot_period)
#                     fib_sim.MANUAL_FIRST_ANCHOR_SEQ = manual_anchor_seq
#                     fib_sim.DEBUG_EXTREME_PIVOT_H_INDEX = debug_extreme_index
#                     fib_sim.DO_PLOT = False

#                     st.info("📊 Loading and resampling data...")
#                     df = load_and_resample(csv_path, resample_timeframe, None)

#                     symbols = df["symbol"].unique()
#                     st.success(f"✅ Loaded {len(df)} candles | Symbols: {list(symbols)}")

#                     all_results = {}
#                     for sym in symbols:
#                         legs = simulate_live_fibonacci_professional(
#                             df=df,
#                             symbol=sym,
#                             trend=trend,
#                             pivot_period=int(pivot_period),
#                             min_gap=int(min_candles),
#                         )

#                         sym_df = df[df["symbol"] == sym].copy().reset_index(drop=True)
#                         pdf_name = f"{Path(uploaded_file.name).stem}_{resample_timeframe}_{trend}_{sym}_report.pdf"
#                         pdf_path = os.path.join(temp_dir, pdf_name)
#                         export_fib_summary_pdf(pdf_path, sym, resample_timeframe, sym_df, legs)

#                         all_results[sym] = {
#                             "legs": legs,
#                             "pdf_path": pdf_path,
#                             "pdf_name": pdf_name,
#                             "df": df,
#                             "df_preview": pd.read_csv(csv_path),  # for File Status section later
#                         }

#                     st.session_state["results"] = all_results
#                     st.session_state["processed"] = True
#                     st.session_state["settings_snapshot"] = {
#                         "timeframe": resample_timeframe,
#                         "trend": trend,
#                         "min_candles": int(min_candles),
#                         "pivot_period": int(pivot_period),
#                         "manual_anchor": manual_anchor_seq if use_manual_anchor else None,
#                         "debug_extreme": debug_extreme_index if use_debug_extreme else None,
#                     }

#                     st.success("✅ Analysis complete!")
#                     # st.balloons()

#                 except Exception as e:
#                     st.error(f"❌ Error during processing: {str(e)}")
#                     import traceback
#                     with st.expander("🔍 View Error Details"):
#                         st.code(traceback.format_exc())

#     # =========================================================
#     # ✅ ORDER YOU WANT:
#     # 1) Analysis Results (Chart)
#     # 2) PDF Report
#     # 3) File Status (moved bottom)
#     # =========================================================

#     # 1) Analysis Results (Chart)
#     if st.session_state.get("processed", False):
#         st.header("📈 Analysis Results")
#         results = st.session_state["results"]

#         for sym, data in results.items():
#             st.subheader(f"Symbol: {sym}")
#             legs = data["legs"]

#             total_legs = len(legs)
#             valid_legs = sum(1 for x in legs if x.validation_info.get("is_valid_fib"))
#             invalid_legs = sum(1 for x in legs if x.state == FibState.INVALIDATED and not getattr(x, "quality_rejected", False))
#             target_hits = sum(1 for x in legs if x.state == FibState.TARGET_HIT)
#             sl_hits = sum(1 for x in legs if x.state == FibState.STOPLOSS_HIT)
#             active_legs = sum(1 for x in legs if x.state in (FibState.ACTIVE, FibState.VALIDATED))

#             c1, c2, c3, c4, c5, c6 = st.columns(6)
#             c1.metric("Total Legs", total_legs)
#             c2.metric("Valid", valid_legs, delta=f"{(valid_legs/total_legs*100) if total_legs > 0 else 0:.1f}%")
#             c3.metric("Invalid", invalid_legs)
#             c4.metric("Target Hits", target_hits, delta="✅")
#             c5.metric("Stoploss Hits", sl_hits, delta="❌")
#             c6.metric("Active", active_legs)

#             st.subheader("📊 Interactive Fibonacci Chart")
#             fig = plot_fibonacci_chart(data["df"], sym, legs, int(st.session_state["settings_snapshot"]["min_candles"]))
#             if fig is not None:
#                 st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.warning("⚠️ No chart data available for this symbol")

#             st.markdown("---")

#         # 2) PDF Report (after charts)
#         st.header("📄 PDF Report")
#         for sym, data in results.items():
#             st.subheader(f"PDF for: {sym}")
#             with open(data["pdf_path"], "rb") as f:
#                 st.download_button(
#                     label=f"📥 Download PDF Report - {sym}",
#                     data=f.read(),
#                     file_name=data["pdf_name"],
#                     mime="application/pdf",
#                     use_container_width=True
#                 )

#         st.markdown("---")

#         # 3) File Status (moved to bottom)
#         st.header("📁 File Status")
#         if uploaded_file is None:
#             st.info("👆 No file uploaded.")
#         else:
#             st.success(f"✅ File uploaded: **{uploaded_file.name}**")

#             # Show settings snapshot (optional)
#             s = st.session_state.get("settings_snapshot", {})
#             with st.expander("⚙️ Settings Used"):
#                 st.markdown(
#                     f"""
# - **Timeframe:** `{s.get("timeframe")}`
# - **Trend:** `{s.get("trend")}`
# - **Min Candles:** `{s.get("min_candles")}`
# - **Pivot Period:** `{s.get("pivot_period")}`
# - **Manual Anchor:** `{s.get("manual_anchor")}`
# - **Debug Extreme:** `{s.get("debug_extreme")}`
# """
#                 )

#             # Preview + columns check
#             # df_preview = st.session_state["results"][list(st.session_state["results"].keys())[0]].get("df_preview")
#             # if df_preview is not None:
#             #     with st.expander("📋 Preview Data (first 10 rows)"):
#             #         st.dataframe(df_preview.head(10), use_container_width=True)

#             #     required_cols = {'open', 'high', 'low', 'close'}
#             #     actual_cols = {col.strip().lower() for col in df_preview.columns}
#             #     missing_cols = required_cols - actual_cols
#             #     if missing_cols:
#             #         st.error(f"❌ Missing required columns: {missing_cols}")
#             #     else:
#             #         st.success("✅ All required columns present (open, high, low, close)")

#     else:
#         # If not processed yet, still keep File Status at bottom (but small)
#         st.header("📁 File Status")
#         if uploaded_file is None:
#             st.info("👆 Please upload a CSV file from the sidebar to begin")
#         else:
#             st.success(f"✅ File uploaded: **{uploaded_file.name}**")


# if __name__ == "__main__":
#     main()





# streamlit_app.py
import os
import re
import base64
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ✅ Import from your fibonacci_simulator.py
from fibonacci_simulator import (
    load_and_resample,
    simulate_live_fibonacci_professional,
    export_fib_summary_pdf,
    compute_pnl_summary,
    FibState,
    FIB_LEVELS,
    LABEL_LEVELS,
)

st.set_page_config(
    page_title="Fibonacci Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)


# -------------------------
# Theme Management
# -------------------------
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

def toggle_theme():
    st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"

def get_theme_colors():
    """Returns color scheme based on current theme"""
    if st.session_state["theme"] == "dark":
        return {
            "bg": "#1a1a1a",
            "paper": "#242424",
            "text": "#d0d0d0",
            "title": "#4a9eff",
            "grid": "rgba(100, 100, 100, 0.2)",
            "candle_up": "#10b981",
            "candle_down": "#ef4444",
            "palette": ["#4a9eff", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981", "#06b6d4", "#6366f1", "#f97316"],
            "gz_marker": "#10b981",
            "entry_marker": "#4a9eff",
            "sl_marker": "#ef4444",
            "tgt_marker": "#10b981",
            "exit_marker": "#f59e0b",
        }
    else:
        return {
            "bg": "#ffffff",
            "paper": "#f8f9fa",
            "text": "#1f2937",
            "title": "#1e40af",
            "grid": "rgba(200, 200, 200, 0.4)",
            "candle_up": "#059669",
            "candle_down": "#dc2626",
            "palette": ["#1e40af", "#7c3aed", "#db2777", "#d97706", "#059669", "#0891b2", "#4f46e5", "#ea580c"],
            "gz_marker": "#059669",
            "entry_marker": "#1e40af",
            "sl_marker": "#dc2626",
            "tgt_marker": "#059669",
            "exit_marker": "#d97706",
        }

def apply_custom_css():
    """Apply theme-specific CSS for entire application"""
    theme = st.session_state["theme"]
    
    if theme == "dark":
        st.markdown("""
            <style>
            /* Main app background */
            .stApp {
                background: #1a1a1a;
            }
            .main {
                background: transparent;
            }
            
            /* Sidebar styling */
            [data-testid="stSidebar"] {
                background: #242424;
                border-right: 1px solid #3a3a3a;
            }
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
                color: #d0d0d0;
            }
            
            /* All text elements */
            .stMarkdown, .stText, p, span, label {
                color: #d0d0d0 !important;
            }
            
            /* Headers */
            h1 {
                color: #4a9eff !important;
                font-size: 2.5rem !important;
                font-weight: 600 !important;
                letter-spacing: -0.5px;
            }
            h2 {
                color: #4a9eff !important;
                font-size: 1.75rem !important;
                font-weight: 600 !important;
            }
            h3, h4, h5, h6 {
                color: #8b9dc3 !important;
                font-weight: 500 !important;
            }
            
            /* Buttons */
            .stButton>button {
                background: #4a9eff !important;
                color: #ffffff !important;
                font-weight: 600;
                border-radius: 6px;
                padding: 12px 24px;
                border: none;
                transition: all 0.2s;
            }
            .stButton>button:hover {
                background: #3b82f6 !important;
                transform: translateY(-1px);
            }
            
            /* Metrics */
            [data-testid="stMetricValue"] {
                font-size: 1.75rem !important;
                font-weight: 600 !important;
                color: #4a9eff !important;
            }
            .stMetric {
                background: #242424 !important;
                padding: 16px;
                border-radius: 8px;
                border: 1px solid #3a3a3a;
            }
            .stMetric label {
                color: #8b9dc3 !important;
                font-weight: 500 !important;
                font-size: 0.875rem !important;
            }
            .stMetric [data-testid="stMetricDelta"] {
                font-weight: 500 !important;
            }
            
            /* Input fields */
            .stTextInput>div>div>input,
            .stNumberInput>div>div>input,
            .stSelectbox>div>div>div {
                background: #2a2a2a !important;
                color: #d0d0d0 !important;
                border: 1px solid #3a3a3a !important;
                border-radius: 6px;
            }
            .stTextInput>div>div>input:focus,
            .stNumberInput>div>div>input:focus,
            .stSelectbox>div>div>div:focus {
                border-color: #4a9eff !important;
            }
            
            /* File uploader */
            [data-testid="stFileUploader"] {
                background: #242424 !important;
                border: 2px dashed #3a3a3a !important;
                border-radius: 8px;
                padding: 20px;
            }
            [data-testid="stFileUploader"]:hover {
                border-color: #4a9eff !important;
            }
            [data-testid="stFileUploader"] label {
                color: #8b9dc3 !important;
                font-weight: 500 !important;
            }
            
            /* Dataframe */
            [data-testid="stDataFrame"] {
                background: #242424 !important;
                border-radius: 8px;
                border: 1px solid #3a3a3a;
            }
            
            /* Alert boxes */
            .stAlert {
                background: #242424 !important;
                border-left: 3px solid #4a9eff !important;
                border-radius: 6px;
                color: #d0d0d0 !important;
            }
            
            /* Divider */
            hr {
                border: none;
                height: 1px;
                background: #3a3a3a;
                margin: 1.5rem 0;
            }
            
            /* Checkbox */
            .stCheckbox label {
                color: #d0d0d0 !important;
                font-weight: 500;
            }
            
            /* Caption */
            .stCaptionContainer {
                color: #8b9dc3 !important;
            }
            
            /* Selectbox dropdown */
            [data-baseweb="select"] > div {
                background: #2a2a2a !important;
                border-color: #3a3a3a !important;
            }
            
            /* Scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            ::-webkit-scrollbar-track {
                background: #1a1a1a;
            }
            ::-webkit-scrollbar-thumb {
                background: #3a3a3a;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #4a4a4a;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            /* Main app background */
            .stApp {
                background: #ffffff;
            }
            .main {
                background: transparent;
            }
            
            /* Sidebar styling */
            [data-testid="stSidebar"] {
                background: #f8f9fa;
                border-right: 1px solid #e5e7eb;
            }
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
                color: #1f2937;
            }
            
            /* All text elements */
            .stMarkdown, .stText, p, span, label {
                color: #1f2937 !important;
            }
            
            /* Headers */
            h1 {
                color: #1e40af !important;
                font-size: 2.5rem !important;
                font-weight: 600 !important;
                letter-spacing: -0.5px;
            }
            h2 {
                color: #1e40af !important;
                font-size: 1.75rem !important;
                font-weight: 600 !important;
            }
            h3, h4, h5, h6 {
                color: #374151 !important;
                font-weight: 500 !important;
            }
            
            /* Buttons */
            .stButton>button {
                background: #1e40af !important;
                color: #ffffff !important;
                font-weight: 600;
                border-radius: 6px;
                padding: 12px 24px;
                border: none;
                transition: all 0.2s;
            }
            .stButton>button:hover {
                background: #1e3a8a !important;
                transform: translateY(-1px);
            }
            
            /* Metrics */
            [data-testid="stMetricValue"] {
                font-size: 1.75rem !important;
                font-weight: 600 !important;
                color: #1e40af !important;
            }
            .stMetric {
                background: #f8f9fa !important;
                padding: 16px;
                border-radius: 8px;
                border: 1px solid #e5e7eb;
            }
            .stMetric label {
                color: #6b7280 !important;
                font-weight: 500 !important;
                font-size: 0.875rem !important;
            }
            .stMetric [data-testid="stMetricDelta"] {
                font-weight: 500 !important;
            }
            
            /* Input fields */
            .stTextInput>div>div>input,
            .stNumberInput>div>div>input,
            .stSelectbox>div>div>div {
                background: #ffffff !important;
                color: #1f2937 !important;
                border: 1px solid #d1d5db !important;
                border-radius: 6px;
            }
            .stTextInput>div>div>input:focus,
            .stNumberInput>div>div>input:focus,
            .stSelectbox>div>div>div:focus {
                border-color: #1e40af !important;
            }
            
            /* File uploader */
            [data-testid="stFileUploader"] {
                background: #f8f9fa !important;
                border: 2px dashed #d1d5db !important;
                border-radius: 8px;
                padding: 20px;
            }
            [data-testid="stFileUploader"]:hover {
                border-color: #1e40af !important;
            }
            [data-testid="stFileUploader"] label {
                color: #6b7280 !important;
                font-weight: 500 !important;
            }
            
            /* Dataframe */
            [data-testid="stDataFrame"] {
                background: #ffffff !important;
                border-radius: 8px;
                border: 1px solid #e5e7eb;
            }
            
            /* Alert boxes */
            .stAlert {
                background: #f8f9fa !important;
                border-left: 3px solid #1e40af !important;
                border-radius: 6px;
                color: #1f2937 !important;
            }
            
            /* Divider */
            hr {
                border: none;
                height: 1px;
                background: #e5e7eb;
                margin: 1.5rem 0;
            }
            
            /* Checkbox */
            .stCheckbox label {
                color: #1f2937 !important;
                font-weight: 500;
            }
            
            /* Caption */
            .stCaptionContainer {
                color: #6b7280 !important;
            }
            
            /* Selectbox dropdown */
            [data-baseweb="select"] > div {
                background: #ffffff !important;
                border-color: #d1d5db !important;
            }
            
            /* Scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            ::-webkit-scrollbar-track {
                background: #f8f9fa;
            }
            ::-webkit-scrollbar-thumb {
                background: #d1d5db;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #9ca3af;
            }
            </style>
        """, unsafe_allow_html=True)

apply_custom_css()


# -------------------------
# Helpers: PDF download + preview
# -------------------------
def get_pdf_download_link(pdf_path: str, filename: str) -> str:
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    b64 = base64.b64encode(pdf_bytes).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF Report</a>'


def display_pdf(pdf_path: str, height: int = 650):
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    b64 = base64.b64encode(pdf_bytes).decode()
    pdf_display = f"""
        <iframe
            src="data:application/pdf;base64,{b64}"
            width="100%"
            height="{height}"
            style="border: none; border-radius: 8px;"
        ></iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)


# -------------------------
# Enhanced Plot builder with theme support
# -------------------------
def build_plotly_chart(sym_df: pd.DataFrame, symbol: str, legs: list) -> go.Figure:
    """
    Professional candlestick chart with Fibonacci levels and markers
    """
    colors = get_theme_colors()
    fig = go.Figure()

    # Enhanced Candlestick
    fig.add_trace(
        go.Candlestick(
            x=sym_df["datetime"],
            open=sym_df["open"],
            high=sym_df["high"],
            low=sym_df["low"],
            close=sym_df["close"],
            increasing_line_color=colors["candle_up"],
            decreasing_line_color=colors["candle_down"],
            increasing_fillcolor=colors["candle_up"],
            decreasing_fillcolor=colors["candle_down"],
            increasing_line_width=2,
            decreasing_line_width=2,
            name=symbol,
            showlegend=False,
        )
    )

    # seq -> datetime mapping
    seq_to_dt = dict(zip(sym_df["seq"].astype(int), pd.to_datetime(sym_df["datetime"])))

    def s2dt(s):
        return seq_to_dt.get(int(s), None)

    def leg_color(leg_id: int) -> str:
        return colors["palette"][(leg_id - 1) % len(colors["palette"])]

    # Draw legs
    legs_sorted = sorted(legs, key=lambda x: x.anchor_seq)
    end_seq_global = int(sym_df["seq"].max())

    for leg in legs_sorted:
        color = leg_color(leg.leg_id)

        # Skip if no fib yet
        fib = leg.get_fib_levels()
        if not fib:
            continue

        # Determine line style based on state
        opacity = 0.7
        dash = "dot"
        line_width = 1.5
        
        if leg.state == FibState.TARGET_HIT:
            opacity = 1.0
            dash = "solid"
            line_width = 3
        elif leg.state == FibState.STOPLOSS_HIT:
            opacity = 0.5
            dash = "dashdot"
            line_width = 1.5
        elif leg.state == FibState.VALIDATED:
            opacity = 0.85
            dash = "dash"
            line_width = 2

        # Decide leg end seq
        end_seq = end_seq_global
        if leg.state == FibState.TARGET_HIT and leg.target_seq is not None:
            end_seq = int(leg.target_seq)
        elif leg.state == FibState.STOPLOSS_HIT and leg.stoploss_seq is not None:
            end_seq = int(leg.stoploss_seq)
        elif leg.state == FibState.INVALIDATED and leg.invalidation_seq is not None:
            end_seq = int(leg.invalidation_seq)
        elif leg.state == FibState.TIME_EXIT and leg.time_exit_seq is not None:
            end_seq = int(leg.time_exit_seq)

        # Prevent overlap with next leg
        i = legs_sorted.index(leg)
        if i < len(legs_sorted) - 1:
            next_anchor = int(legs_sorted[i + 1].anchor_seq)
            if next_anchor > leg.anchor_seq:
                end_seq = min(end_seq, next_anchor - 1)

        anchor_dt = s2dt(leg.anchor_seq)
        end_dt = s2dt(end_seq)

        if anchor_dt is None or end_dt is None:
            continue

        # Enhanced Anchor marker (smaller size)
        fig.add_trace(
            go.Scatter(
                x=[anchor_dt],
                y=[leg.anchor_price],
                mode="markers+text",
                text=[f"Fib-{leg.leg_id}"],
                textposition="bottom center",
                textfont=dict(size=9, color=color, family="Arial"),
                marker=dict(
                    size=10,
                    symbol="star",
                    color=color,
                    line=dict(width=1, color=colors["text"]),
                    opacity=1.0
                ),
                showlegend=False,
                hovertemplate=f"<b>Anchor</b><br>Price: {leg.anchor_price:.2f}<extra></extra>"
            )
        )

        # Fib lines with enhanced styling
        for lv, price in fib.items():
            width = line_width
            if lv in (0.382, 0.5, 0.618):
                width = line_width + 0.5
            elif lv in (0, 1.0):
                width = line_width + 1
            elif lv in (1.5, 1.6):
                width = line_width + 0.5

            fig.add_shape(
                type="line",
                x0=anchor_dt,
                x1=end_dt,
                y0=float(price),
                y1=float(price),
                line=dict(color=color, width=width, dash=dash),
                opacity=opacity,
            )

            # Enhanced labels
            if lv in LABEL_LEVELS:
                label_bg = colors["paper"] if st.session_state["theme"] == "dark" else "rgba(255,255,255,0.9)"
                fig.add_annotation(
                    x=end_dt,
                    y=float(price),
                    text=f"Fib-{leg.leg_id} | {lv:.3f}",
                    showarrow=False,
                    xanchor="left",
                    font=dict(size=9, color=color, family="Arial"),
                    bgcolor=label_bg,
                    bordercolor=color,
                    borderwidth=1,
                    borderpad=2,
                    opacity=0.9,
                )

        # Enhanced Golden zone shading
        if 0.382 in fib and 0.618 in fib:
            gz_low = min(float(fib[0.382]), float(fib[0.618]))
            gz_high = max(float(fib[0.382]), float(fib[0.618]))
            gz_color = "rgba(0,255,255,0.12)" if st.session_state["theme"] == "dark" else "rgba(0,200,200,0.15)"
            fig.add_shape(
                type="rect",
                x0=anchor_dt,
                x1=end_dt,
                y0=gz_low,
                y1=gz_high,
                fillcolor=gz_color,
                line=dict(width=0),
                layer="below",
            )

        # Stoploss zone
        if 0 in fib and 0.382 in fib:
            sl_low = min(float(fib[0]), float(fib[0.382]))
            sl_high = max(float(fib[0]), float(fib[0.382]))
            sl_color = "rgba(255,0,0,0.12)" if st.session_state["theme"] == "dark" else "rgba(255,0,0,0.15)"
            fig.add_shape(
                type="rect",
                x0=anchor_dt,
                x1=end_dt,
                y0=sl_low,
                y1=sl_high,
                fillcolor=sl_color,
                line=dict(width=0),
                layer="below",
            )

        # Target zone
        if 1.5 in fib and 1.6 in fib:
            t_low = min(float(fib[1.5]), float(fib[1.6]))
            t_high = max(float(fib[1.5]), float(fib[1.6]))
            t_color = "rgba(0,255,0,0.12)" if st.session_state["theme"] == "dark" else "rgba(76,175,80,0.15)"
            fig.add_shape(
                type="rect",
                x0=anchor_dt,
                x1=end_dt,
                y0=t_low,
                y1=t_high,
                fillcolor=t_color,
                line=dict(width=0),
                layer="below",
            )

        # Enhanced Markers: GZ / ENTRY / SL / TGT / EXIT (all smaller)
        gz_seq = leg.validation_info.get("golden_zone_entry_seq")
        if gz_seq is not None and s2dt(gz_seq) is not None:
            gz_row = sym_df[sym_df["seq"] == int(gz_seq)]
            if not gz_row.empty:
                fig.add_trace(
                    go.Scatter(
                        x=[s2dt(gz_seq)],
                        y=[gz_row["low"].iloc[0]],
                        mode="markers+text",
                        text=["GZ"],
                        textposition="bottom center",
                        textfont=dict(size=8, color="white", family="Arial"),
                        marker=dict(size=10, symbol="circle", color=colors["gz_marker"], line=dict(width=1, color=colors["text"])),
                        showlegend=False,
                        hovertemplate="<b>Golden Zone Entry</b><extra></extra>"
                    )
                )

        if leg.entry_seq is not None and s2dt(leg.entry_seq) is not None:
            fig.add_trace(
                go.Scatter(
                    x=[s2dt(leg.entry_seq)],
                    y=[leg.entry_price],
                    mode="markers+text",
                    text=["ENTRY"],
                    textposition="top center",
                    textfont=dict(size=8, color="white", family="Arial"),
                    marker=dict(size=10, symbol="triangle-up", color=colors["entry_marker"], line=dict(width=1, color=colors["text"])),
                    showlegend=False,
                    hovertemplate=f"<b>Entry</b><br>Price: {leg.entry_price:.2f}<extra></extra>"
                )
            )

        if leg.state == FibState.STOPLOSS_HIT and leg.stoploss_seq is not None and s2dt(leg.stoploss_seq) is not None:
            fig.add_trace(
                go.Scatter(
                    x=[s2dt(leg.stoploss_seq)],
                    y=[leg.stoploss_price],
                    mode="markers+text",
                    text=["SL"],
                    textposition="top center",
                    textfont=dict(size=8, color="white", family="Arial"),
                    marker=dict(size=12, symbol="x", color=colors["sl_marker"], line=dict(width=2, color=colors["text"])),
                    showlegend=False,
                    hovertemplate=f"<b>Stoploss Hit</b><br>Price: {leg.stoploss_price:.2f}<extra></extra>"
                )
            )

        if leg.state == FibState.TARGET_HIT and leg.target_seq is not None and s2dt(leg.target_seq) is not None:
            fig.add_trace(
                go.Scatter(
                    x=[s2dt(leg.target_seq)],
                    y=[leg.target_price],
                    mode="markers+text",
                    text=["TGT"],
                    textposition="top center",
                    textfont=dict(size=8, color="white", family="Arial"),
                    marker=dict(size=12, symbol="star", color=colors["tgt_marker"], line=dict(width=2, color=colors["text"])),
                    showlegend=False,
                    hovertemplate=f"<b>Target Hit!</b><br>Price: {leg.target_price:.2f}<extra></extra>"
                )
            )

        if leg.state == FibState.TIME_EXIT and leg.time_exit_seq is not None and s2dt(leg.time_exit_seq) is not None:
            fig.add_trace(
                go.Scatter(
                    x=[s2dt(leg.time_exit_seq)],
                    y=[leg.time_exit_price],
                    mode="markers+text",
                    text=["EXIT"],
                    textposition="top center",
                    textfont=dict(size=8, color="white", family="Arial"),
                    marker=dict(size=10, symbol="square", color=colors["exit_marker"], line=dict(width=1, color=colors["text"])),
                    showlegend=False,
                    hovertemplate=f"<b>Time Exit</b><br>Price: {leg.time_exit_price:.2f}<extra></extra>"
                )
            )

    # Enhanced layout
    fig.update_layout(
        title=dict(
            text=f"{symbol} — Fibonacci Retracement Analysis",
            font=dict(color=colors["title"], size=18, family="Arial"),
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            title="Time",
            showgrid=True,
            gridcolor=colors["grid"],
            gridwidth=1,
            color=colors["text"],
            type="date",
            tickfont=dict(size=11),
            showline=True,
            linewidth=1,
            linecolor=colors["grid"],
        ),
        yaxis=dict(
            title="Price",
            showgrid=True,
            gridcolor=colors["grid"],
            gridwidth=1,
            color=colors["text"],
            tickfont=dict(size=11),
            showline=True,
            linewidth=1,
            linecolor=colors["grid"],
        ),
        plot_bgcolor=colors["bg"],
        paper_bgcolor=colors["paper"],
        font=dict(color=colors["text"], family="Arial"),
        xaxis_rangeslider_visible=False,
        height=900,
        hovermode="x unified",
        hoverlabel=dict(bgcolor=colors["paper"], font_size=12, font_family="Arial"),
        margin=dict(l=80, r=80, t=80, b=80),
    )
    
    return fig


# -------------------------
# Header with Theme Toggle
# -------------------------
col_title, col_theme = st.columns([5, 1])
with col_title:
    st.markdown("# Professional Fibonacci Simulator")
with col_theme:
    theme_icon = "☀" if st.session_state["theme"] == "dark" else "☾"
    if st.button(f"{theme_icon}", key="theme_toggle_btn", use_container_width=True):
        toggle_theme()
        st.rerun()


# -------------------------
# Sidebar UI
# -------------------------
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Run button immediately after upload
run_btn = st.sidebar.button("Run Analysis", type="primary", use_container_width=True, disabled=(uploaded_file is None))

st.sidebar.divider()
st.sidebar.subheader("Basic Parameters")
resample_timeframe = st.sidebar.selectbox(
    "Resample Timeframe",
    ["5min", "10min", "15min", "30min", "60min"],
    index=0,
)

trend = st.sidebar.selectbox("Trend Direction", ["UPTREND", "DOWNTREND"], index=0)
min_candles = st.sidebar.number_input("Min Candles (Low→High)", min_value=1, value=2, step=1)
pivot_period = st.sidebar.number_input("Pivot Period", min_value=1, value=1, step=1)

st.sidebar.divider()
st.sidebar.subheader("Advanced (Optional)")
use_manual_anchor = st.sidebar.checkbox("Use Manual First Anchor", value=False)
manual_anchor_seq = st.sidebar.number_input("Manual Anchor Seq", min_value=0, value=0, step=1, disabled=not use_manual_anchor)

use_debug_extreme = st.sidebar.checkbox("Lock Extreme Pivot (Debug)", value=False)
debug_extreme_index = st.sidebar.number_input("Extreme Index (1=H1/L1)", min_value=1, value=1, step=1, disabled=not use_debug_extreme)


# -------------------------
# Main content
# -------------------------
if "processed" not in st.session_state:
    st.session_state["processed"] = False

if run_btn:
    if uploaded_file is None:
        st.error("Please upload a CSV file first.")
    else:
        status = st.empty()
        status.info("Processing...")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                csv_path = os.path.join(temp_dir, uploaded_file.name)
                with open(csv_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Load + resample
                df = load_and_resample(csv_path, resample_timeframe, base_date=None)
                symbols = df["symbol"].unique().tolist()

                # Apply settings
                import fibonacci_simulator as fibmod
                fibmod.MANUAL_FIRST_ANCHOR_SEQ = int(manual_anchor_seq) if use_manual_anchor else None
                fibmod.DEBUG_EXTREME_PIVOT_H_INDEX = int(debug_extreme_index) if use_debug_extreme else None
                fibmod.RESAMPLE_TIMEFRAME = resample_timeframe

                all_results = {}

                for sym in symbols:
                    legs = simulate_live_fibonacci_professional(
                        df=df,
                        symbol=sym,
                        trend=trend,
                        pivot_period=int(pivot_period),
                        min_gap=int(min_candles),
                    )

                    # PnL summary
                    pnl = compute_pnl_summary(legs)

                    sym_df = df[df["symbol"] == sym].copy().reset_index(drop=True)

                    pdf_name = f"{Path(uploaded_file.name).stem}_{resample_timeframe}_{trend}_{sym}_report.pdf"
                    pdf_path = os.path.join(temp_dir, pdf_name)
                    export_fib_summary_pdf(pdf_path, sym, resample_timeframe, sym_df, legs)

                    # Persist PDF
                    stable_pdf = os.path.join(os.getcwd(), f"_latest_{pdf_name}")
                    with open(pdf_path, "rb") as src, open(stable_pdf, "wb") as dst:
                        dst.write(src.read())

                    all_results[sym] = {
                        "legs": legs,
                        "pnl": pnl,
                        "pdf_path": stable_pdf,
                        "pdf_name": pdf_name,
                        "df_resampled": df,
                        "sym_df_resampled": sym_df,
                        "df_preview": pd.read_csv(csv_path),
                    }

                st.session_state["results"] = all_results
                st.session_state["processed"] = True
                st.session_state["settings_snapshot"] = {
                    "timeframe": resample_timeframe,
                    "trend": trend,
                    "min_candles": int(min_candles),
                    "pivot_period": int(pivot_period),
                    "manual_anchor": int(manual_anchor_seq) if use_manual_anchor else None,
                    "debug_extreme": int(debug_extreme_index) if use_debug_extreme else None,
                }

                status.empty()
                st.success("Analysis complete")

        except Exception as e:
            status.empty()
            st.error(f"Analysis failed: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())


# -------------------------
# Render results
# -------------------------
if not st.session_state.get("processed", False):
    st.info("Upload a CSV file from the sidebar and click Run Analysis to begin.")
else:
    results = st.session_state["results"]
    symbols = list(results.keys())

    st.caption(f"Settings: {st.session_state['settings_snapshot']}")
    st.divider()

    selected_sym = st.selectbox("Select Symbol", symbols, index=0)
    res = results[selected_sym]

    legs = res["legs"]
    pnl = res["pnl"]
    sym_df = res["sym_df_resampled"]

    # PnL Summary with enhanced styling
    st.subheader("P&L Summary (Points)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Profit", f"{pnl['total_profit_points']:.2f}")
    col2.metric("Total Loss", f"{pnl['total_loss_points']:.2f}", delta_color="inverse")
    col3.metric("Net Points", f"{pnl['net_points']:.2f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Win Count", int(pnl["win_count"]))
    col5.metric("Loss Count", int(pnl["loss_count"]))
    col6.metric("Trades Closed", int(pnl["trade_count"]))

    col7, col8 = st.columns(2)
    col7.metric("Avg Win", f"{pnl['avg_win_points']:.2f}")
    col8.metric("Avg Loss", f"{pnl['avg_loss_points']:.2f}")

    st.divider()

    # Chart with theme toggle visible
    st.subheader("Interactive Chart")
    st.caption(f"Current Theme: {st.session_state['theme'].title()}")
    
    with st.spinner("Generating chart..."):
        fig = build_plotly_chart(sym_df, selected_sym, legs)
        st.plotly_chart(fig, use_container_width=True)
    
    st.success("Chart loaded successfully")

    st.divider()

    # PDF Report
    st.subheader("PDF Report")
    st.markdown(get_pdf_download_link(res["pdf_path"], res["pdf_name"]), unsafe_allow_html=True)
    st.divider()
    # display_pdf(res["pdf_path"], height=700)

    # st.divider()

    # # Legs table
    # st.subheader("Detailed Legs Summary")
    # rows = []
    # for leg in legs:
    #     rows.append(
    #         {
    #             "Leg ID": leg.leg_id,
    #             "State": leg.state.name,
    #             "Anchor Seq": leg.anchor_seq,
    #             "Anchor Price": f"{leg.anchor_price:.2f}",
    #             "GZ Seq": leg.validation_info.get("golden_zone_entry_seq"),
    #             "Entry Seq": leg.entry_seq,
    #             "Entry Price": f"{leg.entry_price:.2f}" if leg.entry_price else "-",
    #             "SL Seq": leg.stoploss_seq,
    #             "SL Price": f"{leg.stoploss_price:.2f}" if leg.stoploss_price else "-",
    #             "Target Seq": leg.target_seq,
    #             "Target Price": f"{leg.target_price:.2f}" if leg.target_price else "-",
    #             "Exit Seq": leg.time_exit_seq,
    #             "Exit Price": f"{leg.time_exit_price:.2f}" if leg.time_exit_price else "-",
    #         }
    #     )
    # st.dataframe(pd.DataFrame(rows), use_container_width=True, height=400)
