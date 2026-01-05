import os
import tempfile
import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go

# Import your existing functions
from fibonacci_simulator import (
    load_and_resample,
    simulate_live_fibonacci_professional,
    export_fib_summary_pdf,
    FibState,
    FIB_LEVELS,
    LABEL_LEVELS
)


# -----------------------------
# ✅ Chrome-safe PDF preview (Blob URL)
# -----------------------------
# def display_pdf(pdf_path: str):
#     """Chrome-safe PDF viewer using Blob URL"""
#     with open(pdf_path, "rb") as f:
#         pdf_bytes = f.read()

#     b64 = base64.b64encode(pdf_bytes).decode("utf-8")

#     pdf_viewer = f"""
#     <html>
#     <head>
#     <script>
#     function openPDF() {{
#         const base64 = "{b64}";
#         const byteCharacters = atob(base64);
#         const byteNumbers = new Array(byteCharacters.length);
#         for (let i = 0; i < byteCharacters.length; i++) {{
#             byteNumbers[i] = byteCharacters.charCodeAt(i);
#         }}
#         const byteArray = new Uint8Array(byteNumbers);
#         const blob = new Blob([byteArray], {{type: "application/pdf"}});
#         const blobUrl = URL.createObjectURL(blob);
#         document.getElementById("pdfFrame").src = blobUrl;
#     }}
#     window.onload = openPDF;
#     </script>
#     </head>
#     <body style="margin:0">
#         <iframe id="pdfFrame" width="100%" height="800px" style="border:none;"></iframe>
#     </body>
#     </html>
#     """
#     components.html(pdf_viewer, height=820, scrolling=True)


# -----------------------------
# Plotting (Candles + Fib levels + markers)
# -----------------------------
def plot_fibonacci_chart(df: pd.DataFrame, symbol: str, legs: list, min_gap: int):
    """Generate interactive Plotly chart for Fibonacci analysis"""
    sym_df = df[df["symbol"] == symbol].copy().reset_index(drop=True)

    # Filter legs (keep only non-quality-rejected)
    display_legs = []
    for leg in legs:
        if getattr(leg, "quality_rejected", False):
            continue
        display_legs.append(leg)

    display_legs = sorted(display_legs, key=lambda x: x.anchor_seq)

    if not display_legs:
        st.warning(f"⚠️ No legs to plot for {symbol}")
        return None

    # seq -> datetime mapping
    seq_to_dt = {}
    if "seq" in sym_df.columns and "datetime" in sym_df.columns:
        for _, r in sym_df.iterrows():
            seq_to_dt[int(r["seq"])] = pd.to_datetime(r["datetime"])

    def seq_to_datetime(seq_val):
        return seq_to_dt.get(int(seq_val), pd.NaT)

    # Create figure
    fig = go.Figure()

    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=sym_df["datetime"],
            open=sym_df["open"],
            high=sym_df["high"],
            low=sym_df["low"],
            close=sym_df["close"],
            increasing_line_color="#00ff5f",
            decreasing_line_color="#ff4d4d",
            increasing_fillcolor="#00ff5f",
            decreasing_fillcolor="#ff4d4d",
            name=symbol,
        )
    )

    palette = ["cyan", "magenta", "yellow", "orange", "lime", "pink", "white", "deepskyblue", "gold"]

    def leg_color(leg):
        return palette[(leg.leg_id - 1) % len(palette)]

    y_min = float(sym_df["low"].min())
    y_max = float(sym_df["high"].max())

    for leg in display_legs:
        color = leg_color(leg)

        opacity = 0.55
        dash = "dot"
        if leg.state == FibState.TARGET_HIT:
            opacity = 0.90
            dash = "solid"
        elif leg.state == FibState.VALIDATED:
            opacity = 0.80
            dash = "dash"
        elif leg.state == FibState.INVALIDATED:
            opacity = 0.35
            dash = "dot"

        # Min-gap shading
        fig.add_shape(
            type="rect",
            x0=seq_to_datetime(leg.anchor_seq),
            x1=seq_to_datetime(leg.min_gap_satisfied_seq),
            y0=y_min,
            y1=y_max,
            fillcolor="rgba(255, 0, 0, 0.05)",
            line=dict(width=0),
            layer="below",
        )

        # Anchor marker
        fig.add_trace(
            go.Scatter(
                x=[seq_to_datetime(leg.anchor_seq)],
                y=[leg.anchor_price],
                mode="markers+text",
                marker=dict(symbol="star", size=18, color=color, line=dict(color="black", width=2)),
                text=[f"Fib-{leg.leg_id}"],
                textposition="bottom center",
                textfont=dict(size=11, color=color, family="Arial Black"),
                showlegend=False,
            )
        )

        fib = leg.get_fib_levels()
        if not fib:
            continue

        # end_seq
        end_seq = int(sym_df["seq"].max())
        if leg.state == FibState.TARGET_HIT and leg.target_seq is not None:
            end_seq = int(leg.target_seq)
        elif leg.state == FibState.STOPLOSS_HIT and leg.stoploss_seq is not None:
            end_seq = int(leg.stoploss_seq)
        elif leg.state == FibState.INVALIDATED and leg.invalidation_seq is not None:
            end_seq = int(leg.invalidation_seq)

        idx_leg = display_legs.index(leg)
        if idx_leg < len(display_legs) - 1:
            next_anchor = int(display_legs[idx_leg + 1].anchor_seq)
            if next_anchor > leg.anchor_seq:
                end_seq = min(end_seq, next_anchor - 1)

        anchor_dt = seq_to_datetime(leg.anchor_seq)
        end_dt = seq_to_datetime(end_seq)

        # Levels + labels
        for lv, price in fib.items():
            width = 1
            if lv in (0.382, 0.5, 0.618):
                width = 2
            elif lv in (0, 1.0):
                width = 2.5
            elif lv in (1.5, 1.6):
                width = 2

            fig.add_shape(
                type="line",
                x0=anchor_dt,
                x1=end_dt,
                y0=float(price),
                y1=float(price),
                line=dict(color=color, width=width, dash=dash),
                opacity=opacity,
            )

            if lv in LABEL_LEVELS:
                fig.add_annotation(
                    x=end_dt,
                    y=float(price),
                    text=f"Fib-{leg.leg_id}|{lv:.3f}",
                    showarrow=False,
                    xanchor="left",
                    font=dict(size=12, color=color),
                    bgcolor="rgba(0,0,0,0.0)",
                    opacity=0.95,
                )

        # Golden zone shading
        if 0.382 in fib and 0.618 in fib:
            gz_low = min(float(fib[0.382]), float(fib[0.618]))
            gz_high = max(float(fib[0.382]), float(fib[0.618]))
            fig.add_shape(
                type="rect",
                x0=anchor_dt,
                x1=end_dt,
                y0=gz_low,
                y1=gz_high,
                fillcolor="rgba(0, 255, 255, 0.08)",
                line=dict(width=0),
                layer="below",
            )

        # Stoploss zone shading
        if 0 in fib and 0.382 in fib:
            sl_low = min(float(fib[0]), float(fib[0.382]))
            sl_high = max(float(fib[0]), float(fib[0.382]))
            fig.add_shape(
                type="rect",
                x0=anchor_dt,
                x1=end_dt,
                y0=sl_low,
                y1=sl_high,
                fillcolor="rgba(255, 0, 0, 0.08)",
                line=dict(width=0),
                layer="below",
            )

        # Target zone shading
        if 1.5 in fib and 1.6 in fib:
            t_low = min(float(fib[1.5]), float(fib[1.6]))
            t_high = max(float(fib[1.5]), float(fib[1.6]))
            fig.add_shape(
                type="rect",
                x0=anchor_dt,
                x1=end_dt,
                y0=t_low,
                y1=t_high,
                fillcolor="rgba(0, 255, 0, 0.08)",
                line=dict(width=0),
                layer="below",
            )

        # Extreme marker
        if leg.current_extreme_seq is not None and leg.current_extreme_price is not None:
            text = (leg.locked_extreme_label + " (locked)") if (leg.locked_extreme and leg.locked_extreme_label) else ""
            fig.add_trace(
                go.Scatter(
                    x=[seq_to_datetime(leg.current_extreme_seq)],
                    y=[leg.current_extreme_price],
                    mode="markers+text" if text else "markers",
                    marker=dict(symbol="diamond", size=14, color=color, line=dict(color="white", width=1)),
                    text=[text] if text else None,
                    textposition="top center",
                    textfont=dict(size=11, color=color),
                    showlegend=False,
                )
            )

        # ✅ FIXED GZ marker (seq is not dataframe index)
        gz_seq = leg.validation_info.get("golden_zone_entry_seq")
        if gz_seq is not None:
            candle_rows = sym_df[sym_df["seq"] == int(gz_seq)]
            if not candle_rows.empty:
                candle = candle_rows.iloc[0]
                candle_low = float(candle["low"])
                candle_high = float(candle["high"])
                candle_dt = pd.to_datetime(candle["datetime"])

                if leg.trend == "UPTREND":
                    gz_y = candle_low
                    textpos = "bottom right"
                else:
                    gz_y = candle_high
                    textpos = "top right"

                fig.add_trace(
                    go.Scatter(
                        x=[candle_dt],
                        y=[gz_y],
                        mode="markers+text",
                        marker=dict(symbol="circle", size=18, color="lime", line=dict(color="white", width=2)),
                        text=["GZ"],
                        textposition=textpos,
                        textfont=dict(size=12, color="white"),
                        showlegend=False,
                    )
                )

        # ENTRY marker
        if leg.entry_seq is not None:
            fig.add_trace(
                go.Scatter(
                    x=[seq_to_datetime(leg.entry_seq)],
                    y=[leg.entry_price],
                    mode="markers+text",
                    marker=dict(symbol="triangle-up", size=16, color="deepskyblue", line=dict(color="white", width=1)),
                    text=["ENTRY"],
                    textfont=dict(size=11, color="white"),
                    showlegend=False,
                )
            )

        # STOPLOSS marker
        if leg.state == FibState.STOPLOSS_HIT and leg.stoploss_seq is not None:
            fig.add_trace(
                go.Scatter(
                    x=[seq_to_datetime(leg.stoploss_seq)],
                    y=[leg.stoploss_price],
                    mode="markers+text",
                    marker=dict(symbol="x", size=20, color="red", line=dict(color="white", width=2)),
                    text=["SL"],
                    textfont=dict(size=12, color="white"),
                    showlegend=False,
                )
            )

        # TARGET marker
        if leg.state == FibState.TARGET_HIT and leg.target_seq is not None:
            fig.add_trace(
                go.Scatter(
                    x=[seq_to_datetime(leg.target_seq)],
                    y=[leg.target_price],
                    mode="markers+text",
                    marker=dict(symbol="circle", size=20, color="lime", line=dict(color="white", width=2)),
                    text=["TGT"],
                    textfont=dict(size=12, color="white"),
                    showlegend=False,
                )
            )

    # Summary
    valid = sum(1 for x in display_legs if x.validation_info.get("is_valid_fib"))
    invalidated = sum(1 for x in display_legs if x.state == FibState.INVALIDATED and not getattr(x, "quality_rejected", False))
    sl_hits = sum(1 for x in display_legs if x.state == FibState.STOPLOSS_HIT)
    tgt_hits = sum(1 for x in display_legs if x.state == FibState.TARGET_HIT)
    active = sum(1 for x in display_legs if x.state in (FibState.ACTIVE, FibState.VALIDATED))

    fig.update_layout(
        title=dict(
            text=f"{symbol} | Legs:{len(display_legs)} | Valid:{valid} | SL:{sl_hits} | TGT:{tgt_hits} | Invalid:{invalidated} | Active:{active} | MinGap={min_gap}",
            font=dict(color="white", size=16),
        ),
        xaxis=dict(
            title="Time",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.15)",
            color="white",
            type="date",
        ),
        yaxis=dict(title="Price", showgrid=True, gridcolor="rgba(128,128,128,0.15)", color="white"),
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        xaxis_rangeslider_visible=False,
        height=800,
        hovermode="x unified",
    )

    return fig


def main():
    st.set_page_config(
        page_title="Fibonacci Retracement Simulator",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        .main { background-color: #0e1117; }
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 5px;
            padding: 10px;
        }
        .stButton>button:hover { background-color: #45a049; }
        h1 { color: #4CAF50; }
        </style>
    """, unsafe_allow_html=True)

    st.title("📊 Fibonacci Retracement Simulator")
    st.markdown("### Professional Trading Analysis Tool")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("1️⃣ Data Input")
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

        st.markdown("---")

        st.subheader("2️⃣ Basic Parameters")
        resample_timeframe = st.selectbox("Resample Timeframe", ["5min", "10min", "15min", "30min", "1H"], index=0)
        trend = st.selectbox("Trend Direction", ["UPTREND", "DOWNTREND"], index=0)
        min_candles = st.number_input("Min Candles (Low to High)", min_value=1, max_value=100, value=2)
        pivot_period = st.number_input("Pivot Period", min_value=1, max_value=20, value=1)

        st.markdown("---")

        st.subheader("3️⃣ Advanced (Optional)")
        use_manual_anchor = st.checkbox("Use Manual First Anchor")
        manual_anchor_seq = None
        if use_manual_anchor:
            manual_anchor_seq = st.number_input("Manual First Anchor Seq", min_value=0, value=0)

        use_debug_extreme = st.checkbox("Use Debug Extreme Lock")
        debug_extreme_index = None
        if use_debug_extreme:
            debug_extreme_index = st.number_input("Debug Extreme Pivot Index", min_value=1, value=1)

    # Status
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📁 File Status")
        if uploaded_file is None:
            st.info("👆 Please upload a CSV file from the sidebar to begin")
        else:
            st.success(f"✅ File uploaded: **{uploaded_file.name}**")
            try:
                df_preview = pd.read_csv(uploaded_file)
                uploaded_file.seek(0)
                with st.expander("📋 Preview Data (first 10 rows)"):
                    st.dataframe(df_preview.head(10), use_container_width=True)

                required_cols = {'open', 'high', 'low', 'close'}
                actual_cols = {col.strip().lower() for col in df_preview.columns}
                missing_cols = required_cols - actual_cols
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                else:
                    st.success("✅ All required columns present (open, high, low, close)")
            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")

    with col2:
        st.subheader("📊 Current Settings")
        st.markdown(f"""
        - **Timeframe:** `{resample_timeframe}`
        - **Trend:** `{trend}`
        - **Min Candles:** `{min_candles}`
        - **Pivot Period:** `{pivot_period}`
        - **Manual Anchor:** `{manual_anchor_seq if use_manual_anchor else 'None'}`
        - **Debug Extreme:** `{debug_extreme_index if use_debug_extreme else 'None'}`
        """)

    st.markdown("---")

    # Run
    if uploaded_file is not None:
        if st.button("🚀 Run Analysis & Generate PDF", type="primary"):
            with st.spinner("🔄 Processing..."):
                try:
                    temp_dir = tempfile.mkdtemp()
                    csv_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(csv_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    import fibonacci_simulator as fib_sim
                    fib_sim.CSV_FILE = csv_path
                    fib_sim.RESAMPLE_TIMEFRAME = resample_timeframe
                    fib_sim.TREND = trend
                    fib_sim.MIN_CANDLES_LOW_TO_HIGH = min_candles
                    fib_sim.PIVOT_PERIOD = pivot_period
                    fib_sim.MANUAL_FIRST_ANCHOR_SEQ = manual_anchor_seq
                    fib_sim.DEBUG_EXTREME_PIVOT_H_INDEX = debug_extreme_index
                    fib_sim.DO_PLOT = False

                    st.info("📊 Loading and resampling data...")
                    df = load_and_resample(csv_path, resample_timeframe, None)

                    symbols = df["symbol"].unique()
                    st.success(f"✅ Loaded {len(df)} candles | Symbols: {list(symbols)}")

                    all_results = {}
                    for sym in symbols:
                        legs = simulate_live_fibonacci_professional(
                            df=df,
                            symbol=sym,
                            trend=trend,
                            pivot_period=int(pivot_period),
                            min_gap=int(min_candles),
                        )

                        sym_df = df[df["symbol"] == sym].copy().reset_index(drop=True)
                        pdf_name = f"{Path(uploaded_file.name).stem}_{resample_timeframe}_{trend}_{sym}_report.pdf"
                        pdf_path = os.path.join(temp_dir, pdf_name)
                        export_fib_summary_pdf(pdf_path, sym, resample_timeframe, sym_df, legs)

                        all_results[sym] = {
                            "legs": legs,
                            "pdf_path": pdf_path,
                            "pdf_name": pdf_name,
                            "df": df,
                        }

                    st.session_state["results"] = all_results
                    st.session_state["processed"] = True

                    st.success("✅ Analysis complete!")
                    st.balloons()

                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    import traceback
                    with st.expander("🔍 View Error Details"):
                        st.code(traceback.format_exc())

    # # Results
    # if st.session_state.get("processed", False):
    #     st.markdown("---")
    #     st.header("📈 Analysis Results")

    #     results = st.session_state["results"]

    #     for sym, data in results.items():
    #         st.subheader(f"Symbol: {sym}")
    #         legs = data["legs"]

    #         total_legs = len(legs)
    #         valid_legs = sum(1 for x in legs if x.validation_info.get("is_valid_fib"))
    #         invalid_legs = sum(1 for x in legs if x.state == FibState.INVALIDATED and not getattr(x, "quality_rejected", False))
    #         target_hits = sum(1 for x in legs if x.state == FibState.TARGET_HIT)
    #         sl_hits = sum(1 for x in legs if x.state == FibState.STOPLOSS_HIT)
    #         active_legs = sum(1 for x in legs if x.state in (FibState.ACTIVE, FibState.VALIDATED))

    #         c1, c2, c3, c4, c5, c6 = st.columns(6)
    #         c1.metric("Total Legs", total_legs)
    #         c2.metric("Valid", valid_legs, delta=f"{(valid_legs/total_legs*100) if total_legs > 0 else 0:.1f}%")
    #         c3.metric("Invalid", invalid_legs)
    #         c4.metric("Target Hits", target_hits, delta="✅")
    #         c5.metric("Stoploss Hits", sl_hits, delta="❌")
    #         c6.metric("Active", active_legs)

    #         st.markdown("---")
    #         st.subheader("📊 Interactive Fibonacci Chart")
    #         with st.spinner("🎨 Generating interactive chart..."):
    #             fig = plot_fibonacci_chart(data["df"], sym, legs, int(min_candles))
    #             if fig is not None:
    #                 st.plotly_chart(fig, use_container_width=True)
    #             else:
    #                 st.warning("⚠️ No chart data available for this symbol")

    #         st.markdown("---")
    #         st.subheader("📄 PDF Report")

    #         col_pdf1, col_pdf2 = st.columns(2)

    #         with col_pdf1:
    #             if st.button(f"👁️ View PDF Report - {sym}", key=f"view_{sym}"):
    #                 st.session_state[f"show_pdf_{sym}"] = not st.session_state.get(f"show_pdf_{sym}", False)

    #         with col_pdf2:
    #             with open(data["pdf_path"], "rb") as f:
    #                 st.download_button(
    #                     label="📥 Download PDF Report",
    #                     data=f.read(),
    #                     file_name=data["pdf_name"],
    #                     mime="application/pdf",
    #                     use_container_width=True
    #                 )

    #         if st.session_state.get(f"show_pdf_{sym}", False):
    #             st.markdown("### 📄 PDF Report Preview")
    #             display_pdf(data["pdf_path"])

    #         st.markdown("---")

st.subheader("📄 PDF Report")

col_pdf1, col_pdf2 = st.columns(2)

# Open PDF in new tab (browser-native)
with col_pdf1:
    with open(data["pdf_path"], "rb") as f:
        pdf_bytes = f.read()
        b64 = base64.b64encode(pdf_bytes).decode()

        st.markdown(
            f"""
            <a href="data:application/pdf;base64,{b64}" 
               target="_blank"
               style="text-decoration:none;">
               <button style="
                   width:100%;
                   padding:10px;
                   font-weight:bold;
                   background:#4CAF50;
                   color:white;
                   border:none;
                   border-radius:6px;
                   cursor:pointer;">
                   👁️ Open PDF in New Tab
               </button>
            </a>
            """,
            unsafe_allow_html=True
        )

# Download PDF
with col_pdf2:
    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_bytes,
        file_name=data["pdf_name"],
        mime="application/pdf",
        use_container_width=True
    )



if __name__ == "__main__":
    main()

