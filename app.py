"""
AI-Based 8-Puzzle Game Solver
Streamlit Interactive Application

Features:
- Complete Step-by-Step Puzzle Solution with A* Search
- Display initial state, every intermediate state, and goal state
- Clearly numbered steps: Step 0, Step 1, Step 2, ...
- Identify which tile moved and direction of movement at each step
- Distinctly highlighted blank tile and moved tile
- Previous Step and Next Step navigation buttons + Slider
- Animated Auto Play button for automated demonstration
- Heuristic comparison: Manhattan Distance vs Misplaced Tiles
- Side-by-side comparison tables, charts, and solution paths
- Light blue-and-white presentation-ready interface
- Robust Streamlit session state callbacks for Generate Puzzle & Reset
"""

import time
import pandas as pd
import altair as alt
import streamlit as st

from solver import (
    GOAL_STATE,
    DEFAULT_STATE,
    manhattan_distance,
    misplaced_tiles,
    count_inversions,
    is_solvable,
    validate_puzzle_input,
    solve_puzzle,
    generate_random_puzzle,
    SearchResult,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI 8-Puzzle Game Solver | A* Search",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling: Modern Blue & White Palette, Highlighted Blank & Moved Tiles
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main Background and Fonts */
    .main {
        background-color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Title and Header Banner */
    .title-banner {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #3B82F6 100%);
        color: white;
        padding: 22px 28px;
        border-radius: 14px;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.16);
        margin-bottom: 20px;
    }
    .title-banner h1 {
        margin: 0;
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: white !important;
    }
    .title-banner p {
        margin: 5px 0 0 0;
        font-size: 13.5px;
        opacity: 0.92;
        color: #E0E7FF !important;
    }
    
    /* Card Container */
    .custom-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.04);
        margin-bottom: 18px;
    }
    .card-title {
        font-size: 16px;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* 3x3 Puzzle Board Layout */
    .puzzle-board {
        display: grid;
        grid-template-columns: repeat(3, 86px);
        grid-template-rows: repeat(3, 86px);
        gap: 10px;
        justify-content: center;
        padding: 16px;
        background: #F1F5F9;
        border-radius: 14px;
        border: 2px solid #CBD5E1;
        width: fit-content;
        margin: 0 auto;
        box-shadow: inset 0 2px 6px rgba(0,0,0,0.06);
    }
    
    /* Standard Numbered Tiles */
    .tile-number {
        background: linear-gradient(145deg, #2563EB, #1D4ED8);
        color: #FFFFFF;
        font-size: 28px;
        font-weight: 800;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(29, 78, 216, 0.28), 0 2px 4px rgba(0, 0, 0, 0.08);
        border: 1px solid #60A5FA;
        position: relative;
    }

    /* Highlighted Moved Tile */
    .tile-moved {
        background: linear-gradient(145deg, #0284C7, #0369A1) !important;
        color: #FFFFFF !important;
        font-size: 28px;
        font-weight: 800;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        box-shadow: 0 0 14px rgba(14, 165, 233, 0.6), 0 4px 8px rgba(0, 0, 0, 0.15) !important;
        border: 3px solid #38BDF8 !important;
        position: relative;
    }
    .tile-moved-badge {
        font-size: 9px;
        font-weight: 800;
        letter-spacing: 0.5px;
        background: #38BDF8;
        color: #082F49;
        padding: 1px 5px;
        border-radius: 4px;
        margin-top: 2px;
        text-transform: uppercase;
    }
    
    /* Prominently Highlighted Blank Tile (0) */
    .tile-blank-highlighted {
        background: #FEF3C7 !important;
        color: #92400E !important;
        font-size: 13px !important;
        font-weight: 800 !important;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        border: 3px dashed #F59E0B !important;
        box-shadow: 0 0 12px rgba(245, 158, 11, 0.35) !important;
        letter-spacing: 0.5px;
    }
    .tile-blank-sub {
        font-size: 10px;
        font-weight: 700;
        color: #B45309;
        text-transform: uppercase;
    }
    
    /* Metric Cards */
    .metric-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #2563EB;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    }
    .metric-value {
        font-size: 24px;
        font-weight: 800;
        color: #1E3A8A;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 11.5px;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.5px;
        margin-top: 2px;
    }
    
    /* Status Badges */
    .badge-solvable {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 16px;
        background-color: #DCFCE7;
        color: #15803D;
        font-weight: 700;
        font-size: 13px;
        border: 1px solid #86EFAC;
    }
    .badge-unsolvable {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 16px;
        background-color: #FEE2E2;
        color: #B91C1C;
        font-weight: 700;
        font-size: 13px;
        border: 1px solid #FCA5A5;
    }
    .badge-info {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 16px;
        background-color: #DBEAFE;
        color: #1D4ED8;
        font-weight: 700;
        font-size: 13px;
        border: 1px solid #93C5FD;
    }

    /* Step Banner */
    .step-header-box {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 1px solid #BFDBFE;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 14px 0;
        text-align: center;
    }
    .step-title {
        font-size: 18px;
        font-weight: 800;
        color: #1E3A8A;
    }
    .step-detail {
        font-size: 14px;
        font-weight: 600;
        color: #2563EB;
        margin-top: 4px;
    }

    /* Intermediate State Card in Sequence */
    .seq-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        margin-bottom: 12px;
        text-align: center;
    }
    .seq-header {
        font-size: 13px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 6px;
        background: #F1F5F9;
        padding: 4px 8px;
        border-radius: 6px;
    }
    .seq-action {
        font-size: 12px;
        font-weight: 600;
        color: #059669;
        margin-bottom: 8px;
        min-height: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Default Initial State & Session State Initialization
# -----------------------------------------------------------------------------
# Default puzzle: 1 2 3 / 4 5 6 / 7 0 8 (requires exactly 1 move to reach goal)
DEFAULT_INITIAL_PUZZLE = DEFAULT_STATE

for i in range(9):
    if f"tile_in_{i}" not in st.session_state:
        st.session_state[f"tile_in_{i}"] = DEFAULT_INITIAL_PUZZLE[i]

if "solution_result" not in st.session_state:
    st.session_state.solution_result = None

if "comp_res_manhattan" not in st.session_state:
    st.session_state.comp_res_manhattan = None

if "comp_res_misplaced" not in st.session_state:
    st.session_state.comp_res_misplaced = None

if "current_step" not in st.session_state:
    st.session_state.current_step = 0

if "selected_heuristic" not in st.session_state:
    st.session_state.selected_heuristic = "Manhattan Distance"

if "feedback_msg" not in st.session_state:
    st.session_state.feedback_msg = None


# -----------------------------------------------------------------------------
# Streamlit Callbacks (Ensures instant state update without widget conflicts)
# -----------------------------------------------------------------------------
def on_generate_puzzle():
    """Callback to generate a solvable puzzle according to the chosen difficulty."""
    diff_selection = st.session_state.get("difficulty_selector", "Medium (8-14 moves)")
    diff_key = diff_selection.split(" ")[0]
    generated = generate_random_puzzle(diff_key)

    # Update all 9 cell inputs directly in session state BEFORE widgets render
    for idx, val in enumerate(generated):
        st.session_state[f"tile_in_{idx}"] = val

    # Clear previous solutions, comparison results, and step navigation state
    st.session_state.solution_result = None
    st.session_state.comp_res_manhattan = None
    st.session_state.comp_res_misplaced = None
    st.session_state.current_step = 0
    st.session_state.feedback_msg = {
        "type": "success",
        "text": f"🎲 Generated a new solvable {diff_key} puzzle! All input fields updated.",
    }


def on_reset_default():
    """Callback to restore default initial puzzle: 1 2 3 / 4 5 6 / 7 0 8 (1 move from goal)."""
    for idx, val in enumerate(DEFAULT_INITIAL_PUZZLE):
        st.session_state[f"tile_in_{idx}"] = val

    # Clear previous solutions, comparison results, and step navigation state
    st.session_state.solution_result = None
    st.session_state.comp_res_manhattan = None
    st.session_state.comp_res_misplaced = None
    st.session_state.current_step = 0
    st.session_state.feedback_msg = {
        "type": "info",
        "text": "🔄 Reset to default puzzle: 1 2 3 / 4 5 6 / 7 0 8 (requires exactly 1 move to solve).",
    }


def on_tile_input_change():
    """Callback when user edits any of the 9 number input fields manually."""
    st.session_state.solution_result = None
    st.session_state.comp_res_manhattan = None
    st.session_state.comp_res_misplaced = None
    st.session_state.current_step = 0
    st.session_state.feedback_msg = None


# -----------------------------------------------------------------------------
# Helper Functions: HTML Puzzle Board Renderers
# -----------------------------------------------------------------------------
def render_puzzle_html(state: tuple[int, ...] | list[int], moved_tile: int | None = None) -> str:
    """
    Renders 3x3 puzzle board HTML.
    - Numbered tiles are styled with gradient blue.
    - The moved tile at the current step is highlighted with an emerald/cyan border and badge.
    - The blank tile (0) is prominently highlighted in amber/yellow with '0 BLANK'.
    """
    cells_html = ""
    for tile in state:
        if tile == 0:
            cells_html += (
                '<div class="tile-blank-highlighted">'
                '<span>0</span>'
                '<span class="tile-blank-sub">BLANK</span>'
                '</div>'
            )
        elif moved_tile is not None and tile == moved_tile:
            cells_html += (
                f'<div class="tile-moved">'
                f'<span>{tile}</span>'
                f'<span class="tile-moved-badge">MOVED</span>'
                f'</div>'
            )
        else:
            cells_html += f'<div class="tile-number">{tile}</div>'
    return f'<div class="puzzle-board">{cells_html}</div>'


def render_compact_board_html(
    state: tuple[int, ...] | list[int],
    moved_tile: int | None = None,
    cell_size: int = 44,
) -> str:
    """Renders a compact 3x3 board for sequential intermediate state cards."""
    cells_html = ""
    for tile in state:
        if tile == 0:
            cells_html += (
                f'<div style="background:#FEF3C7; color:#B45309; font-size:11px; font-weight:800; '
                f'border-radius:6px; display:flex; flex-direction:column; align-items:center; justify-content:center; '
                f'border:2px dashed #F59E0B;">'
                f'<span>0</span>'
                f'</div>'
            )
        elif moved_tile is not None and tile == moved_tile:
            cells_html += (
                f'<div style="background:linear-gradient(135deg, #0284C7, #0369A1); color:white; font-size:15px; '
                f'font-weight:800; border-radius:6px; display:flex; align-items:center; justify-content:center; '
                f'border:2px solid #38BDF8; box-shadow:0 0 6px rgba(56,189,248,0.6);">'
                f'{tile}'
                f'</div>'
            )
        else:
            cells_html += (
                f'<div style="background:linear-gradient(135deg, #2563EB, #1D4ED8); color:white; font-size:15px; '
                f'font-weight:700; border-radius:6px; display:flex; align-items:center; justify-content:center;">'
                f'{tile}'
                f'</div>'
            )

    board_html = (
        f'<div style="display:grid; grid-template-columns:repeat(3, {cell_size}px); '
        f'grid-template-rows:repeat(3, {cell_size}px); gap:5px; justify-content:center; '
        f'padding:8px; background:#F8FAFC; border-radius:10px; border:1px solid #CBD5E1; margin:0 auto; width:fit-content;">'
        f'{cells_html}'
        f'</div>'
    )
    return board_html


# -----------------------------------------------------------------------------
# Top Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="title-banner">
        <h1>AI-Based 8-Puzzle Game Solver</h1>
        <p>A* Search Algorithm with Manhattan Distance &amp; Misplaced Tiles Heuristics &bull; Complete Step-by-Step Solution</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar: Puzzle Controls (Presets completely removed)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Puzzle Controls")
    st.markdown("Generate a solvable puzzle or restore the default initial configuration.")

    # Random Solvable Generator Section
    st.markdown("##### 🎲 Generate Solvable Puzzle")
    st.selectbox(
        "Difficulty Level",
        ["Easy (2-5 moves)", "Medium (8-14 moves)", "Hard (15-25 moves)", "Random Permutation"],
        index=1,
        key="difficulty_selector",
    )
    st.button(
        "Generate Solvable Puzzle",
        type="primary",
        use_container_width=True,
        on_click=on_generate_puzzle,
    )

    st.markdown("---")

    # Reset to Default Section
    st.markdown("##### 🔄 Reset Puzzle")
    st.button(
        "Reset to Default",
        type="secondary",
        use_container_width=True,
        on_click=on_reset_default,
    )

    st.markdown("---")
    st.markdown("##### 🎯 Fixed Goal State Reference")
    st.markdown(
        """
        ```
        1   2   3
        4   5   6
        7   8   0 (Blank)
        ```
        """
    )


# -----------------------------------------------------------------------------
# Extract Current Puzzle Values from Session State & Validate
# -----------------------------------------------------------------------------
current_puzzle_values = [st.session_state[f"tile_in_{i}"] for i in range(9)]
is_valid, err_msg, state_tuple = validate_puzzle_input(current_puzzle_values)
inversions = count_inversions(state_tuple) if is_valid and state_tuple is not None else 0
solvable = is_solvable(state_tuple) if is_valid and state_tuple is not None else False


# -----------------------------------------------------------------------------
# Tabs Layout: Clear Organization
# -----------------------------------------------------------------------------
tab_solver, tab_comparison = st.tabs(
    ["🧩 Step-by-Step Puzzle Solver", "⚖️ Algorithm Comparison"]
)

# =============================================================================
# TAB 1: STEP-BY-STEP PUZZLE SOLVER
# =============================================================================
with tab_solver:
    # Display any active feedback / confirmation message
    if st.session_state.feedback_msg:
        f_type = st.session_state.feedback_msg.get("type", "info")
        f_text = st.session_state.feedback_msg.get("text", "")
        if f_type == "success":
            st.success(f_text)
        else:
            st.info(f_text)

    col_input, col_ctrl = st.columns([1.1, 1], gap="large")

    with col_input:
        st.markdown(
            """
            <div class="custom-card">
                <div class="card-title">🧩 Initial Puzzle Board (3×3)</div>
                <p style="color:#64748B; font-size:13px; margin-top:-8px;">
                    Enter tile numbers 0-8 (0 = Blank) or generate a puzzle from the sidebar.
                </p>
            """,
            unsafe_allow_html=True,
        )

        # 3x3 interactive input grid bound directly to session state
        for r in range(3):
            row_cols = st.columns(3)
            for c in range(3):
                idx = r * 3 + c
                with row_cols[c]:
                    st.number_input(
                        label=f"Pos ({r+1},{c+1})",
                        min_value=0,
                        max_value=8,
                        step=1,
                        key=f"tile_in_{idx}",
                        label_visibility="collapsed",
                        on_change=on_tile_input_change,
                    )

        # Render initial board with highlighted blank tile (always in sync with inputs)
        st.markdown(render_puzzle_html(current_puzzle_values), unsafe_allow_html=True)

        # Real-time Status Badge
        st.markdown("<div style='margin-top:14px; text-align:center;'>", unsafe_allow_html=True)
        if not is_valid:
            st.markdown(
                f'<div class="badge-unsolvable">⚠️ Invalid Input: {err_msg}</div>',
                unsafe_allow_html=True,
            )
        elif state_tuple == GOAL_STATE:
            st.markdown(
                '<div class="badge-info">🎯 Already at Goal State (0 moves needed)</div>',
                unsafe_allow_html=True,
            )
        elif solvable:
            st.markdown(
                f'<div class="badge-solvable">✅ Solvable &bull; Inversion Count: {inversions} (Even Parity)</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="badge-unsolvable">❌ Mathematically Unsolvable &bull; Inversion Count: {inversions} (Odd Parity)</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div></div>", unsafe_allow_html=True)

    with col_ctrl:
        st.markdown(
            """
            <div class="custom-card">
                <div class="card-title">🤖 A* Algorithm Heuristic Selection</div>
                <p style="color:#64748B; font-size:13px; margin-top:-8px;">
                    Select the heuristic function used by the A* search formula: f(n) = g(n) + h(n).
                </p>
            """,
            unsafe_allow_html=True,
        )

        chosen_heuristic = st.radio(
            "Select Heuristic Function:",
            ["Manhattan Distance", "Misplaced Tiles"],
            index=0 if st.session_state.selected_heuristic == "Manhattan Distance" else 1,
            key="radio_heuristic",
        )
        st.session_state.selected_heuristic = chosen_heuristic

        if chosen_heuristic == "Manhattan Distance":
            st.info(
                "📐 **Manhattan Distance ($h_1$):** Total horizontal & vertical distances each tile must move to reach its goal position. (Blank space excluded)."
            )
        else:
            st.info(
                "📍 **Misplaced Tiles ($h_2$):** Total count of numbered tiles currently not in their goal position. (Blank space excluded)."
            )

        # Initial heuristic estimates
        if is_valid and state_tuple is not None:
            h_val = (
                manhattan_distance(state_tuple)
                if chosen_heuristic == "Manhattan Distance"
                else misplaced_tiles(state_tuple)
            )
            st.markdown(
                f"""
                <div style="background:#F1F5F9; border-radius:10px; padding:12px; margin: 12px 0; border:1px solid #E2E8F0;">
                    <span style="color:#475569; font-size:13px;">Initial Heuristic Value for Initial State:</span><br>
                    <strong style="color:#1E3A8A; font-size:20px;">h(S₀) = {h_val}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Solve Button
        solve_clicked = st.button(
            "🚀 Solve Puzzle",
            type="primary",
            use_container_width=True,
            disabled=(not is_valid or not solvable),
        )

        if not solvable and is_valid:
            st.error("Cannot solve: An odd inversion count makes reaching the goal state mathematically impossible.")

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Execute A* Search
    # -------------------------------------------------------------------------
    if solve_clicked and is_valid and solvable and state_tuple is not None:
        with st.spinner("Calculating optimal solution using A* Search..."):
            h_func = (
                manhattan_distance
                if chosen_heuristic == "Manhattan Distance"
                else misplaced_tiles
            )
            res = solve_puzzle(state_tuple, h_func, chosen_heuristic)
            st.session_state.solution_result = res
            st.session_state.current_step = 0
            st.session_state.feedback_msg = None

    # -------------------------------------------------------------------------
    # Solution Display & Step-by-Step Interactive Navigator
    # -------------------------------------------------------------------------
    if st.session_state.solution_result is not None:
        res: SearchResult = st.session_state.solution_result
        total_steps = len(res.path) - 1

        st.markdown("---")
        st.markdown("### 📊 Solution Performance Metrics")

        # Metrics Banner
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-value">{res.moves_count}</div>
                    <div class="metric-label">Total Moves</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_c2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-value">{round(res.execution_time * 1000, 2)} ms</div>
                    <div class="metric-label">Execution Time</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_c3:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-value">{res.nodes_explored:,}</div>
                    <div class="metric-label">Nodes Explored</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_c4:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-value">{res.nodes_generated:,}</div>
                    <div class="metric-label">Nodes Generated</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # Step-by-Step Solution Navigator (Step 0 to Step N)
        # ---------------------------------------------------------------------
        st.markdown("### 🎬 Step-by-Step Puzzle Solution")
        st.markdown(
            f"**Total Moves Required:** `{res.moves_count}` | **Current Step:** `{st.session_state.current_step}` of `{total_steps}`"
        )

        # Navigation Controls Row: Previous, Next, Auto Play
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5 = st.columns([1, 1, 1, 1, 1.2])

        with ctrl_col1:
            if st.button("⏮ Step 0 (Start)", use_container_width=True, disabled=(st.session_state.current_step == 0)):
                st.session_state.current_step = 0
                st.rerun()

        with ctrl_col2:
            if st.button("◀ Previous Step", use_container_width=True, disabled=(st.session_state.current_step == 0)):
                st.session_state.current_step = max(0, st.session_state.current_step - 1)
                st.rerun()

        with ctrl_col3:
            if st.button("Next Step ▶", use_container_width=True, disabled=(st.session_state.current_step >= total_steps)):
                st.session_state.current_step = min(total_steps, st.session_state.current_step + 1)
                st.rerun()

        with ctrl_col4:
            if st.button("⏭ Goal State", use_container_width=True, disabled=(st.session_state.current_step == total_steps)):
                st.session_state.current_step = total_steps
                st.rerun()

        with ctrl_col5:
            auto_play_clicked = st.button("▶️ Auto Play", type="secondary", use_container_width=True)

        # Auto Play Speed Setting
        play_speed_str = st.select_slider(
            "Auto Play Speed:",
            options=["Fast (0.3s)", "Normal (0.7s)", "Slow (1.2s)"],
            value="Normal (0.7s)",
        )
        speed_map = {"Fast (0.3s)": 0.3, "Normal (0.7s)": 0.7, "Slow (1.2s)": 1.2}
        play_speed = speed_map[play_speed_str]

        # Step Slider for direct access
        step_val = st.slider(
            "Navigate steps manually:",
            min_value=0,
            max_value=total_steps,
            value=st.session_state.current_step,
            step=1,
            format="Step %d",
            key="solver_step_slider",
        )
        if step_val != st.session_state.current_step:
            st.session_state.current_step = step_val
            st.rerun()

        # Auto Play Animation Execution Loop
        if auto_play_clicked and total_steps > 0:
            anim_placeholder = st.empty()
            for step_idx in range(0, total_steps + 1):
                st.session_state.current_step = step_idx
                cur_state = res.path[step_idx]
                if step_idx == 0:
                    title_text = "Step 0: Initial Puzzle State"
                    detail_text = "Starting configuration before any moves."
                    moved_t = None
                else:
                    moved_t = res.tiles_moved[step_idx - 1]
                    m_dir = res.directions[step_idx - 1]
                    if step_idx == total_steps:
                        title_text = f"🎯 Step {step_idx} of {total_steps}: Goal State Reached!"
                    else:
                        title_text = f"Step {step_idx} of {total_steps}: Moved Tile {moved_t} {m_dir}"
                    detail_text = f"Tile {moved_t} was slid {m_dir.upper()} into the blank space."

                with anim_placeholder.container():
                    st.markdown(
                        f"""
                        <div class="step-header-box">
                            <div class="step-title">{title_text}</div>
                            <div class="step-detail">{detail_text}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(render_puzzle_html(cur_state, moved_tile=moved_t), unsafe_allow_html=True)
                time.sleep(play_speed)
            st.rerun()

        # Regular Static Step Display
        cur_step = st.session_state.current_step
        current_state = res.path[cur_step]

        if cur_step == 0:
            step_title = "Step 0: Initial Puzzle State"
            step_detail = "Initial puzzle configuration before any moves."
            moved_tile_num = None
        else:
            moved_tile_num = res.tiles_moved[cur_step - 1]
            move_dir = res.directions[cur_step - 1]
            if cur_step == total_steps:
                step_title = f"🎯 Step {cur_step} of {total_steps}: Goal State Reached!"
            else:
                step_title = f"Step {cur_step} of {total_steps}: Moved Tile {moved_tile_num} {move_dir}"
            step_detail = f"Tile {moved_tile_num} slid {move_dir.upper()} into the blank space."

        st.markdown(
            f"""
            <div class="step-header-box">
                <div class="step-title">{step_title}</div>
                <div class="step-detail">{step_detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Center Active Board
        st.markdown(render_puzzle_html(current_state, moved_tile=moved_tile_num), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # Complete Sequence: Display EVERY Intermediate State in Correct Order
        # ---------------------------------------------------------------------
        st.markdown("### 📋 Complete Sequence of All Intermediate States")
        st.markdown(
            "Every state from **Step 0 (Initial State)** through **all intermediate moves** to the **Final Goal State** in exact sequence:"
        )

        cols_per_row = 4
        for row_start in range(0, len(res.path), cols_per_row):
            batch_cols = st.columns(cols_per_row)
            for offset in range(cols_per_row):
                s_idx = row_start + offset
                if s_idx < len(res.path):
                    with batch_cols[offset]:
                        state_at_s = res.path[s_idx]
                        if s_idx == 0:
                            s_hdr = "Step 0 (Initial)"
                            s_act = "Starting State"
                            m_t = None
                        elif s_idx == total_steps:
                            m_t = res.tiles_moved[s_idx - 1]
                            m_d = res.directions[s_idx - 1]
                            s_hdr = f"Step {s_idx} (Goal State)"
                            s_act = f"Tile {m_t} moved {m_d}"
                        else:
                            m_t = res.tiles_moved[s_idx - 1]
                            m_d = res.directions[s_idx - 1]
                            s_hdr = f"Step {s_idx}"
                            s_act = f"Tile {m_t} moved {m_d}"

                        st.markdown(
                            f"""
                            <div class="seq-card">
                                <div class="seq-header">{s_hdr}</div>
                                <div class="seq-action">{s_act}</div>
                                {render_compact_board_html(state_at_s, moved_tile=m_t)}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        # Download Solution Report Button
        report_lines = [
            "=" * 60,
            "AI-BASED 8-PUZZLE GAME SOLVER - SOLUTION REPORT",
            "=" * 60,
            f"Heuristic Used: {res.heuristic_name}",
            f"Initial State : {res.path[0]}",
            f"Goal State    : {GOAL_STATE}",
            f"Total Moves   : {res.moves_count}",
            f"Nodes Explored: {res.nodes_explored:,}",
            f"Nodes Generated: {res.nodes_generated:,}",
            f"Execution Time: {round(res.execution_time * 1000, 2)} ms",
            "-" * 60,
            "COMPLETE STEP-BY-STEP SOLUTION PATH:",
            "-" * 60,
        ]
        for step_i in range(len(res.path)):
            st_b = res.path[step_i]
            grid_str = f"[{st_b[0]} {st_b[1]} {st_b[2]}] / [{st_b[3]} {st_b[4]} {st_b[5]}] / [{st_b[6]} {st_b[7]} {st_b[8]}]"
            if step_i == 0:
                report_lines.append(f"Step 0: Initial State -> {grid_str}")
            elif step_i == len(res.path) - 1:
                t_m = res.tiles_moved[step_i - 1]
                d_m = res.directions[step_i - 1]
                report_lines.append(f"Step {step_i}: Slid Tile {t_m} {d_m} -> GOAL REACHED -> {grid_str}")
            else:
                t_m = res.tiles_moved[step_i - 1]
                d_m = res.directions[step_i - 1]
                report_lines.append(f"Step {step_i}: Slid Tile {t_m} {d_m} -> {grid_str}")

        report_txt = "\n".join(report_lines)
        st.download_button(
            label="📥 Download Complete Solution Report (.txt)",
            data=report_txt,
            file_name=f"8_puzzle_solution_{res.heuristic_name.lower().replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=False,
        )


# =============================================================================
# TAB 2: ALGORITHM COMPARISON
# =============================================================================
with tab_comparison:
    st.markdown("### ⚖️ Algorithm Comparison: Manhattan Distance vs Misplaced Tiles")
    st.markdown(
        "Benchmarking both heuristics on the **exact same initial board state** using the unified A* search algorithm."
    )

    comp_btn_clicked = st.button(
        "⚡ Run Comparison on Current Initial State",
        type="primary",
        disabled=(not is_valid or not solvable),
    )

    # Solve both heuristics on click or if already cached
    if comp_btn_clicked and is_valid and solvable and state_tuple is not None:
        with st.spinner("Executing A* with both Manhattan Distance and Misplaced Tiles..."):
            res_m = solve_puzzle(state_tuple, manhattan_distance, "Manhattan Distance")
            res_mis = solve_puzzle(state_tuple, misplaced_tiles, "Misplaced Tiles")
            st.session_state.comp_res_manhattan = res_m
            st.session_state.comp_res_misplaced = res_mis

    # Display comparison if available
    if (
        st.session_state.comp_res_manhattan is not None
        and st.session_state.comp_res_misplaced is not None
    ):
        r_m = st.session_state.comp_res_manhattan
        r_mis = st.session_state.comp_res_misplaced

        # Side-by-side metric boxes
        col_m_card, col_mis_card = st.columns(2, gap="large")

        with col_m_card:
            st.markdown(
                f"""
                <div class="custom-card" style="border-top: 4px solid #2563EB;">
                    <div style="font-size:18px; font-weight:800; color:#1E3A8A; margin-bottom:12px;">
                        📐 Manhattan Distance Heuristic (h₁)
                    </div>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
                        <div class="metric-box">
                            <div class="metric-value">{r_m.moves_count}</div>
                            <div class="metric-label">Optimal Moves</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{round(r_m.execution_time * 1000, 2)} ms</div>
                            <div class="metric-label">Execution Time</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{r_m.nodes_explored:,}</div>
                            <div class="metric-label">Nodes Explored</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{r_m.nodes_generated:,}</div>
                            <div class="metric-label">Nodes Generated</div>
                        </div>
                    </div>
                    <div style="margin-top:14px; font-size:13px; color:#475569;">
                        <strong>Initial h₁ value:</strong> {r_m.initial_h}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_mis_card:
            st.markdown(
                f"""
                <div class="custom-card" style="border-top: 4px solid #0D9488;">
                    <div style="font-size:18px; font-weight:800; color:#0F766E; margin-bottom:12px;">
                        📍 Misplaced Tiles Heuristic (h₂)
                    </div>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
                        <div class="metric-box">
                            <div class="metric-value">{r_mis.moves_count}</div>
                            <div class="metric-label">Optimal Moves</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{round(r_mis.execution_time * 1000, 2)} ms</div>
                            <div class="metric-label">Execution Time</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{r_mis.nodes_explored:,}</div>
                            <div class="metric-label">Nodes Explored</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{r_mis.nodes_generated:,}</div>
                            <div class="metric-label">Nodes Generated</div>
                        </div>
                    </div>
                    <div style="margin-top:14px; font-size:13px; color:#475569;">
                        <strong>Initial h₂ value:</strong> {r_mis.initial_h}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Comparative Efficiency Insights
        if r_mis.nodes_explored > 0:
            diff_nodes = r_mis.nodes_explored - r_m.nodes_explored
            pct_savings = round((diff_nodes / r_mis.nodes_explored) * 100, 1) if r_mis.nodes_explored > 0 else 0
            if pct_savings > 0:
                st.success(
                    f"💡 **AI Search Space Analysis:** Manhattan Distance explored **{diff_nodes:,} fewer nodes ({pct_savings}% reduction)** "
                    f"than Misplaced Tiles while both guaranteed the exact same optimal solution length of **{r_m.moves_count} moves**!"
                )
            else:
                st.info(
                    f"💡 **AI Search Space Analysis:** Both heuristics required the exact same optimal solution length of **{r_m.moves_count} moves**."
                )

        # ---------------------------------------------------------------------
        # Comparison Table
        # ---------------------------------------------------------------------
        st.markdown("#### 📋 Comparison Table")
        comp_data = {
            "Performance Metric": [
                "Number of Moves (Solution Length)",
                "Nodes Explored",
                "Nodes Generated",
                "Execution Time (ms)",
                "Initial Heuristic Value h(S₀)",
                "Optimality Guaranteed?",
            ],
            "Manhattan Distance (h₁)": [
                r_m.moves_count,
                f"{r_m.nodes_explored:,}",
                f"{r_m.nodes_generated:,}",
                f"{round(r_m.execution_time * 1000, 2)} ms",
                r_m.initial_h,
                "Yes (Optimal)",
            ],
            "Misplaced Tiles (h₂)": [
                r_mis.moves_count,
                f"{r_mis.nodes_explored:,}",
                f"{r_mis.nodes_generated:,}",
                f"{round(r_mis.execution_time * 1000, 2)} ms",
                r_mis.initial_h,
                "Yes (Optimal)",
            ],
        }
        st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)

        # ---------------------------------------------------------------------
        # Comparison Charts (Altair)
        # ---------------------------------------------------------------------
        st.markdown("#### 📊 Empirical Comparison Charts")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            df_nodes = pd.DataFrame(
                [
                    {"Heuristic": "Manhattan Distance", "Metric": "Nodes Explored", "Count": r_m.nodes_explored},
                    {"Heuristic": "Manhattan Distance", "Metric": "Nodes Generated", "Count": r_m.nodes_generated},
                    {"Heuristic": "Misplaced Tiles", "Metric": "Nodes Explored", "Count": r_mis.nodes_explored},
                    {"Heuristic": "Misplaced Tiles", "Metric": "Nodes Generated", "Count": r_mis.nodes_generated},
                ]
            )
            chart1 = (
                alt.Chart(df_nodes)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Metric:N", title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Count:Q", title="Number of Nodes"),
                    color=alt.Color(
                        "Heuristic:N",
                        scale=alt.Scale(
                            domain=["Manhattan Distance", "Misplaced Tiles"],
                            range=["#2563EB", "#0D9488"],
                        ),
                    ),
                    xOffset="Heuristic:N",
                    tooltip=["Heuristic", "Metric", "Count"],
                )
                .properties(title="Search Space: Nodes Explored vs Generated", height=320)
            )
            st.altair_chart(chart1, use_container_width=True)

        with chart_col2:
            df_time = pd.DataFrame(
                [
                    {"Heuristic": "Manhattan Distance", "Time (ms)": round(r_m.execution_time * 1000, 2)},
                    {"Heuristic": "Misplaced Tiles", "Time (ms)": round(r_mis.execution_time * 1000, 2)},
                ]
            )
            chart2 = (
                alt.Chart(df_time)
                .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, width=54)
                .encode(
                    x=alt.X("Heuristic:N", title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Time (ms):Q", title="Execution Time (milliseconds)"),
                    color=alt.Color(
                        "Heuristic:N",
                        scale=alt.Scale(
                            domain=["Manhattan Distance", "Misplaced Tiles"],
                            range=["#2563EB", "#0D9488"],
                        ),
                        legend=None,
                    ),
                    tooltip=["Heuristic", "Time (ms)"],
                )
                .properties(title="Execution Time (Milliseconds)", height=320)
            )
            st.altair_chart(chart2, use_container_width=True)

        # ---------------------------------------------------------------------
        # Full Solution Path Comparison
        # ---------------------------------------------------------------------
        st.markdown("#### 🔍 Solution Path Comparison")
        st.markdown(
            "Side-by-side walkthrough of the exact sequence of moves found by both heuristics:"
        )

        p_tab1, p_tab2 = st.tabs(["📐 Manhattan Distance Solution Path", "📍 Misplaced Tiles Solution Path"])

        with p_tab1:
            st.markdown(f"**Total Moves:** `{r_m.moves_count}` | **Nodes Explored:** `{r_m.nodes_explored:,}`")
            for idx in range(len(r_m.path)):
                st_state = r_m.path[idx]
                if idx == 0:
                    st.markdown(f"**Step 0 (Initial State):**")
                    st.markdown(render_compact_board_html(st_state, cell_size=38), unsafe_allow_html=True)
                else:
                    t_moved = r_m.tiles_moved[idx - 1]
                    d_moved = r_m.directions[idx - 1]
                    st.markdown(f"**Step {idx}:** Slid Tile **{t_moved}** {d_moved}")
                    st.markdown(render_compact_board_html(st_state, moved_tile=t_moved, cell_size=38), unsafe_allow_html=True)
                st.markdown("---")

        with p_tab2:
            st.markdown(f"**Total Moves:** `{r_mis.moves_count}` | **Nodes Explored:** `{r_mis.nodes_explored:,}`")
            for idx in range(len(r_mis.path)):
                st_state = r_mis.path[idx]
                if idx == 0:
                    st.markdown(f"**Step 0 (Initial State):**")
                    st.markdown(render_compact_board_html(st_state, cell_size=38), unsafe_allow_html=True)
                else:
                    t_moved = r_mis.tiles_moved[idx - 1]
                    d_moved = r_mis.directions[idx - 1]
                    st.markdown(f"**Step {idx}:** Slid Tile **{t_moved}** {d_moved}")
                    st.markdown(render_compact_board_html(st_state, moved_tile=t_moved, cell_size=38), unsafe_allow_html=True)
                st.markdown("---")
    else:
        st.info("Click the button above to run both heuristics on the current board configuration and view the comparison.")
