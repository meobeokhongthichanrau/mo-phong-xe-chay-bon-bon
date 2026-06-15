import heapq
import itertools
import time

import graphviz
import streamlit as st
import streamlit.components.v1 as components

# --- 1. CẤU HÌNH HỆ THỐNG MÔ PHỎNG ---
st.set_page_config(page_title="AGV Thuyết Trình Đồ Án", layout="wide")


def rerun_page():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# --- 2. MA TRẬN BÃI ĐỖ XE ---
GRID_SIZE = 6
GRID = [
    [0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0],
]

START_POS = (0, 0, 2)  # Ô (0,0), Hướng Nam
GOAL_POS = (5, 5)      # Ô (5,5)

COST_FORWARD = 1
COST_TURN = 2
COST_BACKWARD = 3

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_NAMES = ["Bắc 🔼", "Đông ▶️", "Nam 🔽", "Tây ◀️"]
DIR_ARROWS = ["🔼", "▶️", "🔽", "◀️"]
DIR_ROTATION = {0: 0, 1: 90, 2: 180, 3: 270}

# Tree item: (r, c, d, cost, node_id, parent_id, action)


# --- 3. HÀM PHỤ TRỢ ---
def make_node_id(r, c, d, g, extra=""):
    """Tạo ID riêng cho từng node để Graphviz không bị đè node khi đi lại cùng một ô."""
    suffix = f"-{extra}" if extra != "" else ""
    return f"node-{r}-{c}-{d}-{g}{suffix}"


def is_inside(r, c):
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE


def is_free_cell(r, c):
    return is_inside(r, c) and GRID[r][c] == 0


def heuristic(r, c, g_r, g_c):
    """Heuristic Manhattan cho A*."""
    return (abs(r - g_r) + abs(c - g_c)) * COST_FORWARD


def reset_system():
    start_id = make_node_id(START_POS[0], START_POS[1], START_POS[2], 0, "start")
    st.session_state.car_pos = START_POS
    st.session_state.current_cost = 0
    st.session_state.explored_cells = {(START_POS[0], START_POS[1])}
    st.session_state.tree_history = [(START_POS[0], START_POS[1], START_POS[2], 0, start_id, None, "START")]
    st.session_state.is_auto_running = False
    st.session_state.auto_steps = []
    st.session_state.auto_index = 0
    st.session_state.auto_plan_cost = None
    st.session_state.status_message = ""


def add_manual_step(r, c, d, new_cost, action):
    parent_id = st.session_state.tree_history[-1][4]
    new_id = make_node_id(r, c, d, new_cost, len(st.session_state.tree_history))
    st.session_state.tree_history.append((r, c, d, new_cost, new_id, parent_id, action))
    st.session_state.car_pos = (r, c, d)
    st.session_state.current_cost = new_cost
    st.session_state.explored_cells.add((r, c))
    st.session_state.status_message = ""
    rerun_page()


# --- 4. KHỞI TẠO SESSION STATE ---
if "tree_history" not in st.session_state:
    reset_system()

if "auto_delay" not in st.session_state:
    st.session_state.auto_delay = 0.25


# --- 5. THUẬT TOÁN A* - ĐÃ SỬA NÚT AUTO ---
def run_astar_search_from_current():
    """
    Tìm đường tối ưu từ vị trí hiện tại đến GOAL_POS.
    Fix chính: điều kiện đích phải là (r, c) == (r_goal, c_goal), không phải (r, c) == c_goal.
    """
    curr_r, curr_c, curr_d = st.session_state.car_pos
    curr_cost = st.session_state.current_cost
    r_goal, c_goal = GOAL_POS

    root_id = st.session_state.tree_history[-1][4]
    counter = itertools.count()

    # Hàng đợi ưu tiên: (f, g, thứ_tự, r, c, d, node_id, history)
    pq = [(
        curr_cost + heuristic(curr_r, curr_c, r_goal, c_goal),
        curr_cost,
        next(counter),
        curr_r,
        curr_c,
        curr_d,
        root_id,
        [],
    )]

    best_cost = {}

    while pq:
        f, g, _, r, c, d, current_id, history = heapq.heappop(pq)

        if best_cost.get((r, c, d), float("inf")) <= g:
            continue
        best_cost[(r, c, d)] = g

        # BUG CŨ Ở ĐÂY: code cũ ghi (r, c) == c_goal nên auto không bao giờ tới đích.
        if (r, c) == (r_goal, c_goal):
            return history

        dr, dc = DIRS[d]

        actions = [
            ("Tiến", COST_FORWARD, r + dr, c + dc, d),
            ("Lùi", COST_BACKWARD, r - dr, c - dc, d),
            ("Xoay trái", COST_TURN, r, c, (d - 1) % 4),
            ("Xoay phải", COST_TURN, r, c, (d + 1) % 4),
        ]

        for action_name, action_cost, nr, nc, nd in actions:
            # Xoay hướng thì xe vẫn đứng tại ô hiện tại, không cần kiểm tra vật cản mới.
            if action_name.startswith("Xoay") or is_free_cell(nr, nc):
                next_g = g + action_cost

                if best_cost.get((nr, nc, nd), float("inf")) <= next_g:
                    continue

                step_no = next(counter)
                next_id = make_node_id(nr, nc, nd, next_g, f"auto-{step_no}")
                next_step = (nr, nc, nd, next_g, next_id, current_id, action_name)
                next_history = history + [next_step]
                next_f = next_g + heuristic(nr, nc, r_goal, c_goal)

                heapq.heappush(pq, (next_f, next_g, step_no, nr, nc, nd, next_id, next_history))

    return []


# --- 6. DỰNG SƠ ĐỒ CÂY NHỎ + TỎA NGANG ---
def draw_compact_tree(history_list):
    dot = graphviz.Digraph(engine="dot")

    # LR = mọc từ trái sang phải, không cắm đầu xuống dưới.
    # ranksep/nodesep nhỏ giúp cây gọn hơn.
    dot.attr(
        rankdir="LR",
        splines="polyline",
        bgcolor="transparent",
        margin="0.03",
        pad="0.02",
        ranksep="0.28",
        nodesep="0.14",
        ratio="compress",
        concentrate="false",
    )

    dot.attr(
        "node",
        shape="box",
        style="filled,rounded",
        fillcolor="#1e252b",
        fontcolor="#ffffff",
        color="#34414c",
        fontname="Arial",
        fontsize="8",
        penwidth="1",
        margin="0.04,0.03",
        width="0.42",
        height="0.22",
    )

    dot.attr(
        "edge",
        color="#cfd8dc",
        fontcolor="#cfd8dc",
        fontname="Arial",
        fontsize="7",
        arrowsize="0.35",
        penwidth="0.8",
    )

    last_index = len(history_list) - 1

    for i, item in enumerate(history_list):
        # Tương thích nếu còn dữ liệu kiểu cũ 6 phần tử.
        if len(item) == 6:
            r, c, d, g, node_id, parent_id = item
            action = ""
        else:
            r, c, d, g, node_id, parent_id, action = item

        label_text = f"({r},{c})\n{DIR_ARROWS[d]} g={g}"

        node_kwargs = {}
        if i == 0:
            node_kwargs = {"fillcolor": "#2ed573", "color": "#26af5f", "fontcolor": "#ffffff"}
        elif (r, c) == GOAL_POS:
            node_kwargs = {"fillcolor": "#ff4757", "color": "#ff2e44", "fontcolor": "#ffffff"}
        elif i == last_index:
            node_kwargs = {"fillcolor": "#1e90ff", "color": "#0984e3", "fontcolor": "#ffffff"}

        dot.node(node_id, label=label_text, **node_kwargs)

        if parent_id:
            short_action = {
                "Tiến": "+1",
                "Lùi": "+3",
                "Xoay trái": "L+2",
                "Xoay phải": "R+2",
            }.get(action, "")
            dot.edge(parent_id, node_id, label=short_action)

    return dot.source


def render_tree_box(tree_source, height=470):
    """Render SVG trong khung có cuộn ngang/dọc, nhìn gọn hơn st.graphviz_chart mặc định."""
    try:
        svg = graphviz.Source(tree_source).pipe(format="svg").decode("utf-8")
        html = f"""
        <style>
            body {{ margin: 0; background: transparent; }}
            .tree-box {{
                height: {height}px;
                overflow: auto;
                background: #0e1117;
                border: 1px solid #263238;
                border-radius: 14px;
                padding: 10px;
                box-sizing: border-box;
            }}
            .tree-box svg {{
                max-width: none !important;
                transform: scale(0.88);
                transform-origin: top left;
            }}
        </style>
        <div class="tree-box">{svg}</div>
        """
        components.html(html, height=height + 25, scrolling=False)
    except Exception:
        st.graphviz_chart(tree_source, use_container_width=True)


# --- 7. RENDER KHUNG BẢN ĐỒ HTML ---
def render_grid_dynamic(explored, car_pos, is_done):
    car_r, car_c, car_d = car_pos
    html = """
    <style>
        body { background-color: transparent; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; overflow: hidden; }
        .grid-container { display: flex; justify-content: center; align-items: center; background-color: #12181b; padding: 10px; border-radius: 12px; border: 2.5px solid #ffffff; box-shadow: 0 0 18px rgba(255, 255, 255, 0.2); width: 330px; max-width: 100%; margin: 0 auto; box-sizing: border-box; }
        .grid-table { border-collapse: separate; border-spacing: 4px; table-layout: fixed; width: 300px; height: 300px; }
        .grid-cell { width: 44px !important; height: 44px !important; box-sizing: border-box; text-align: center; vertical-align: middle; font-family: 'Segoe UI', sans-serif; font-size: 13px; font-weight: bold; border-radius: 6px; position: relative; background-color: #1e252b; border: 1px dashed #34414c; color: #ffffff; }
        .cell-obstacle { background-color: #3d4d59 !important; border: 1px solid #4f616f !important; }
        .cell-start { background-color: #2ed573 !important; color: #ffffff !important; }
        .cell-goal { background-color: #ff4757 !important; color: #ffffff !important; }
        .cell-explored { background-color: #103136 !important; border: 1px dashed #00cec9 !important; }
        .cell-path { background-color: #2ed573 !important; border: 1px solid #55efc4 !important; box-shadow: inset 0 0 14px rgba(85, 239, 196, 0.85) !important; }
        .path-dot { width: 12px; height: 12px; background-color: #ffffff; border-radius: 50%; margin: auto; box-shadow: 0 0 12px #ffffff; }
        .car-wrapper { width: 100%; height: 100%; display: flex; justify-content: center; align-items: center; position: absolute; top: 0; left: 0; z-index: 10; }
    </style>
    <div class='grid-container'><table class='grid-table'>
    """

    for r in range(GRID_SIZE):
        html += "<tr>"
        for c in range(GRID_SIZE):
            cell_class = "grid-cell"
            content = ""

            if GRID[r][c] == 1:
                cell_class += " cell-obstacle"
            elif (r, c) == (START_POS[0], START_POS[1]):
                cell_class += " cell-start"
                content = "S"
            elif (r, c) == GOAL_POS:
                cell_class += " cell-goal"
                content = "G"
            elif (r, c) in explored:
                cell_class += " cell-path" if is_done else " cell-explored"
                if is_done:
                    content = "<div class='path-dot'></div>"

            if r == car_r and c == car_c:
                angle = DIR_ROTATION[car_d]
                content = f"""
                <div class='car-wrapper'>
                    <svg width="34" height="34" viewBox="0 0 24 24" style="transform: rotate({angle}deg); overflow: visible;">
                        <rect x="5" y="3" width="14" height="18" rx="4" fill="#1e90ff" stroke="#ffffff" stroke-width="1.5"/>
                        <path d="M7 8 C 7 6, 17 6, 17 8 L16 11 L8 11 Z" fill="#ffffff" opacity="0.9"/>
                        <circle cx="8" cy="3.4" r="1.8" fill="#fffa65"/>
                        <circle cx="16" cy="3.4" r="1.8" fill="#fffa65"/>
                    </svg>
                </div>
                """

            html += f"<td class='{cell_class}'>{content}</td>"
        html += "</tr>"

    html += "</table></div>"
    return html


# --- 8. LUỒNG AUTO REAL-TIME ---
if st.session_state.is_auto_running:
    if st.session_state.auto_index < len(st.session_state.auto_steps):
        time.sleep(st.session_state.auto_delay)
        next_step = st.session_state.auto_steps[st.session_state.auto_index]

        st.session_state.car_pos = (next_step[0], next_step[1], next_step[2])
        st.session_state.current_cost = next_step[3]
        st.session_state.explored_cells.add((next_step[0], next_step[1]))
        st.session_state.tree_history.append(next_step)

        st.session_state.auto_index += 1
        rerun_page()
    else:
        st.session_state.is_auto_running = False
        st.session_state.status_message = "✅ Auto đã chạy xong đường đi tối ưu đến G."
        rerun_page()


# --- 9. GIAO DIỆN HIỂN THỊ CHÍNH ---
st.title("🖥️ HỆ THỐNG MÔ PHỎNG VÀ ĐỊNH TUYẾN XE AGV")
st.markdown("---")

col_left, col_right = st.columns([4, 6], gap="medium")
is_finished = (st.session_state.car_pos[0], st.session_state.car_pos[1]) == GOAL_POS

with col_left:
    st.markdown("### 🗺️ Sa Bàn Lưới")
    components.html(
        render_grid_dynamic(st.session_state.explored_cells, st.session_state.car_pos, is_finished),
        height=330,
        scrolling=False,
    )

    st.info(
        f"📍 Ô: `({st.session_state.car_pos[0]}, {st.session_state.car_pos[1]})` "
        f"| 🧭 Hướng: **{DIR_NAMES[st.session_state.car_pos[2]]}** "
        f"| 💰 Chi phí: `{st.session_state.current_cost}`"
    )

    if st.session_state.status_message:
        st.success(st.session_state.status_message)

    with st.expander("⚙️ Cài đặt mô phỏng", expanded=False):
        st.session_state.auto_delay = st.slider(
            "Tốc độ auto: số càng nhỏ chạy càng nhanh",
            min_value=0.05,
            max_value=0.80,
            value=float(st.session_state.auto_delay),
            step=0.05,
        )

    st.markdown("##### 🕹️ Lái xe thủ công:")
    current_r, current_c, current_d = st.session_state.car_pos

    btn_col1, btn_col2 = st.columns(2)
    if btn_col1.button("🔼 TIẾN (+1)", use_container_width=True, disabled=st.session_state.is_auto_running or is_finished):
        dr, dc = DIRS[current_d]
        nr, nc = current_r + dr, current_c + dc
        if is_free_cell(nr, nc):
            add_manual_step(nr, nc, current_d, st.session_state.current_cost + COST_FORWARD, "Tiến")
        else:
            st.warning("🚧 Không thể tiến: phía trước là tường hoặc ngoài bản đồ.")

    if btn_col2.button("🔽 LÙI (+3)", use_container_width=True, disabled=st.session_state.is_auto_running or is_finished):
        dr, dc = DIRS[current_d]
        nr, nc = current_r - dr, current_c - dc
        if is_free_cell(nr, nc):
            add_manual_step(nr, nc, current_d, st.session_state.current_cost + COST_BACKWARD, "Lùi")
        else:
            st.warning("🚧 Không thể lùi: phía sau là tường hoặc ngoài bản đồ.")

    btn_col3, btn_col4 = st.columns(2)
    if btn_col3.button("↩️ XOAY TRÁI (+2)", use_container_width=True, disabled=st.session_state.is_auto_running or is_finished):
        new_d = (current_d - 1) % 4
        add_manual_step(current_r, current_c, new_d, st.session_state.current_cost + COST_TURN, "Xoay trái")

    if btn_col4.button("↪️ XOAY PHẢI (+2)", use_container_width=True, disabled=st.session_state.is_auto_running or is_finished):
        new_d = (current_d + 1) % 4
        add_manual_step(current_r, current_c, new_d, st.session_state.current_cost + COST_TURN, "Xoay phải")

    st.markdown("---")
    ctrl1, ctrl2 = st.columns(2)

    if ctrl1.button("🤖 KÍCH HOẠT AUTO", use_container_width=True, type="primary", disabled=st.session_state.is_auto_running or is_finished):
        steps = run_astar_search_from_current()
        if steps:
            st.session_state.auto_steps = steps
            st.session_state.auto_index = 0
            st.session_state.auto_plan_cost = steps[-1][3]
            st.session_state.is_auto_running = True
            st.session_state.status_message = f"🤖 Auto đã tìm thấy {len(steps)} bước. Dự kiến tổng chi phí: {steps[-1][3]}."
            rerun_page()
        else:
            st.session_state.status_message = "⚠️ Không tìm thấy đường đi hợp lệ đến đích G."
            rerun_page()

    if ctrl2.button("🔄 RESET HỆ THỐNG", use_container_width=True):
        reset_system()
        rerun_page()

with col_right:
    st.markdown("### 🌳 Sơ Đồ Cây Trạng Thái")
    st.caption("Cây đã đổi sang hướng trái → phải, node nhỏ hơn, có khung cuộn để không bị dài cắm xuống dưới.")

    tree_src = draw_compact_tree(st.session_state.tree_history)
    render_tree_box(tree_src, height=500)
