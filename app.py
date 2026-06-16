import heapq
import itertools
import time
import textwrap

import graphviz
import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# 1. CẤU HÌNH TRANG
# ============================================================
st.set_page_config(page_title="AGV - A* Auto Scan", layout="wide")


def rerun_page():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# CSS tổng: đổi sang nền sáng, chữ đậm, nhìn rõ khi chiếu slide/máy chiếu.
st.markdown(
    """
    <style>
        .stApp {
            background: #F8FAFC !important;
            color: #111827 !important;
        }
        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: #111827;
        }
        [data-testid="stHeader"] {
            background: rgba(248, 250, 252, 0.92) !important;
        }
        [data-testid="stToolbar"] {
            right: 1rem;
        }
        .stButton > button {
            font-weight: 800 !important;
            border-radius: 12px !important;
            min-height: 42px !important;
            border: 2px solid #CBD5E1 !important;
            background: #FFFFFF !important;
            color: #0F172A !important;
        }
        .stButton > button:hover {
            border-color: #2563EB !important;
            color: #1D4ED8 !important;
        }
        .stButton > button[kind="primary"] {
            background: #2563EB !important;
            color: white !important;
            border-color: #1D4ED8 !important;
        }
        div[data-testid="stAlert"] {
            border-radius: 14px !important;
            font-weight: 750 !important;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. MA TRẬN BÃI ĐỖ XE
# 0 = đường đi, 1 = vật cản
# ============================================================
GRID_SIZE = 6
GRID = [
    [0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0],
]

START_POS = (0, 0, 2)  # hàng, cột, hướng. 2 = Nam
GOAL_POS = (5, 5)

COST_FORWARD = 1
COST_TURN = 2
COST_BACKWARD = 3

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_NAMES = ["Bắc 🔼", "Đông ▶️", "Nam 🔽", "Tây ◀️"]
DIR_ARROWS = ["🔼", "▶️", "🔽", "◀️"]
DIR_ROTATION = {0: 0, 1: 90, 2: 180, 3: 270}

ACTION_SHORT = {
    "START": "S",
    "Tiến": "+1",
    "Lùi": "+3",
    "Xoay trái": "L+2",
    "Xoay phải": "R+2",
}

ACTION_ICON = {
    "START": "START",
    "Tiến": "TIẾN",
    "Lùi": "LÙI",
    "Xoay trái": "TRÁI",
    "Xoay phải": "PHẢI",
}


# ============================================================
# 3. HÀM PHỤ TRỢ
# ============================================================
def is_inside(r, c):
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE


def is_free_cell(r, c):
    return is_inside(r, c) and GRID[r][c] == 0


def heuristic(r, c, goal_r, goal_c):
    return (abs(r - goal_r) + abs(c - goal_c)) * COST_FORWARD


def make_node(node_id, r, c, d, g, parent_id=None, action="START"):
    return {
        "id": node_id,
        "r": r,
        "c": c,
        "d": d,
        "g": g,
        "parent": parent_id,
        "action": action,
    }


def make_edge(src, dst, action):
    return {"src": src, "dst": dst, "label": ACTION_SHORT.get(action, "")}


def reset_system():
    start_id = "root-start"
    st.session_state.car_pos = START_POS
    st.session_state.current_cost = 0
    st.session_state.explored_cells = {(START_POS[0], START_POS[1])}
    st.session_state.scanned_cells = set()
    st.session_state.optimal_cells = set()
    st.session_state.tree_nodes = [make_node(start_id, START_POS[0], START_POS[1], START_POS[2], 0, None, "START")]
    st.session_state.tree_edges = []
    st.session_state.current_tree_id = start_id
    st.session_state.scanned_node_ids = set()
    st.session_state.path_node_ids = set()
    st.session_state.status_message = "Sẵn sàng. Bấm Auto để A* quét bản đồ, rồi xe mới chạy từng bước."
    st.session_state.last_action = "START"


def ensure_defaults():
    if "tree_nodes" not in st.session_state:
        reset_system()
    if "scan_delay" not in st.session_state:
        st.session_state.scan_delay = 0.18
    if "move_delay" not in st.session_state:
        st.session_state.move_delay = 0.35
    if "tree_zoom" not in st.session_state:
        st.session_state.tree_zoom = 0.65  # Mặc định nhỏ gọn hơn vừa mắt
    if "tree_height" not in st.session_state:
        st.session_state.tree_height = 480
    if "show_scan_tree" not in st.session_state:
        st.session_state.show_scan_tree = True
    if "anti_flash_mode" not in st.session_state:
        st.session_state.anti_flash_mode = False


ensure_defaults()


# ============================================================
# 4. THUẬT TOÁN A*
# ============================================================
def get_neighbors(r, c, d):
    dr, dc = DIRS[d]
    candidates = [
        ("Tiến", COST_FORWARD, r + dr, c + dc, d),
        ("Lùi", COST_BACKWARD, r - dr, c - dc, d),
        ("Xoay trái", COST_TURN, r, c, (d - 1) % 4),
        ("Xoay phải", COST_TURN, r, c, (d + 1) % 4),
    ]
    for action, cost, nr, nc, nd in candidates:
        if action.startswith("Xoay") or is_free_cell(nr, nc):
            yield action, cost, nr, nc, nd


def build_astar_plan_from_current():
    start_r, start_c, start_d = st.session_state.car_pos
    start_g = st.session_state.current_cost
    goal_r, goal_c = GOAL_POS
    root_id = st.session_state.current_tree_id
    start_state = (start_r, start_c, start_d)

    counter = itertools.count(1)
    pq = []
    heapq.heappush(pq, (start_g + heuristic(start_r, start_c, goal_r, goal_c), start_g, 0, start_state))

    discovered_cost = {start_state: start_g}
    record = {start_state: {"node_id": root_id, "parent_state": None, "action": "START", "g": start_g}}

    closed = set()
    new_nodes = []
    new_edges = []
    scan_frames = []
    scanned_node_ids = set()
    scanned_cells = set()
    goal_state = None

    while pq:
        _, g, _, state = heapq.heappop(pq)
        r, c, d = state

        if state in closed:
            continue
        if g != discovered_cost.get(state, float("inf")):
            continue

        closed.add(state)
        current_node_id = record[state]["node_id"]
        scanned_node_ids.add(current_node_id)
        scanned_cells.add((r, c))

        if (r, c) == (goal_r, goal_c):
            goal_state = state
            scan_frames.append({
                "visible_count": len(new_nodes),
                "current_node_id": current_node_id,
                "scanned_node_ids": set(scanned_node_ids),
                "scanned_cells": set(scanned_cells),
                "message": f"Đã tìm thấy G tại ({r},{c}) với chi phí tối ưu g={g}.",
            })
            break

        for action, action_cost, nr, nc, nd in get_neighbors(r, c, d):
            next_state = (nr, nc, nd)
            next_g = g + action_cost

            if next_g >= discovered_cost.get(next_state, float("inf")):
                continue

            discovered_cost[next_state] = next_g
            step_no = next(counter)
            node_id = f"auto-{step_no}-{nr}-{nc}-{nd}-{next_g}"

            record[next_state] = {
                "node_id": node_id,
                "parent_state": state,
                "action": action,
                "g": next_g,
            }

            new_nodes.append(make_node(node_id, nr, nc, nd, next_g, current_node_id, action))
            new_edges.append(make_edge(current_node_id, node_id, action))

            next_f = next_g + heuristic(nr, nc, goal_r, goal_c)
            heapq.heappush(pq, (next_f, next_g, step_no, next_state))

        scan_frames.append({
            "visible_count": len(new_nodes),
            "current_node_id": current_node_id,
            "scanned_node_ids": set(scanned_node_ids),
            "scanned_cells": set(scanned_cells),
            "message": f"Đang quét node ({r},{c}) {DIR_ARROWS[d]} | g={g}.",
        })

    if goal_state is None:
        return None

    path_states = []
    cur = goal_state
    while cur is not None and cur != start_state:
        path_states.append(cur)
        cur = record[cur]["parent_state"]
    path_states.reverse()

    path_steps = []
    path_node_ids = set()
    path_cells = {(start_r, start_c)}

    for state in path_states:
        r, c, d = state
        node_id = record[state]["node_id"]
        parent_state = record[state]["parent_state"]
        parent_node_id = record[parent_state]["node_id"] if parent_state else root_id
        action = record[state]["action"]
        g = record[state]["g"]
        path_node_ids.add(node_id)
        path_cells.add((r, c))
        path_steps.append(make_node(node_id, r, c, d, g, parent_node_id, action))

    return {
        "new_nodes": new_nodes,
        "new_edges": new_edges,
        "scan_frames": scan_frames,
        "path_steps": path_steps,
        "path_node_ids": path_node_ids,
        "path_cells": path_cells,
        "total_cost": path_steps[-1]["g"] if path_steps else start_g,
        "expanded_count": len(scan_frames),
    }


# ============================================================
# 5. RENDER SA BÀN LƯỚI
# ============================================================
def render_grid_html(explored, scanned, optimal, car_pos, is_done):
    car_r, car_c, car_d = car_pos
    html = """
    <style>
        .agv-grid-wrap {
            width: 390px;
            max-width: 100%;
            margin: 0 auto;
            background: #FFFFFF;
            border: 4px solid #0F172A;
            border-radius: 18px;
            padding: 14px;
            box-shadow: 0 14px 35px rgba(15, 23, 42, 0.18);
            box-sizing: border-box;
        }
        .agv-grid {
            border-collapse: separate;
            border-spacing: 6px;
            width: 100%;
            table-layout: fixed;
        }
        .agv-cell {
            width: 52px;
            height: 52px;
            box-sizing: border-box;
            text-align: center;
            vertical-align: middle;
            font-family: Arial, sans-serif;
            font-size: 18px;
            font-weight: 900;
            border-radius: 11px;
            position: relative;
            background: #E2E8F0;
            border: 2px dashed #94A3B8;
            color: #0F172A;
        }
        .agv-obstacle {
            background: #334155 !important;
            border: 2px solid #0F172A !important;
            color: #FFFFFF !important;
        }
        .agv-start {
            background: #22C55E !important;
            color: #052E16 !important;
            border: 3px solid #15803D !important;
        }
        .agv-goal {
            background: #EF4444 !important;
            color: #FFFFFF !important;
            border: 3px solid #B91C1C !important;
        }
        .agv-scanned {
            background: #93C5FD !important;
            border: 3px solid #2563EB !important;
        }
        .agv-explored {
            background: #86EFAC !important;
            border: 3px solid #16A34A !important;
        }
        .agv-optimal {
            background: #FDE047 !important;
            border: 3px solid #CA8A04 !important;
            box-shadow: inset 0 0 0 3px rgba(255,255,255,0.65), 0 0 12px rgba(202,138,4,0.45);
        }
        .agv-dot {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: #0F172A;
            margin: auto;
        }
        .agv-car {
            width: 100%;
            height: 100%;
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10;
        }
    </style>
    <div class="agv-grid-wrap"><table class="agv-grid">
    """
    for r in range(GRID_SIZE):
        html += "<tr>"
        for c in range(GRID_SIZE):
            cls = "agv-cell"
            content = ""

            if GRID[r][c] == 1:
                cls += " agv-obstacle"
                content = "■"
            elif (r, c) == (START_POS[0], START_POS[1]):
                cls += " agv-start"
                content = "S"
            elif (r, c) == GOAL_POS:
                cls += " agv-goal"
                content = "G"
            elif (r, c) in optimal:
                cls += " agv-optimal"
                content = "<div class='agv-dot'></div>"
            elif (r, c) in explored:
                cls += " agv-explored"
                content = "<div class='agv-dot'></div>"
            elif (r, c) in scanned:
                cls += " agv-scanned"
                content = "●"

            if (r, c) == (car_r, car_c):
                angle = DIR_ROTATION[car_d]
                content = f"""
                <div class="agv-car">
                    <svg width="44" height="44" viewBox="0 0 24 24" style="transform: rotate({angle}deg); overflow: visible;">
                        <rect x="4" y="3" width="16" height="18" rx="4" fill="#2563EB" stroke="#0F172A" stroke-width="1.5"/>
                        <path d="M7 8 C7 6, 17 6, 17 8 L16 11 L8 11 Z" fill="#FFFFFF" opacity="0.95"/>
                        <circle cx="8" cy="3.5" r="2" fill="#FACC15" stroke="#0F172A" stroke-width="0.4"/>
                        <circle cx="16" cy="3.5" r="2" fill="#FACC15" stroke="#0F172A" stroke-width="0.4"/>
                    </svg>
                </div>
                """
            html += f"<td class='{cls}'>{content}</td>"
        html += "</tr>"
    html += "</table></div>"
    return html


# ============================================================
# 6. RENDER SƠ ĐỒ CÂY (ĐÃ FIX PHÓNG TO)
# ============================================================
def draw_tree_source(nodes, edges, current_id=None, scanned_ids=None, path_ids=None):
    scanned_ids = scanned_ids or set()
    path_ids = path_ids or set()

    # LIÊN KẾT GIÁ TRỊ ZOOM TRỰC TIẾP VÀO GRAPHVIZ ĐỂ NÚT CÂY THU NHỎ LẠI TỰ NHIÊN
    zoom_factor = st.session_state.get("tree_zoom", 0.65) * 1.4

    dot = graphviz.Digraph(engine="dot")
    dot.attr(
        rankdir="TB",
        splines="ortho",
        bgcolor="#FFFFFF",
        margin="0.04",
        pad="0.04",
        ranksep=str(0.32 * zoom_factor),
        nodesep=str(0.22 * zoom_factor),
        concentrate="false",
    )
    dot.attr(
        "node",
        shape="box",
        style="filled,rounded",
        fontname="Arial Bold",
        fontsize=str(int(10 * zoom_factor)),
        penwidth="1.55",
        margin="0.07,0.04",
        width=str(0.88 * zoom_factor),
        height=str(0.34 * zoom_factor),
        color="#334155",
        fillcolor="#FFFFFF",
        fontcolor="#0F172A",
    )
    dot.attr(
        "edge",
        color="#334155",
        fontname="Arial Bold",
        fontsize=str(int(8 * zoom_factor)),
        fontcolor="#0F172A",
        arrowsize=str(0.52 * zoom_factor),
        penwidth="1.25",
    )

    node_ids = {node["id"] for node in nodes}

    for node in nodes:
        node_id = node["id"]
        r, c, d, g = node["r"], node["c"], node["d"], node["g"]
        action = node.get("action", "")
        label = f"{ACTION_ICON.get(action, action)}\n({r},{c}) {DIR_ARROWS[d]} | g={g}"

        style = {"fillcolor": "#FFFFFF", "fontcolor": "#0F172A", "color": "#334155"}

        if node_id == "root-start":
            style = {"fillcolor": "#22C55E", "fontcolor": "#052E16", "color": "#15803D"}
        if node_id in scanned_ids:
            style = {"fillcolor": "#BFDBFE", "fontcolor": "#0F172A", "color": "#2563EB"}
        if node_id in path_ids:
            style = {"fillcolor": "#FDE047", "fontcolor": "#111827", "color": "#CA8A04"}
        if (r, c) == GOAL_POS:
            style = {"fillcolor": "#EF4444", "fontcolor": "#FFFFFF", "color": "#B91C1C"}
        if node_id == current_id:
            style = {"fillcolor": "#2563EB", "fontcolor": "#FFFFFF", "color": "#1E3A8A", "penwidth": "2.8"}

        dot.node(node_id, label=label, **style)

    for edge in edges:
        if edge["src"] in node_ids and edge["dst"] in node_ids:
            dot.edge(edge["src"], edge["dst"], label=edge.get("label", ""))

    return dot.source


# ============================================================
# 7. RENDER TOÀN BỘ MÀN HÌNH
# ============================================================
def render_scene(
    grid_slot,
    info_slot,
    tree_slot,
    log_slot,
    *,
    update_grid=True,
    update_info=True,
    update_tree=True,
    update_log=True,
    car_pos=None,
    current_cost=None,
    explored_cells=None,
    scanned_cells=None,
    optimal_cells=None,
    tree_nodes=None,
    tree_edges=None,
    current_tree_id=None,
    scanned_node_ids=None,
    path_node_ids=None,
    message=None,
    last_action=None,
):
    car_pos = car_pos if car_pos is not None else st.session_state.car_pos
    current_cost = current_cost if current_cost is not None else st.session_state.current_cost
    explored_cells = explored_cells if explored_cells is not None else st.session_state.explored_cells
    scanned_cells = scanned_cells if scanned_cells is not None else st.session_state.scanned_cells
    optimal_cells = optimal_cells if optimal_cells is not None else st.session_state.optimal_cells
    tree_nodes = tree_nodes if tree_nodes is not None else st.session_state.tree_nodes
    tree_edges = tree_edges if tree_edges is not None else st.session_state.tree_edges
    current_tree_id = current_tree_id if current_tree_id is not None else st.session_state.current_tree_id
    scanned_node_ids = scanned_node_ids if scanned_node_ids is not None else st.session_state.scanned_node_ids
    path_node_ids = path_node_ids if path_node_ids is not None else st.session_state.path_node_ids
    message = message if message is not None else st.session_state.status_message
    last_action = last_action if last_action is not None else st.session_state.last_action

    is_done = (car_pos[0], car_pos[1]) == GOAL_POS

    if update_grid:
        grid_slot.markdown(
            render_grid_html(explored_cells, scanned_cells, optimal_cells, car_pos, is_done),
            unsafe_allow_html=True,
        )

    if update_info:
        info_slot.info(
            f"📍 Ô: `({car_pos[0]}, {car_pos[1]})`  |  "
            f"🧭 Hướng: **{DIR_NAMES[car_pos[2]]}**  |  "
            f"💰 Chi phí: `{current_cost}`  |  "
            f"🎮 Lệnh: **{last_action}**"
        )

    if update_tree:
        tree_source = draw_tree_source(tree_nodes, tree_edges, current_tree_id, scanned_node_ids, path_node_ids)
        height = st.session_state.get("tree_height", 480)

        # TIÊM CSS ĐỂ KHỐNG CHẾ CHIỀU CAO VÀ NGĂN BLOCK GRAPHVIZ BỊ PHÓNG TO QUÁ ĐÀ
        st.markdown(
            f"""
            <style>
                [data-testid="stGraphvizChart"] {{
                    height: {height}px !important;
                    max-height: {height}px !important;
                    overflow: auto !important;
                    background: #FFFFFF !important;
                    border: 4px solid #0F172A !important;
                    border-radius: 18px !important;
                    padding: 15px !important;
                    box-shadow: 0 14px 35px rgba(15, 23, 42, 0.14) !important;
                }}
                [data-testid="stGraphvizChart"] svg {{
                    width: auto !important;
                    height: auto !important;
                    margin: 0 auto;
                    display: block;
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )

        with tree_slot.container():
            # Thêm dòng chip Legend trực quan ngay phía trên cây
            st.markdown(
                """
                <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; font-size: 13px; font-weight: 800; color: #0F172A;">
                    <span style="border: 2px solid #CBD5E1; border-radius: 999px; padding: 4px 9px; background: #F8FAFC;">🟩 Gốc Start</span>
                    <span style="border: 2px solid #CBD5E1; border-radius: 999px; padding: 4px 9px; background: #F8FAFC;">🟦 Node hiện tại</span>
                    <span style="border: 2px solid #CBD5E1; border-radius: 999px; padding: 4px 9px; background: #F8FAFC;">🟨 Đường tối ưu</span>
                    <span style="border: 2px solid #CBD5E1; border-radius: 999px; padding: 4px 9px; background: #F8FAFC;">🔹 Node đã quét</span>
                    <span style="border: 2px solid #CBD5E1; border-radius: 999px; padding: 4px 9px; background: #F8FAFC;">🟥 Đích G</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            # CHUYỂN use_container_width SANG False ĐỂ TRÁNH BỊ GIÃN NODE CHỮ TO ĐÙNG
            st.graphviz_chart(tree_source, use_container_width=False)

    if update_log:
        with log_slot.container():
            if message:
                if "Không" in message or "lỗi" in message.lower():
                    st.warning(message)
                elif "xong" in message.lower() or "tối ưu" in message.lower() or "tìm thấy" in message.lower():
                    st.success(message)
                else:
                    st.info(message)


# ============================================================
# 8. THAO TÁC THỦ CÔNG
# ============================================================
def add_manual_step(r, c, d, new_cost, action):
    parent_id = st.session_state.current_tree_id
    node_id = f"manual-{len(st.session_state.tree_nodes)}-{r}-{c}-{d}-{new_cost}"

    st.session_state.tree_nodes.append(make_node(node_id, r, c, d, new_cost, parent_id, action))
    st.session_state.tree_edges.append(make_edge(parent_id, node_id, action))
    st.session_state.current_tree_id = node_id
    st.session_state.car_pos = (r, c, d)
    st.session_state.current_cost = new_cost
    st.session_state.explored_cells.add((r, c))
    st.session_state.scanned_cells = set()
    st.session_state.optimal_cells = set()
    st.session_state.scanned_node_ids = set()
    st.session_state.path_node_ids = set()
    st.session_state.last_action = action
    st.session_state.status_message = f"Đã thực hiện: {action}."
    rerun_page()


# ============================================================
# 9. AUTO ANIMATION
# ============================================================
def animate_auto(grid_slot, info_slot, tree_slot, log_slot):
    plan = build_astar_plan_from_current()

    if plan is None or not plan["path_steps"]:
        st.session_state.status_message = "⚠️ Không tìm thấy đường đi hợp lệ đến đích G."
        render_scene(grid_slot, info_slot, tree_slot, log_slot)
        return

    base_nodes = list(st.session_state.tree_nodes)
    base_edges = list(st.session_state.tree_edges)
    base_explored = set(st.session_state.explored_cells)

    last_visible_count = 0
    for i, frame in enumerate(plan["scan_frames"], start=1):
        visible_count = frame["visible_count"] if st.session_state.show_scan_tree else last_visible_count
        visible_nodes = base_nodes + plan["new_nodes"][:visible_count]
        visible_node_ids = {node["id"] for node in visible_nodes}
        visible_edges = base_edges + [edge for edge in plan["new_edges"] if edge["dst"] in visible_node_ids]
        last_visible_count = visible_count

        render_scene(
            grid_slot,
            info_slot,
            tree_slot,
            log_slot,
            car_pos=st.session_state.car_pos,
            current_cost=st.session_state.current_cost,
            explored_cells=base_explored,
            scanned_cells=frame["scanned_cells"],
            optimal_cells=set(),
            tree_nodes=visible_nodes,
            tree_edges=visible_edges,
            current_tree_id=frame["current_node_id"],
            scanned_node_ids=frame["scanned_node_ids"],
            path_node_ids=set(),
            message=(
                f"🔎 Pha 1/3: A* đang quét trước khi chạy... {i}/{len(plan['scan_frames'])}. "
                f"{frame['message']}"
            ),
            last_action="A* SCAN",
            update_tree=True,
        )
        time.sleep(float(st.session_state.scan_delay))

    all_nodes = base_nodes + plan["new_nodes"]
    all_edges = base_edges + plan["new_edges"]

    render_scene(
        grid_slot,
        info_slot,
        tree_slot,
        log_slot,
        car_pos=st.session_state.car_pos,
        current_cost=st.session_state.current_cost,
        explored_cells=base_explored,
        scanned_cells=set().union(*(frame["scanned_cells"] for frame in plan["scan_frames"])),
        optimal_cells=plan["path_cells"],
        tree_nodes=all_nodes,
        tree_edges=all_edges,
        current_tree_id=st.session_state.current_tree_id,
        scanned_node_ids=set().union(*(frame["scanned_node_ids"] for frame in plan["scan_frames"])),
        path_node_ids=plan["path_node_ids"],
        message=f"✅ Pha 2/3: A* đã tính xong đường tối ưu: {len(plan['path_steps'])} bước, tổng chi phí g={plan['total_cost']}. Xe chuẩn bị chạy từng bước.",
        last_action="CHỐT PATH",
    )
    time.sleep(0.75)

    move_explored = set(base_explored)
    current_node_id = st.session_state.current_tree_id
    for step_index, step in enumerate(plan["path_steps"], start=1):
        move_explored.add((step["r"], step["c"]))
        current_node_id = step["id"]

        render_scene(
            grid_slot,
            info_slot,
            tree_slot,
            log_slot,
            car_pos=(step["r"], step["c"], step["d"]),
            current_cost=step["g"],
            explored_cells=move_explored,
            scanned_cells=set(),
            optimal_cells=plan["path_cells"],
            tree_nodes=all_nodes,
            tree_edges=all_edges,
            current_tree_id=current_node_id,
            scanned_node_ids=set(),
            path_node_ids=plan["path_node_ids"],
            message=f"🚗 Pha 3/3: Xe đang chạy từng bước theo đường tối ưu... bước {step_index}/{len(plan['path_steps'])}: {step['action']}.",
            last_action=step["action"],
            update_tree=True,
        )
        time.sleep(float(st.session_state.move_delay))

    final_step = plan["path_steps"][-1]
    st.session_state.tree_nodes = all_nodes
    st.session_state.tree_edges = all_edges
    st.session_state.current_tree_id = current_node_id
    st.session_state.car_pos = (final_step["r"], final_step["c"], final_step["d"])
    st.session_state.current_cost = final_step["g"]
    st.session_state.explored_cells = move_explored
    st.session_state.scanned_cells = set()
    st.session_state.optimal_cells = plan["path_cells"]
    st.session_state.scanned_node_ids = set()
    st.session_state.path_node_ids = plan["path_node_ids"]
    st.session_state.last_action = "AUTO DONE"
    st.session_state.status_message = f"🎯 Auto chạy xong. Đường tối ưu có {len(plan['path_steps'])} bước, tổng chi phí g={plan['total_cost']}."

    render_scene(grid_slot, info_slot, tree_slot, log_slot)


# ============================================================
# 10. GIAO DIỆN CHÍNH
# ============================================================
st.title("🖥️ HỆ THỐNG MÔ PHỎNG VÀ ĐỊNH TUYẾN XE AGV")
st.markdown("---")

col_left, col_right = st.columns([4, 7], gap="large")

with col_left:
    st.markdown("### 🗺️ Sa Bàn Lưới")
    grid_slot = st.empty()
    info_slot = st.empty()

    with st.expander("⚙️ Cài đặt mô phỏng", expanded=False):
        st.session_state.scan_delay = st.slider(
            "Tốc độ quét A* trước khi xe chạy",
            min_value=0.05,
            max_value=0.80,
            value=float(st.session_state.scan_delay),
            step=0.05,
        )
        st.session_state.move_delay = st.slider(
            "Tốc độ xe chạy từng bước",
            min_value=0.08,
            max_value=1.20,
            value=float(st.session_state.move_delay),
            step=0.04,
        )
        st.session_state.tree_zoom = st.slider(
            "Thu nhỏ / phóng to sơ đồ cây",
            min_value=0.40,
            max_value=1.20,
            value=float(st.session_state.tree_zoom),
            step=0.05,
        )
        st.session_state.tree_height = st.slider(
            "Chiều cao khung cây",
            min_value=300,
            max_value=700,
            value=int(st.session_state.tree_height),
            step=20,
        )
        st.session_state.show_scan_tree = st.checkbox(
            "Hiện cây mọc dần khi A* quét",
            value=bool(st.session_state.show_scan_tree),
        )
        st.caption("✅ Cây được hiển thị bằng st.graphviz_chart kết hợp tinh chỉnh kích thước vector.")

    controls_slot = st.container()

with col_right:
    st.markdown("### 🌳 Sơ Đồ Cây Trạng Thái")
    st.caption("Cây đi từ trên xuống dưới; nếu thuật toán sinh nhiều nhánh thì tự bung sang ngang. Auto sẽ cho cây mọc từng bước, không nhảy cái đùng.")
    tree_slot = st.empty()
    log_slot = st.empty()

# Render ban đầu.
render_scene(grid_slot, info_slot, tree_slot, log_slot)

# Nút điều khiển đặt sau khi đã có placeholder, để Auto có thể cập nhật màn hình trực tiếp.
with controls_slot:
    is_finished = (st.session_state.car_pos[0], st.session_state.car_pos[1]) == GOAL_POS
    current_r, current_c, current_d = st.session_state.car_pos

    st.markdown("##### 🕹️ Lái xe thủ công")
    row1_col1, row1_col2 = st.columns(2)

    if row1_col1.button("🔼 TIẾN (+1)", use_container_width=True, disabled=is_finished):
        dr, dc = DIRS[current_d]
        nr, nc = current_r + dr, current_c + dc
        if is_free_cell(nr, nc):
            add_manual_step(nr, nc, current_d, st.session_state.current_cost + COST_FORWARD, "Tiến")
        else:
            st.warning("🚧 Không thể tiến: phía trước là tường hoặc ngoài bản đồ.")

    if row1_col2.button("🔽 LÙI (+3)", use_container_width=True, disabled=is_finished):
        dr, dc = DIRS[current_d]
        nr, nc = current_r - dr, current_c - dc
        if is_free_cell(nr, nc):
            add_manual_step(nr, nc, current_d, st.session_state.current_cost + COST_BACKWARD, "Lùi")
        else:
            st.warning("🚧 Không thể lùi: phía sau là tường hoặc ngoài bản đồ.")

    row2_col1, row2_col2 = st.columns(2)

    if row2_col1.button("↩️ XOAY TRÁI (+2)", use_container_width=True, disabled=is_finished):
        add_manual_step(current_r, current_c, (current_d - 1) % 4, st.session_state.current_cost + COST_TURN, "Xoay trái")

    if row2_col2.button("↪️ XOAY PHẢI (+2)", use_container_width=True, disabled=is_finished):
        add_manual_step(current_r, current_c, (current_d + 1) % 4, st.session_state.current_cost + COST_TURN, "Xoay phải")

    st.markdown("---")
    auto_col, reset_col = st.columns(2)

    if auto_col.button("🤖 AUTO: QUÉT RỒI CHẠY", use_container_width=True, type="primary", disabled=is_finished):
        animate_auto(grid_slot, info_slot, tree_slot, log_slot)

    if reset_col.button("🔄 RESET", use_container_width=True):
        reset_system()
        rerun_page()
