import streamlit as st
import streamlit.components.v1 as components
import heapq
import time
import graphviz 

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
    [0, 1, 1, 1, 1, 0]
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


# --- 3. KHỞI TẠO BỘ NHỚ TRẠNG THÁI (SESSION STATE) ---
if "car_pos" not in st.session_state:
    st.session_state.car_pos = START_POS

if "current_cost" not in st.session_state:
    st.session_state.current_cost = 0

if "explored_cells" not in st.session_state:
    st.session_state.explored_cells = set([(0, 0)])

# Lịch sử lưu cây: (r, c, d, cost, node_id, parent_id)
if "tree_history" not in st.session_state:
    start_id = "0-0-2-0"
    st.session_state.tree_history = [(0, 0, 2, 0, start_id, None)]

if "is_auto_running" not in st.session_state:
    st.session_state.is_auto_running = False

if "auto_steps" not in st.session_state:
    st.session_state.auto_steps = []

if "auto_index" not in st.session_state:
    st.session_state.auto_index = 0


# --- 4. THUẬT TOÁN A* TÌM ĐƯỜNG TỐI ƯU CHO NÚT AUTO ---
def heuristic(r, c, g_r, g_c):
    return (abs(r - g_r) + abs(c - g_c)) * COST_FORWARD

def run_astar_search():
    start_r, start_c, start_d = START_POS
    r_goal, c_goal = GOAL_POS
    start_id = f"{start_r}-{start_c}-{start_d}-0"
    h_start = heuristic(start_r, start_c, r_goal, c_goal)
    
    pq = [(h_start, 0, start_r, start_c, start_d, [(start_r, start_c, start_d, 0, start_id, None)])]
    visited = {}

    while pq:
        f, g, r, c, d, full_history = heapq.heappop(pq)
        
        if (r, c, d) in visited and visited[(r, c, d)] <= g:
            continue
        visited[(r, c, d)] = g
        
        if (r, c) == c_goal:
            return full_history

        current_id = f"{r}-{c}-{d}-{g}"
        
        # Phát triển các nhánh con hợp lệ
        dr, dc = DIRS[d]
        # 1. TIẾN
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
            next_g = g + COST_FORWARD
            next_id = f"{nr}-{nc}-{d}-{next_g}"
            pq_item = (next_g + heuristic(nr, nc, r_goal, c_goal), next_g, nr, nc, d, full_history + [(nr, nc, d, next_g, next_id, current_id)])
            heapq.heappush(pq, pq_item)
            
        # 2. LÙI
        nr, nc = r - dr, c - dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
            next_g = g + COST_BACKWARD
            next_id = f"{nr}-{nc}-{d}-{next_g}"
            pq_item = (next_g + heuristic(nr, nc, r_goal, c_goal), next_g, nr, nc, d, full_history + [(nr, nc, d, next_g, next_id, current_id)])
            heapq.heappush(pq, pq_item)
            
        # 3. XOAY HƯỚNG
        for next_d in [(d - 1) % 4, (d + 1) % 4]:
            next_g = g + COST_TURN
            next_id = f"{r}-{c}-{next_d}-{next_g}"
            pq_item = (next_g + heuristic(r, c, r_goal, c_goal), next_g, r, c, next_d, full_history + [(r, c, next_d, next_g, next_id, current_id)])
            heapq.heappush(pq, pq_item)
            
    return full_history


# --- 5. DỰNG SƠ ĐỒ CÂY TỰ ĐỘNG TỎA NHÁNH (XỊN MỊN CHO SLIDE) ---
def draw_beautiful_tree(history_list):
    # Sử dụng rankdir='TB' gốc giúp Graphviz tự động tỏa nhánh sang ngang khi rẽ nhánh cực kỳ cân đối
    dot = graphviz.Digraph()
    dot.attr(rankdir='TB', size='4.5,4.5!', fixedsize='true', bgcolor='transparent')
    dot.attr('node', style='filled,rounded', shape='box', 
             fillcolor='#1e252b', fontcolor='#ffffff', color='#34414c',
             fontname='Arial Bold', fontsize='9', penwidth='1.2', 
             width='0.65', height='0.26')
    dot.attr('edge', color='#ffffff', arrowsize='0.4', penwidth='1.0')
    
    for i, (r, c, d, g, node_id, parent_id) in enumerate(history_list):
        if i == 0:
            dot.node(node_id, label=f"({r},{c}) {DIR_ARROWS[d]}", fillcolor='#2ed573', fontcolor='#ffffff', color='#26af5f')
        elif (r, c) == GOAL_POS:
            dot.node(node_id, label=f"({r},{c}) {DIR_ARROWS[d]}", fillcolor='#ff4757', fontcolor='#ffffff', color='#ff2e44')
        elif i == len(history_list) - 1:
            dot.node(node_id, label=f"({r},{c}) {DIR_ARROWS[d]}", fillcolor='#1e90ff', fontcolor='#ffffff', color='#0984e3')
        else:
            dot.node(node_id, label=f"({r},{c}) {DIR_ARROWS[d]}")
            
        if parent_id:
            dot.edge(parent_id, node_id)
            
    return dot.source


# --- 6. RENDER KHUNG BẢN ĐỒ HTML ---
def render_grid_dynamic(explored, car_pos, is_done):
    car_r, car_c, car_d = car_pos
    html = """
    <style>
        body { background-color: transparent; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; overflow: hidden; }
        .grid-container { display: flex; justify-content: center; align-items: center; background-color: #12181b; padding: 12px; border-radius: 14px; border: 3px solid #ffffff; box-shadow: 0 0 22px rgba(255, 255, 255, 0.28); width: 370px; max-width: 100%; margin: 0 auto; box-sizing: border-box; }
        .grid-table { border-collapse: separate; border-spacing: 5px; table-layout: fixed; width: 340px; height: 340px; }
        .grid-cell { width: 50px !important; height: 50px !important; box-sizing: border-box; text-align: center; vertical-align: middle; font-family: 'Segoe UI', sans-serif; font-size: 14px; font-weight: bold; border-radius: 8px; position: relative; background-color: #1e252b; border: 1px dashed #34414c; color: #ffffff; }
        .cell-obstacle { background-color: #3d4d59 !important; border: 1px solid #4f616f !important; }
        .cell-start { background-color: #2ed573 !important; color: #ffffff !important; }
        .cell-goal { background-color: #ff4757 !important; color: #ffffff !important; }
        .cell-explored { background-color: #103136 !important; border: 1px dashed #00cec9 !important; }
        .cell-path { background-color: #2ed573 !important; border: 1px solid #55efc4 !important; box-shadow: inset 0 0 18px rgba(85, 239, 196, 0.85) !important; }
        .path-dot { width: 14px; height: 14px; background-color: #ffffff; border-radius: 50%; margin: auto; box-shadow: 0 0 16px #ffffff; }
        .car-wrapper { width: 100%; height: 100%; display: flex; justify-content: center; align-items: center; position: absolute; top: 0; left: 0; z-index: 10; }
    </style>
    <div class='grid-container'><table class='grid-table'>
    """
    for r in range(GRID_SIZE):
        html += "<tr>"
        for c in range(GRID_SIZE):
            cell_class = "grid-cell"
            content = ""
            if GRID[r][c] == 1: cell_class += " cell-obstacle"
            elif r == START_POS[0] and c == START_POS[1]: cell_class += " cell-start"; content = "S"
            elif r == GOAL_POS[0] and c == GOAL_POS[1]: cell_class += " cell-goal"; content = "G"
            elif (r, c) in explored:
                cell_class += " cell-path" if is_done else " cell-explored"
                if is_done: content = "<div class='path-dot'></div>"

            if r == car_r and c == car_c:
                angle = DIR_ROTATION[car_d]
                content = f"""
                <div class='car-wrapper'>
                    <svg width="40" height="40" viewBox="0 0 24 24" style="transform: rotate({angle}deg); overflow: visible;">
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


# --- 7. XỬ LÝ ĐỘNG CƠ BƯỚC CHẠY AUTO REAL-TIME ---
if st.session_state.is_auto_running:
    if st.session_state.auto_index < len(st.session_state.auto_steps):
        time.sleep(0.4)  # Tốc độ nhảy bước mượt mà
        curr_snap = st.session_state.auto_steps[st.session_state.auto_index]
        st.session_state.car_pos = (curr_snap[0], curr_snap[1], curr_snap[2])
        st.session_state.current_cost = curr_snap[3]
        st.session_state.explored_cells.add((curr_snap[0], curr_snap[1]))
        st.session_state.tree_history = st.session_state.auto_steps[:st.session_state.auto_index + 1]
        st.session_state.auto_index += 1
        rerun_page()
    else:
        st.session_state.is_auto_running = False


# --- 8. GIAO DIỆN HIỂN THỊ CHÍNH ---
st.title("🖥️ HỆ THỐNG MÔ PHỎNG VÀ ĐỊNH TUYẾN XE AGV")
st.markdown("---")

col_left, col_right = st.columns([5, 5], gap="large")

is_finished = (st.session_state.car_pos[0], st.session_state.car_pos[1]) == GOAL_POS

with col_left:
    st.markdown("### 🗺️ Sa Bàn Lưới")
    components.html(render_grid_dynamic(st.session_state.explored_cells, st.session_state.car_pos, is_finished), height=370, scrolling=False)
    st.info(f"📍 Tọa độ: `({st.session_state.car_pos[0]}, {st.session_state.car_pos[1]})` | 🧭 Hướng: **{DIR_NAMES[st.session_state.car_pos[2]]}** | 💰 Tổng chi phí: `{st.session_state.current_cost}`")

with col_right:
    st.markdown("### 🌳 Sơ Đồ Cây Trạng Thái")
    tree_src = draw_beautiful_tree(st.session_state.tree_history)
    st.graphviz_chart(tree_src, use_container_width=False)


# --- 9. CỤM 5 NÚT ĐIỀU KHIỂN TÍCH HỢP Ở DƯỚI CÙNG ---
st.markdown("---")
st.markdown("### 🕹️ BỘ BẢNG ĐIỀU KHIỂN HỆ THỐNG")

cx1, cx2, cx3, cx4, cx5, cx6 = st.columns(6)

current_r, current_c, current_d = st.session_state.car_pos
parent_id = f"{current_r}-{current_c}-{current_d}-{st.session_state.current_cost}"

# NÚT TIẾN
if cx1.button("🔼 TIẾN", use_container_width=True, disabled=st.session_state.is_auto_running or is_finished):
    dr, dc = DIRS[current_d]
    nr, nc = current_r + dr, current_c + dc
    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
        new_cost = st.session_state.current_cost + COST_FORWARD
        new_id = f"{nr}-{nc}-{current_d}-{new_cost}"
        st.session_state.tree_history.append((nr, nc, current_d, new_cost, new_id, parent_id))
        st.session_state.car_pos = (nr, nc, current_d)
        st.session_state.current_cost = new_cost
        st.session_state.explored_cells.add((nr, nc))
        rerun_page()

# NÚT LÙI
if cx2.button("🔽 LÙI", use_container_width=True, disabled=st.session_state.is_auto_running or is_finished):
    dr, dc = DIRS[current_d]
    nr, nc = current_r - dr, current_c - dc
    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
        new_cost = st.session_state.current_cost + COST_BACKWARD
        new_id = f"{nr}-{nc}-{current_d}-{new_cost}"
        st.session_state.tree_history.append((nr, nc, current_d, new_cost, new_id, parent_id))
        st.session_state.car_pos = (nr, nc, current_d)
        st.session_state.current_cost = new_cost
        st.session_state.explored_cells.add((nr, nc))
        rerun_page()

# NÚT XOAY TRÁI
if cx3.button("🔄 XOAY TRÁI", use_container_width=True, disabled=st.session_state.is_auto_running or is_finished):
    new_d = (current_d - 1) % 4
    new_cost = st.session_state.current_cost + COST_TURN
    new_id = f"{current_r}-{current_c}-{new_d}-{new_cost}"
    st.session_state.tree_history.append((current_r, current_c, new_d, new_cost, new_id, parent_id))
    st.session_state.car_pos = (current_r, current_c, new_d)
    st.session_state.current_cost = new_cost
    rerun_page()

# NÚT XOAY PHẢI
if cx4.button("🔄 XOAY PHẢI", use_container_width=True, disabled=st.session_state.is_auto_running or is_finished):
    new_d = (current_d + 1) % 4
    new_cost = st.session_state.current_cost + COST_TURN
    new_id = f"{current_r}-{current_c}-{new_d}-{new_cost}"
    st.session_state.tree_history.append((current_r, current_c, new_d, new_cost, new_id, parent_id))
    st.session_state.car_pos = (current_r, current_c, new_d)
    st.session_state.current_cost = new_cost
    rerun_page()

# NÚT AUTO CHẠY TỰ ĐỘNG REALTIME
if cx5.button("🤖 AUTO", use_container_width=True, fill_container=True, disabled=st.session_state.is_auto_running or is_finished):
    st.session_state.auto_steps = run_astar_search()
    st.session_state.auto_index = 0
    st.session_state.is_auto_running = True
    rerun_page()

# NÚT RESET MÔ PHỎNG
if cx6.button("🔄 RESET", use_container_width=True):
    st.session_state.car_pos = START_POS
    st.session_state.current_cost = 0
    st.session_state.explored_cells = set([(0, 0)])
    start_id = "0-0-2-0"
    st.session_state.tree_history = [(0, 0, 2, 0, start_id, None)]
    st.session_state.is_auto_running = False
    st.session_state.auto_steps = []
    st.session_state.auto_index = 0
    rerun_page()
