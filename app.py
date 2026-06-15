import streamlit as st
import streamlit.components.v1 as components
import heapq
import time
import graphviz 

# --- 1. CẤU HÌNH GIAO DIỆN HỆ THỐNG ---
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
if "run_mode" not in st.session_state:
    st.session_state.run_mode = "THỦ CÔNG"

if "manual_history" not in st.session_state:
    # Lưu lịch sử bước đi thủ công để dựng cây đầy đủ: mỗi phần tử là (r, c, d, g, label, parent_id, node_id)
    start_id = f"0-0-2-0"
    st.session_state.manual_history = [(0, 0, 2, 0, "START (0,0) 🔽", None, start_id)]
    st.session_state.manual_car = (0, 0, 2)
    st.session_state.manual_cost = 0
    st.session_state.manual_explored = set([(0, 0)])

if "snapshots" not in st.session_state:
    st.session_state.snapshots = []
    st.session_state.step_index = 0
    st.session_state.final_path = []


# --- 4. THUẬT TOÁN A* SINH DỮ LIỆU TỰ ĐỘNG ---
def heuristic(r, c, g_r, g_c):
    return (abs(r - g_r) + abs(c - g_c)) * COST_FORWARD

def generate_search_snapshots():
    start_r, start_c, start_d = START_POS
    r_goal, c_goal = GOAL_POS
    start_node_id = f"{start_r}-{start_c}-{start_d}"
    h_start = heuristic(start_r, start_c, r_goal, c_goal)
    
    pq = [(h_start, 0, start_r, start_c, start_d, [(start_r, start_c, start_d)], start_node_id)]
    visited = {}
    snapshots_list = []
    explored_cells_acc = set()

    while pq:
        f, g, r, c, d, path, parent_id = heapq.heappop(pq)
        current_state = (r, c, d)
        
        if current_state in visited and visited[current_state] <= g:
            continue
            
        visited[current_state] = g
        current_node_id = f"{r}-{c}-{d}"
        explored_cells_acc.add((r, c))
        h_val = heuristic(r, c, r_goal, c_goal)
        
        # Tạo cây RÚT GỌN (Chỉ chứa đường đi thẳng tới trạng thái hiện tại)
        dot = graphviz.Digraph(comment='Compact Path Tree')
        dot.attr(rankdir='TB', size='8,8!', ratio='fill', bgcolor='transparent')
        dot.attr('node', style='filled,rounded', shape='box', 
                 fillcolor='#ffa502', fontcolor='#000000', color='#eccc68',
                 fontname='Arial Bold', fontsize='14', penwidth='3.0')
        dot.attr('edge', color='#2d3436', arrowsize='1.0', penwidth='2.5')
        
        # Xây dựng lại các nút thuộc đường đi hiện tại vào cây rút gọn
        for idx, (pr, pc, pd) in enumerate(path):
            p_node_id = f"{pr}-{pc}-{pd}"
            p_h = heuristic(pr, pc, r_goal, c_goal)
            # Tính toán g tạm thời cho nút trong chuỗi đường đi
            if idx == 0:
                dot.node(p_node_id, label=f"START ({pr},{pc}) {DIR_ARROWS[pd]}\nf={p_h}\n[g=0, h={p_h}]", fillcolor='#2ed573', fontcolor='#ffffff', color='#26af5f')
            elif (pr, pc) == GOAL_POS:
                dot.node(p_node_id, label=f"GOAL ({pr},{pc}) {DIR_ARROWS[pd]}\nf={g}\n[g={g}, h=0]", fillcolor='#ff4757', fontcolor='#ffffff', color='#ff2e44')
            else:
                # Nút đầu mút hiện tại tô màu nổi bật hơn
                if idx == len(path) - 1:
                    dot.node(p_node_id, label=f"XE ({pr},{pc}) {DIR_ARROWS[pd]}\nf={f}\n[g={g}, h={h_val}]", fillcolor='#1e90ff', fontcolor='#ffffff', color='#0984e3')
                else:
                    dot.node(p_node_id, label=f"({pr},{pc}) {DIR_ARROWS[pd]}\n[g=..., h={p_h}]", fillcolor='#f1f2f6', fontcolor='#2d3436', color='#ced6e0')
            
            if idx > 0:
                prev_r, prev_c, prev_d = path[idx-1]
                dot.edge(f"{prev_r}-{prev_c}-{prev_d}", p_node_id)

        snapshots_list.append({
            "explored": set(explored_cells_acc),
            "car": (r, c, d),
            "cost": g,
            "tree": dot.source
        })

        if (r, c) == GOAL_POS:
            return path, snapshots_list

        # Phát triển tập con con
        dr, dc = DIRS[d]
        # 1. Đi Tiến
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
            heapq.heappush(pq, (g + COST_FORWARD + heuristic(nr, nc, r_goal, c_goal), g + COST_FORWARD, nr, nc, d, path + [(nr, nc, d)], current_node_id))
        # 2. Đi Lùi
        nr, nc = r - dr, c - dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
            heapq.heappush(pq, (g + COST_BACKWARD + heuristic(nr, nc, r_goal, c_goal), g + COST_BACKWARD, nr, nc, d, path + [(nr, nc, d)], current_node_id))
        # 3. Xoay Hướng
        for next_d in [(d - 1) % 4, (d + 1) % 4]:
            heapq.heappush(pq, (g + COST_TURN + heuristic(r, c, r_goal, c_goal), g + COST_TURN, r, c, next_d, path + [(r, c, next_d)], current_node_id))

    return [], snapshots_list


# --- 5. DỰNG CÂY ĐẦY ĐỦ CHO CHẾ ĐỘ THỦ CÔNG ---
def build_full_manual_tree():
    dot = graphviz.Digraph(comment='Full Manual Tree')
    dot.attr(rankdir='TB', size='9,9!', ratio='fill', bgcolor='transparent')
    dot.attr('node', style='filled,rounded', shape='box', 
             fillcolor='#f1f2f6', fontcolor='#2d3436', color='#4b5563',
             fontname='Arial Bold', fontsize='12', penwidth='2.0')
    dot.attr('edge', color='#2d3436', arrowsize='0.8', penwidth='2.0')
    
    # Vẽ toàn bộ lịch sử các nút đã bấm
    for i, (r, c, d, g, label, parent_id, node_id) in enumerate(st.session_state.manual_history):
        h_val = heuristic(r, c, GOAL_POS[0], GOAL_POS[1])
        f_val = g + h_val
        
        if i == 0:
            dot.node(node_id, label=f"START ({r},{c}) {DIR_ARROWS[d]}\nf={f_val}\n[g=0, h={h_val}]", fillcolor='#2ed573', fontcolor='#ffffff', color='#26af5f')
        elif (r, c) == st.session_state.manual_car:
            # Nút hiện tại của xe tô màu xanh dương đậm to rõ
            dot.node(node_id, label=f"XE HIỆN TẠI ({r},{c}) {DIR_ARROWS[d]}\nf={f_val}\n[g={g}, h={h_val}]", fillcolor='#1e90ff', fontcolor='#ffffff', color='#0984e3')
        else:
            dot.node(node_id, label=f"({r},{c}) {DIR_ARROWS[d]}\nf={f_val}\n[g={g}, h={h_val}]")
            
        if parent_id:
            dot.edge(parent_id, node_id)
            
    return dot.source


# --- 6. RENDER KHUNG BẢN ĐỒ HTML ---
def render_grid_dynamic(explored, car_pos, final_path_cells):
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
            elif (r, c) in final_path_cells: cell_class += " cell-path"; content = "<div class='path-dot'></div>"
            elif (r, c) in explored: cell_class += " cell-explored"

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


# --- 7. GIAO DIỆN ĐIỀU KHIỂN CHÍNH ---
st.title("📊 MÔ PHỎNG PHÁT TRIỂN CÂY THEO CHẾ ĐỘ THUYẾT TRÌNH")

mode_select = st.selectbox("🎯 CHỌN CHẾ ĐỘ HOẠT ĐỘNG:", ["THỦ CÔNG", "TỰ ĐỘNG"])

if mode_select != st.session_state.run_mode:
    st.session_state.run_mode = mode_select
    if mode_select == "TỰ ĐỘNG" and not st.session_state.snapshots:
        path, snapshots = generate_search_snapshots()
        st.session_state.snapshots = snapshots
        st.session_state.final_path = path
        st.session_state.step_index = 0
    rerun_page()

st.markdown("---")

col_left, col_right = st.columns([4, 6], gap="large")

# Tách dữ liệu hiển thị theo chế độ
if st.session_state.run_mode == "TỰ ĐỘNG" and st.session_state.snapshots:
    current_snap = st.session_state.snapshots[st.session_state.step_index]
    explored_now = current_snap["explored"]
    car_now = current_snap["car"]
    cost_now = current_snap["cost"]
    tree_dot_now = current_snap["tree"]
    
    is_last = (st.session_state.step_index == len(st.session_state.snapshots) - 1)
    path_cells_now = set((n[0], n[1]) for n in st.session_state.final_path) if is_last else set()
else:
    explored_now = st.session_state.manual_explored
    car_now = st.session_state.manual_car
    cost_now = st.session_state.manual_cost
    path_cells_now = set()
    tree_dot_now = build_full_manual_tree()

with col_left:
    st.markdown("### 🗺️ Bản Đồ Sa Bàn Lưới")
    components.html(render_grid_dynamic(explored_now, car_now, path_cells_now), height=370, scrolling=False)
    
    st.markdown("#### 📝 Thông Số Trực Quan:")
    st.info(f"• **Vị trí xe:** Ô `({car_now[0]}, {car_now[1]})` — Hướng: **{DIR_NAMES[car_now[2]]}**\n"
            f"• **Chi phí lũy kế g(n):** `{cost_now}`")
    
    # --- KHU VỰC ĐIỀU KHIỂN THỦ CÔNG CÓ ĐẦY ĐỦ 4 NÚT XOAY ---
    if st.session_state.run_mode == "THỦ CÔNG":
        st.markdown("🕹️ **Cụm Phím Lái Thủ Công & Sinh Nhánh Cây:**")
        
        # Lấy thông tin nút cha hiện tại trước khi tạo hành động mới
        parent_node_id = f"{car_now[0]}-{car_now[1]}-{car_now[2]}-{cost_now}"
        
        cx1, cx2 = st.columns(2)
        if cx1.button("🔼 ĐI TIẾN (+1)", use_container_width=True):
            dr, dc = DIRS[car_now[2]]
            nr, nc = car_now[0] + dr, car_now[1] + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
                new_cost = cost_now + COST_FORWARD
                new_id = f"{nr}-{nc}-{car_now[2]}-{new_cost}"
                st.session_state.manual_history.append((nr, nc, car_now[2], new_cost, f"Tiến ({nr},{nc})", parent_node_id, new_id))
                st.session_state.manual_car = (nr, nc, car_now[2])
                st.session_state.manual_cost = new_cost
                st.session_state.manual_explored.add((nr, nc))
                rerun_page()
                
        if cx2.button("🔽 ĐI LÙI (+3)", use_container_width=True):
            dr, dc = DIRS[car_now[2]]
            nr, nc = car_now[0] - dr, car_now[1] - dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
                new_cost = cost_now + COST_BACKWARD
                new_id = f"{nr}-{nc}-{car_now[2]}-{new_cost}"
                st.session_state.manual_history.append((nr, nc, car_now[2], new_cost, f"Lùi ({nr},{nc})", parent_node_id, new_id))
                st.session_state.manual_car = (nr, nc, car_now[2])
                st.session_state.manual_cost = new_cost
                st.session_state.manual_explored.add((nr, nc))
                rerun_page()
                
        cx3, cx4 = st.columns(2)
        if cx3.button("🔄 XOAY TRÁI (+2)", use_container_width=True):
            new_d = (car_now[2] - 1) % 4
            new_cost = cost_now + COST_TURN
            new_id = f"{car_now[0]}-{car_now[1]}-{new_d}-{new_cost}"
            st.session_state.manual_history.append((car_now[0], car_now[1], new_d, new_cost, f"Xoay Trái", parent_node_id, new_id))
            st.session_state.manual_car = (car_now[0], car_now[1], new_d)
            st.session_state.manual_cost = new_cost
            rerun_page()
            
        if cx4.button("🔄 XOAY PHẢI (+2)", use_container_width=True):
            new_d = (car_now[2] + 1) % 4
            new_cost = cost_now + COST_TURN
            new_id = f"{car_now[0]}-{car_now[1]}-{new_d}-{new_cost}"
            st.session_state.manual_history.append((car_now[0], car_now[1], new_d, new_cost, f"Xoay Phải", parent_node_id, new_id))
            st.session_state.manual_car = (car_now[0], car_now[1], new_d)
            st.session_state.manual_cost = new_cost
            rerun_page()

    # --- KHU VỰC ĐIỀU KHIỂN TỰ ĐỘNG CHẠY TỪNG BƯỚC MƯỢT MÀ ---
    if st.session_state.run_mode == "TỰ ĐỘNG" and st.session_state.snapshots:
        st.markdown("🤖 **Bộ Điều Khiển Tiến Trình Tự Động:**")
        total_steps = len(st.session_state.snapshots)
        st.write(f"Tiến độ thuật toán: **Bước {st.session_state.step_index + 1} / {total_steps}**")
        
        c1, c2, c3 = st.columns(3)
        if c1.button("⬅️ LÙI 1 BƯỚC", use_container_width=True):
            if st.session_state.step_index > 0:
                st.session_state.step_index -= 1
                rerun_page()
        if c2.button("TIẾP 1 BƯỚC ➡️", use_container_width=True):
            if st.session_state.step_index < total_steps - 1:
                st.session_state.step_index += 1
                rerun_page()
        if c3.button("▶️ TỰ ĐỘNG CHẠY", use_container_width=True):
            for idx in range(st.session_state.step_index, total_steps):
                st.session_state.step_index = idx
                time.sleep(0.4)  # Giãn thời gian để giám khảo kịp nhìn sự đồng bộ giữa xe và cây rút gọn
                rerun_page()

    st.markdown("---")
    if st.button("🔄 RESET TOÀN BỘ MÔ PHỎNG", use_container_width=True):
        st.session_state.snapshots = []
        st.session_state.step_index = 0
        st.session_state.final_path = []
        start_id = f"0-0-2-0"
        st.session_state.manual_history = [(0, 0, 2, 0, "START (0,0) 🔽", None, start_id)]
        st.session_state.manual_car = (0, 0, 2)
        st.session_state.manual_cost = 0
        st.session_state.manual_explored = set([(0, 0)])
        rerun_page()

with col_right:
    if st.session_state.run_mode == "TỰ ĐỘNG":
        st.markdown("### 🌳 Sơ Đồ Cây Rút Gọn (Chỉ hiển thị đường đi tối ưu)")
    else:
        st.markdown("### 🌳 Sơ Đồ Cây Đầy Đủ (Đồng bộ theo từng nút bạn bấm)")
        
    if tree_dot_now:
        st.graphviz_chart(tree_dot_now, use_container_width=True)
