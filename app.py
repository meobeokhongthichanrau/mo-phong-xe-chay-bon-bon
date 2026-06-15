import streamlit as st
import streamlit.components.v1 as components
import heapq
import time
# 0. IMPORT THÊM THƯ VIỆN ĐỂ VẼ CÂY
import graphviz 

# --- 1. CẤU HÌNH GIAO DIỆN ĐẦU FILE ---
# Chuyển sang layout="wide" để đủ chỗ cho 2 cột lớn
st.set_page_config(page_title="Hệ Thống AGV Bãi Đỗ & AI Search Tree", layout="wide")

def rerun_page():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# --- 2. CẤU HÌNH MA TRẬN BÃI ĐỖ TỐI ƯU ---
GRID_SIZE = 6

GRID = [
    [0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0]
]

START_POS = (0, 0, 2)  # Xuất phát tại ô (0,0), xe hướng Nam
GOAL_POS = (5, 5)      # Đích đỗ tại ô (5,5)

COST_FORWARD = 1
COST_TURN = 2
COST_BACKWARD = 3

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_NAMES = ["Bắc 🔼", "Đông ▶️", "Nam 🔽", "Tây ◀️"]
DIR_ROTATION = {
    0: 0,
    1: 90,
    2: 180,
    3: 270
}


# --- 3. KHỞI TẠO TRẠNG THÁI ---
if "car_r" not in st.session_state:
    st.session_state.car_r = START_POS[0]
    st.session_state.car_c = START_POS[1]
    st.session_state.car_d = START_POS[2]
    st.session_state.cost = 0
    st.session_state.mode = "MANUAL"
    st.session_state.explored_cells = set()
    st.session_state.final_path_cells = set()
    # Thêm state để lưu cây graphviz cuối cùng
    st.session_state.final_tree_dot = None 


# --- 4. THUẬT TOÁN A* LÕI ---
def heuristic(r, c, g_r, g_c):
    return (abs(r - g_r) + abs(c - g_c)) * COST_FORWARD


def solve_astar_with_vis():
    start_r, start_c, start_d = START_POS
    r_goal, c_goal = GOAL_POS

    # Khởi tạo Graphviz để vẽ cây
    dot = graphviz.Digraph(comment='Search Tree')
    
    # Định nghĩa màu sắc các nút cho đẹp
    dot.attr('node', style='filled', color='#34414c', fontcolor='white', shape='circle')
    dot.attr('edge', color='white', fontcolor='white')

    # Trạng thái ban đầu
    start_state = (start_r, start_c, start_d)
    start_h = heuristic(start_r, start_c, r_goal, c_goal)
    start_f = 0 + start_h
    
    # Tạo node gốc trong cây Graphviz: label hiển thị (f=f, g=g, h=h)
    start_label = f"S ({start_r},{start_c})\nD:{start_d}\ng=0, h={start_h}\nf={start_f}"
    # Đặt ID nút duy nhất là "r-c-d" để dễ gọi
    start_node_id = f"{start_r}-{start_c}-{start_d}"
    dot.node(start_node_id, label=start_label, color='#0d3b32', fontcolor='#00b894', penwidth='2')

    pq = [
        (
            start_f,
            0,
            start_r,
            start_c,
            start_d,
            [(start_r, start_c, start_d)],
            start_node_id # Lưu ID nút cha để vẽ cây
        )
    ]

    visited = {} # Dùng dict để lưu g_cost thấp nhất từng trạng thái
    # exploration_order_full lưu: (r, c, d, current_node_id, parent_node_id, g, f, h, is_goal)
    exploration_order_full = [] 

    while pq:
        f, g, r, c, d, path, parent_id = heapq.heappop(pq)
        current_state = (r, c, d)
        
        # Nếu đã visit với chi phí thấp hơn thì bỏ qua
        if current_state in visited and visited[current_state] <= g:
            continue
        
        visited[current_state] = g
        h_cost = f - g
        current_node_id = f"{r}-{c}-{d}"
        
        # Thêm vào order để Phase 1 mô phỏng vẽ dần cây
        exploration_order_full.append((r, c, d, current_node_id, parent_id, g, f, h_cost, (r, c) == GOAL_POS))

        if (r, c) == GOAL_POS:
            return path, exploration_order_full

        # --- Thử các hành động để tạo nút con ---
        
        # 1. Tiến
        dr, dc = DIRS[d]
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
            new_g = g + COST_FORWARD
            new_h = heuristic(nr, nc, r_goal, c_goal)
            new_f = new_g + new_h
            child_node_id = f"{nr}-{nc}-{d}"
            
            heapq.heappush(pq, (new_f, new_g, nr, nc, d, path + [(nr, nc, d)], current_node_id))

        # 2. Lùi
        nr, nc = r - dr, c - dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
            new_g = g + COST_BACKWARD
            new_h = heuristic(nr, nc, r_goal, c_goal)
            new_f = new_g + new_h
            # Lũi không đổi hướng nhưng phải phân biệt ID
            child_node_id = f"{nr}-{nc}-{d}-b" 
            
            heapq.heappush(pq, (new_f, new_g, nr, nc, d, path + [(nr, nc, d)], current_node_id))

        # 3. Rẽ trái / rẽ phải (giữ nguyên tọa độ)
        for next_d in [(d - 1) % 4, (d + 1) % 4]:
            new_g = g + COST_TURN
            new_h = heuristic(r, c, r_goal, c_goal)
            new_f = new_g + new_h
            child_node_id = f"{r}-{c}-{next_d}"
            
            heapq.heappush(pq, (new_f, new_g, r, c, next_d, path + [(r, c, next_d)], current_node_id))

    return [], exploration_order_full


# --- 5. RENDER SÂN ĐỖ ---
def render_grid():
    html = """
    <style>
        body {
            background-color: transparent;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
        }

        .grid-container {
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #12181b;
            padding: 12px;
            border-radius: 14px;
            border: 3px solid #ffffff;
            box-shadow: 0 0 22px rgba(255, 255, 255, 0.28);
            width: 390px;
            max-width: 100%;
            margin: 0 auto;
            box-sizing: border-box;
        }

        .grid-table {
            border-collapse: separate;
            border-spacing: 5px;
            table-layout: fixed;
            width: 350px;
            height: 350px;
        }

        .grid-cell {
            width: 52px !important;
            height: 52px !important;
            box-sizing: border-box;
            text-align: center;
            vertical-align: middle;
            font-family: 'Segoe UI', sans-serif;
            font-size: 14px;
            font-weight: bold;
            border-radius: 8px;
            position: relative;
            background-color: #1e252b;
            border: 1px dashed #34414c;
            overflow: hidden;
            color: #ffffff;
        }

        .cell-obstacle {
            background-color: #3d4d59 !important;
            border: 1px solid #4f616f !important;
        }

        .cell-start {
            background-color: #0d3b32 !important;
            color: #00b894 !important;
            border: 1px solid #00b894 !important;
            box-shadow: inset 0 0 14px rgba(0, 184, 148, 0.35);
        }

        .cell-goal {
            background-color: #4c1c24 !important;
            color: #ff7675 !important;
            border: 1px solid #ff7675 !important;
            box-shadow: inset 0 0 14px rgba(255, 118, 117, 0.35);
        }

        .cell-explored {
            background-color: #103136 !important;
            border: 1px dashed #00cec9 !important;
            box-shadow: inset 0 0 12px rgba(0, 206, 201, 0.25);
        }

        .cell-path {
            background-color: #00b894 !important;
            border: 1px solid #55efc4 !important;
            box-shadow: inset 0 0 18px rgba(85, 239, 196, 0.85) !important;
        }

        .path-dot {
            width: 14px;
            height: 14px;
            background-color: #ffffff;
            border-radius: 50%;
            margin: auto;
            box-shadow: 0 0 16px #ffffff;
        }

        .car-wrapper {
            width: 100%;
            height: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            position: absolute;
            top: 0;
            left: 0;
            z-index: 10;
        }
    </style>

    <div class='grid-container'>
    <table class='grid-table'>
    """

    for r in range(GRID_SIZE):
        html += "<tr>"

        for c in range(GRID_SIZE):
            cell_class = "grid-cell"
            content = ""

            if GRID[r][c] == 1:
                cell_class += " cell-obstacle"

            elif r == START_POS[0] and c == START_POS[1]:
                cell_class += " cell-start"
                content = "S"

            elif r == GOAL_POS[0] and c == GOAL_POS[1]:
                cell_class += " cell-goal"
                content = "G"

            elif (r, c) in st.session_state.final_path_cells:
                cell_class += " cell-path"
                content = "<div class='path-dot'></div>"

            elif (r, c) in st.session_state.explored_cells:
                cell_class += " cell-explored"

            if r == st.session_state.car_r and c == st.session_state.car_c:
                angle = DIR_ROTATION[st.session_state.car_d]

                content = f"""
                <div class='car-wrapper'>
                    <svg width="42" height="42" viewBox="0 0 24 24" style="transform: rotate({angle}deg); overflow: visible;">
                        <defs>
                            <filter id="headlightGlow" x="-80%" y="-80%" width="260%" height="260%">
                                <feGaussianBlur stdDeviation="2.8" result="coloredBlur"/>
                                <feMerge>
                                    <feMergeNode in="coloredBlur"/>
                                    <feMergeNode in="SourceGraphic"/>
                                </feMerge>
                            </filter>
                        </defs>

                        <rect x="5" y="3" width="14" height="18" rx="4"
                              fill="#0984e3" stroke="#a5d8ff" stroke-width="1.3"/>

                        <path d="M7 8 C 7 6, 17 6, 17 8 L16 11 L8 11 Z"
                              fill="#f1f2f6" opacity="0.92"/>

                        <rect x="8" y="15" width="8" height="1.5" rx="0.5"
                              fill="#ff7675" opacity="0.8"/>

                        <circle cx="8" cy="3.4" r="1.8"
                                fill="#7bed9f" filter="url(#headlightGlow)"/>

                        <circle cx="16" cy="3.4" r="1.8"
                                fill="#7bed9f" filter="url(#headlightGlow)"/>

                        <ellipse cx="8" cy="1.2" rx="2.7" ry="1.1"
                                 fill="#7bed9f" opacity="0.35"/>

                        <ellipse cx="16" cy="1.2" rx="2.7" ry="1.1"
                                 fill="#7bed9f" opacity="0.35"/>

                        <rect x="3" y="5" width="2" height="4" rx="1" fill="#1e272e"/>
                        <rect x="19" y="5" width="2" height="4" rx="1" fill="#1e272e"/>
                        <rect x="3" y="15" width="2" height="4" rx="1" fill="#1e272e"/>
                        <rect x="19" y="15" width="2" height="4" rx="1" fill="#1e272e"/>
                    </svg>
                </div>
                """

            html += f"<td class='{cell_class}'>{content}</td>"

        html += "</tr>"

    html += "</table></div>"

    return html


# --- 6. GIAO DIỆN CHÍNH (wide layout) ---
st.title("⚙️ HỆ THỐNG GIÁM SÁT & ĐỊNH TỰ ĐỘNG AGV")
st.markdown("---")

# Phân chia 3 cột: Cột 1 (Grid), Cột 2 (Search Tree), Cột 3 (Status/Controls)
col_grid, col_tree, col_status = st.columns([3, 4, 2], gap="medium")

with col_grid:
    st.subheader("Bản Đồ Số Lưới Không Gian")
    grid_placeholder = st.empty()
    with grid_placeholder:
        components.html(render_grid(), height=420, scrolling=False)

with col_tree:
    st.subheader("🌳 Cây Tìm Kiếm A* (Search Tree)")
    # Placeholder để vẽ lại cây liên tục
    tree_placeholder = st.empty() 
    
    # Nếu đã finished, hiện cây cuối cùng lên
    if st.session_state.mode in ["FINISHED", "MANUAL"] and st.session_state.final_tree_dot:
        tree_placeholder.graphviz_chart(st.session_state.final_tree_dot)


with col_status:
    st.subheader("Trạng Thái Hệ Thống")

    st.metric(label="CHẾ ĐỘ HOẠT ĐỘNG", value=st.session_state.mode)
    st.metric(label="TỔNG CHI PHÍ TÍCH LŨY", value=st.session_state.cost)

    st.write(f"**Hướng Vector Xe:** {DIR_NAMES[st.session_state.car_d]}")

    st.write("---")

    c_btn1, c_btn2 = st.columns(2)

    if c_btn1.button("🤖 KÍCH HOẠT AUTO A*"):
        st.session_state.mode = "AI SEARCHING..."
        st.session_state.explored_cells = set()
        st.session_state.final_path_cells = set()
        # Reset cây
        st.session_state.final_tree_dot = None 
        
        # 1. Khởi tạo Graphviz tạm trong lúc tìm kiếm
        dot_live = graphviz.Digraph()
        dot_live.attr('node', style='filled', color='#34414c', fontcolor='white', shape='circle')
        dot_live.attr('edge', color='white', fontcolor='white')
        
        # Nút S ban đầu
        s_h = heuristic(START_POS[0], START_POS[1], GOAL_POS[0], GOAL_POS[1])
        s_label = f"S ({START_POS[0]},{START_POS[1]})\ng=0, h={s_h}\nf={s_h}"
        dot_live.node(f"{START_POS[0]}-{START_POS[1]}-{START_POS[2]}", label=s_label, color='#0d3b32', fontcolor='#00b894', penwidth='2')

        # Chạy thuật toán để lấy dữ liệu
        path, exploration_order_full = solve_astar_with_vis()

        # PHA 1: AI quét thám thính & Vẽ cây mọc dần
        for r, c, d, node_id, parent_id, g, f, h, is_goal in exploration_order_full:
            # 1. Cập nhật cây tìm kiếm Graphviz
            if is_goal:
                goal_label = f"G ({r},{c})\nD:{d}\ng={g}, h={h}\nf={f}"
                dot_live.node(node_id, label=goal_label, color='#4c1c24', fontcolor='#ff7675', penwidth='2')
            else:
                child_label = f"({r},{c})\nD:{d}\ng={g}, h={h}\nf={f}"
                dot_live.node(node_id, label=child_label)
            
            # Nối cạnh nếu không phải root
            if parent_id != node_id:
                dot_live.edge(parent_id, node_id)
            
            # Cập nhật placeholder cho cây mọc
            with tree_placeholder:
                tree_placeholder.graphviz_chart(dot_live)

            # 2. Cập nhật Grid
            time.sleep(0.04) # Chỉnh tốc độ AI quét
            st.session_state.explored_cells.add((r, c))
            with grid_placeholder:
                components.html(render_grid(), height=420, scrolling=False)

        # Lưu lại cây graphviz hoàn chỉnh vào state để nó không mất khi rerun
        st.session_state.final_tree_dot = dot_live.source

        # PHA 2: Chốt đường đi tối ưu
        st.session_state.mode = "PATH LOCKED"
        for node in path:
            st.session_state.final_path_cells.add((node[0], node[1]))

        with grid_placeholder:
            components.html(render_grid(), height=420, scrolling=False)

        time.sleep(0.6)

        # PHA 3: Cho xe di chuyển thực tế
        st.session_state.mode = "AGV MOVING"
        current_g = 0

        for idx in range(1, len(path)):
            time.sleep(0.3)

            prev_r, prev_c, prev_d = path[idx - 1]
            r, c, d = path[idx]

            if (r, c) != (prev_r, prev_c):
                dr, dc = DIRS[prev_d]
                if prev_r + dr == r and prev_c + dc == c:
                    current_g += COST_FORWARD
                else:
                    current_g += COST_BACKWARD

            if d != prev_d:
                current_g += COST_TURN

            st.session_state.car_r = r
            st.session_state.car_c = c
            st.session_state.car_d = d
            st.session_state.cost = current_g

            with grid_placeholder:
                components.html(render_grid(), height=420, scrolling=False)

        st.session_state.mode = "FINISHED"
        st.success("AGV đã cập bến đỗ an toàn!")
        rerun_page()

    if c_btn2.button("🔄 RESET MÔ PHỎNG"):
        st.session_state.car_r = START_POS[0]
        st.session_state.car_c = START_POS[1]
        st.session_state.car_d = START_POS[2]
        st.session_state.cost = 0
        st.session_state.mode = "MANUAL"
        st.session_state.explored_cells = set()
        st.session_state.final_path_cells = set()
        # Reset cây
        st.session_state.final_tree_dot = None 
        rerun_page()

    if st.session_state.mode == "MANUAL" and (st.session_state.car_r, st.session_state.car_c) != GOAL_POS:
        st.write("**🕹️ Lái Thủ Công:**")

        dr, dc = DIRS[st.session_state.car_d]

        col_m1, col_m2 = st.columns(2)

        if col_m1.button("🔼 TIẾN"):
            nr = st.session_state.car_r + dr
            nc = st.session_state.car_c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
                st.session_state.car_r = nr
                st.session_state.car_c = nc
                st.session_state.cost += COST_FORWARD
                rerun_page()

        if col_m2.button("🔽 LÙI"):
            nr = st.session_state.car_r - dr
            nc = st.session_state.car_c - dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
                st.session_state.car_r = nr
                st.session_state.car_c = nc
                st.session_state.cost += COST_BACKWARD
                rerun_page()

        col_m3, col_m4 = st.columns(2)

        if col_m3.button("↩️ XOAY TRÁI"):
            st.session_state.car_d = (st.session_state.car_d - 1) % 4
            st.session_state.cost += COST_TURN
            rerun_page()

        if col_m4.button("↪️ XOAY PHẢI"):
            st.session_state.car_d = (st.session_state.car_d + 1) % 4
            st.session_state.cost += COST_TURN
            rerun_page()
