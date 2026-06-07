import streamlit as st
import heapq
import time

# --- 1. CẤU HÌNH GIAO DIỆN ĐẦU FILE ---
st.set_page_config(page_title="Hệ Thống AGV Bãi Đỗ", layout="centered")

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

START_POS = (0, 0, 2)  # Xuất phát (0,0), xe hướng Nam
GOAL_POS = (5, 5)      # Đích đỗ (5,5)

COST_FORWARD = 1
COST_TURN = 2
COST_BACKWARD = 3

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)] 
DIR_NAMES = ["Bắc 🔼", "Đông ▶️", "Nam 🔽", "Tây ◀️"]
DIR_ROTATION = {0: 0, 1: 90, 2: 180, 3: 270}

# --- 3. KHỞI TẠO TRẠNG THÁI ---
if "car_r" not in st.session_state:
    st.session_state.car_r = START_POS[0]
    st.session_state.car_c = START_POS[1]
    st.session_state.car_d = START_POS[2]
    st.session_state.cost = 0
    st.session_state.mode = "MANUAL"
    st.session_state.explored_cells = set()
    st.session_state.final_path_cells = set()

# --- 4. THUẬT TOÁN A* LÕI ---
def heuristic(r, c, g_r, g_c):
    return (abs(r - g_r) + abs(c - g_c)) * COST_FORWARD

def solve_astar_with_vis():
    start_r, start_c, start_d = START_POS
    g_goal, r_goal = GOAL_POS
    
    pq = [(heuristic(start_r, start_c, g_goal, r_goal), 0, start_r, start_c, start_d, [(start_r, start_c, start_d)])]
    visited = set()
    exploration_order = []
    
    while pq:
        f, g, r, c, d, path = heapq.heappop(pq)
        
        if (r, c) == GOAL_POS:
            return path, exploration_order
            
        if (r, c, d) in visited:
            continue
        visited.add((r, c, d))
        exploration_order.append((r, c))
        
        # Tiến
        dr, dc = DIRS[d]
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
            heapq.heappush(pq, (g + COST_FORWARD + heuristic(nr, nc, g_goal, r_goal), g + COST_FORWARD, nr, nc, d, path + [(nr, nc, d)]))
            
        # Lùi
        nr, nc = r - dr, c - dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
            heapq.heappush(pq, (g + COST_BACKWARD + heuristic(nr, nc, g_goal, r_goal), g + COST_BACKWARD, nr, nc, d, path + [(nr, nc, d)]))
            
        # Rẽ
        for next_d in [(d - 1) % 4, (d + 1) % 4]:
            heapq.heappush(pq, (g + COST_TURN + heuristic(r, c, g_goal, r_goal), g + COST_TURN, r, c, next_d, path + [(r, c, next_d)]))
            
    return [], exploration_order

# --- 5. RENDER SÂN ĐỖ CAO CẤP CHỐNG ZOOM GIẬT ---
def render_grid():
    html = """
    <style>
        .grid-container {
            display: flex;
            justify-content: center;
            background-color: #263238; /* Nền sàn hầm tối hiện đại */
            padding: 20px;
            border-radius: 12px;
        }
        /* Khóa cứng bố cục bảng, chống giãn dòng, giãn ô khi phần tử chuyển động */
        .grid-table { 
            border-collapse: separate; 
            border-spacing: 6px; 
            table-layout: fixed; 
            width: 430px; 
            height: 430px; 
        }
        .grid-cell { 
            width: 65px !important; 
            height: 65px !important; 
            box-sizing: border-box;
            text-align: center; 
            vertical-align: middle;
            font-family: 'Segoe UI', sans-serif;
            font-size: 13px; 
            font-weight: bold; 
            border-radius: 6px;
            position: relative; 
            /* Thiết kế ô đỗ xe tiêu chuẩn vạch đứt */
            background-color: #cfd8dc; 
            border: 2px dashed #ffffff;
            overflow: hidden;
        }
        /* Chướng ngại vật: Khối bê tông dải phản quang vàng đen chuyên nghiệp */
        .cell-obstacle {
            background-color: #37474f !important;
            border: 2px solid #1c2833 !important;
            background-image: linear-gradient(135deg, #f1c40f 25%, transparent 25%, transparent 50%, #f1c40f 50%, #f1c40f 75%, transparent 75%, transparent) !important;
            background-size: 20px 20px !important;
            opacity: 0.85;
        }
        /* Vị trí xuất phát */
        .cell-start {
            background-color: #2ecc71 !important;
            color: white !important;
            border: 2px solid #27ae60 !important;
        }
        /* Vị trí ô đích đỗ xe */
        .cell-goal {
            background-color: #e74c3c !important;
            color: white !important;
            border: 2px solid #c0392b !important;
        }
        /* Các ô đang quét tìm đường (Màu radar nhẹ) */
        .cell-explored {
            background-color: #ffeaa7 !important;
            border: 2px dashed #f1c40f !important;
        }
        /* Nút chấm xanh của lộ trình chốt */
        .cell-path {
            background-color: #b3e5fc !important;
            border: 2px dashed #0288d1 !important;
        }
        .path-dot {
            width: 12px;
            height: 12px;
            background-color: #0288d1;
            border-radius: 50%;
            margin: auto;
        }
        /* Khung bọc xe chống tràn kích thước */
        .car-wrapper {
            width: 100%;
            height: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            position: absolute;
            top: 0;
            left: 0;
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
                content = "START"
            elif r == GOAL_POS[0] and c == GOAL_POS[1]:
                cell_class += " cell-goal"
                content = "GOAL"
            elif (r, c) in st.session_state.final_path_cells:
                cell_class += " cell-path"
                content = "<div class='path-dot'></div>"
            elif (r, c) in st.session_state.explored_cells:
                cell_class += " cell-explored"
            
            # Đè xe AGV lên ô hiện tại (Giữ nguyên cấu trúc vector chuẩn hướng Bắc)
            if r == st.session_state.car_r and c == st.session_state.car_c:
                angle = DIR_ROTATION[st.session_state.car_d]
                content = f"""
                <div class='car-wrapper'>
                    <svg width="46" height="46" viewBox="0 0 24 24" style="transform: rotate({angle}deg); overflow: visible;">
                        <rect x="5" y="3" width="14" height="18" rx="4" fill="#0984e3" stroke="#130cb7" stroke-width="1.5"/>
                        <path d="M7 8 C 7 6, 17 6, 17 8 L16 11 L8 11 Z" fill="#eccc68" opacity="0.9"/>
                        <rect x="8" y="15" width="8" height="2" rx="0.5" fill="#ffffff" opacity="0.4"/>
                        <circle cx="8" cy="4.5" r="1" fill="#fff"/>
                        <circle cx="16" cy="4.5" r="1" fill="#fff"/>
                        <rect x="3" y="5" width="2" height="4" rx="1" fill="#2d3436"/>
                        <rect x="19" y="5" width="2" height="4" rx="1" fill="#2d3436"/>
                        <rect x="3" y="15" width="2" height="4" rx="1" fill="#2d3436"/>
                        <rect x="19" y="15" width="2" height="4" rx="1" fill="#2d3436"/>
                    </svg>
                </div>
                """
            html += f"<td class='{cell_class}'>{content}</td>"
        html += "</tr>"
    html += "</table></div>"
    return html

# --- 6. GIAO DIỆN CHÍNH ---
st.title("⚙️ HỆ THỐNG ĐIỀU HƯỚNG AGV THƯƠNG MẠI (A* 6x6)")
st.markdown("---")

col1, col2 = st.columns([11, 8])

with col1:
    st.subheader("Mô Hình Lưới Sân Bãi")
    grid_placeholder = st.empty()
    grid_placeholder.markdown(render_grid(), unsafe_allow_html=True)

with col2:
    st.subheader("Trạng Thái")
    st.metric(label="CHẾ ĐỘ", value=st.session_state.mode)
    st.metric(label="CHI PHÍ TÍCH LŨY (c)", value=st.session_state.cost)
    st.write(f"**Hướng xe:** {DIR_NAMES[st.session_state.car_d]}")
    
    st.write("---")
    c_btn1, c_btn2 = st.columns(2)
    
    if c_btn1.button("🤖 CHẠY AUTO A*"):
        st.session_state.mode = "AI SEARCHING..."
        st.session_state.explored_cells = set()
        st.session_state.final_path_cells = set()
        
        path, exploration_order = solve_astar_with_vis()
        
        for cell in exploration_order:
            time.sleep(0.06)
            st.session_state.explored_cells.add(cell)
            grid_placeholder.markdown(render_grid(), unsafe_allow_html=True)
            
        st.session_state.mode = "PATH LOCKED"
        for node in path:
            st.session_state.final_path_cells.add((node[0], node[1]))
        grid_placeholder.markdown(render_grid(), unsafe_allow_html=True)
        time.sleep(0.6)
        
        st.session_state.mode = "AGV MOVING"
        current_g = 0
        for idx in range(1, len(path)):
            time.sleep(0.35)
            prev_r, prev_c, prev_d = path[idx-1]
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
            grid_placeholder.markdown(render_grid(), unsafe_allow_html=True)
            
        st.success("AGV đã cập bến đỗ an toàn!")

    if c_btn2.button("🔄 RESET MÔ PHỎNG"):
        st.session_state.car_r = START_POS[0]
        st.session_state.car_c = START_POS[1]
        st.session_state.car_d = START_POS[2]
        st.session_state.cost = 0
        st.session_state.mode = "MANUAL"
        st.session_state.explored_cells = set()
        st.session_state.final_path_cells = set()
        rerun_page()

    if st.session_state.mode == "MANUAL" and (st.session_state.car_r, st.session_state.car_c) != GOAL_POS:
        st.write("**🕹️ Lái Thủ Công:**")
        dr, dc = DIRS[st.session_state.car_d]
        
        col_m1, col_m2 = st.columns(2)
        if col_m1.button("🔼 TIẾN"):
            nr, nc = st.session_state.car_r + dr, st.session_state.car_c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
                st.session_state.car_r, st.session_state.car_c = nr, nc
                st.session_state.cost += COST_FORWARD
                rerun_page()
        if col_m2.button("🔽 LÙI"):
            nr, nc = st.session_state.car_r - dr, st.session_state.car_c - dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
                st.session_state.car_r, st.session_state.car_c = nr, nc
                st.session_state.cost += COST_BACKWARD
                rerun_page()
                
        col_m3, col_m4 = st.columns(2)
        if col_m3.button("↩️ XOAY TRÁI"):
            st.session_state.car_d = (st.session_state.car_d - 1) % 4
            st.session_state.cost += COST_TURN
            rerun_page()
        if col_m4.button("↪️ XOAY PHẢI"):
            st.session_state.car_d = (st.session_state.car_d + 1) % 4
            st.session_width = COST_TURN
            st.session_state.cost += COST_TURN
            rerun_page()
