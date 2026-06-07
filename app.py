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

# --- 5. RENDER SÂN ĐỖ NÂNG CẤP ĐÈN PHÁT SÁNG & KHUNG TRẮNG ---
def render_grid():
    html = """
    <style>
        .grid-container {
            display: flex;
            justify-content: center;
            background-color: #12181b; 
            padding: 20px;
            border-radius: 14px;
            /* Khung viền màu trắng bao bọc ngoài bãi đỗ xe tạo sự khác biệt rõ rệt */
            border: 3px solid #ffffff; 
            box-shadow: 0 8px 24px rgba(255,255,255,0.1);
        }
        .grid-table { 
            border-collapse: separate; 
            border-spacing: 6px; 
            table-layout: fixed; 
            width: 420px; 
            height: 420px; 
        }
        .grid-cell { 
            width: 62px !important; 
            height: 62px !important; 
            box-sizing: border-box;
            text-align: center; 
            vertical-align: middle;
            font-family: 'Segoe UI', sans-serif;
            font-size: 12px; 
            font-weight: bold; 
            border-radius: 8px;
            position: relative; 
            background-color: #1e252b; 
            border: 1px dashed #34414c;
            overflow: hidden;
        }
        /* Vật cản phẳng tối */
        .cell-obstacle {
            background-color: #3d4d59 !important;
            border: 1px solid #4f616f !important;
        }
        /* Điểm xuất phát */
        .cell-start {
            background-color: #0d3b32 !important;
            color: #00b894 !important;
            border: 1px solid #00b894 !important;
        }
        /* Điểm đích đến */
        .cell-goal {
            background-color: #4c1c24 !important;
            color: #ff7675 !important;
            border: 1px solid #ff7675 !important;
        }
        /* AI quét đa hướng tìm đường (Màu cyan mờ ảo) */
        .cell-explored {
            background-color: #103136 !important;
            border: 1px dashed #00cec9 !important;
        }
        /* ĐƯỜNG ĐI AUTO ĐƯỢC HIGHLIGHT SIÊU RÕ NÉT VỚI HIỆU ỨNG PHÁT SÁNG NGỌC CAO CẤP */
        .cell-path {
            background-color: #00b894 !important;
            border: 1px solid #55efc4 !important;
            box-shadow: inset 0 0 15px rgba(85, 239, 196, 0.6) !important;
        }
        .path-dot {
            width: 14px;
            height: 14px;
            background-color: #ffffff;
            border-radius: 50%;
            margin: auto;
            box-shadow: 0 0 12px #ffffff;
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
            
            # Đè mô hình AGV lên ô hiện tại + Cấu hình hệ thống đèn LED Neon phát sáng cực đại
            if r == st.session_state.car_r and c == st.session_state.car_c:
                angle = DIR_ROTATION[st.session_state.car_d]
                content = f"""
                <div class='car-wrapper'>
                    <svg width="46" height="46" viewBox="0 0 24 24" style="transform: rotate({angle}deg); overflow: visible;">
                        <rect x="5" y="3" width="14" height="18" rx="4" fill="#0984e3" stroke="#74b9ff" stroke-width="1.2"/>
                        <path d="M7 8 C 7 6, 17 6, 17 8 L16 11 L8 11 Z" fill="#dfe6e9" opacity="0.85"/>
                        <rect x="8" y="15" width="8" height="1.5" rx="0.5" fill="#ff7675" opacity="0.6"/>
                        
                        <circle cx="8" cy="3.5" r="1.5" fill="#55efc4" filter="drop-shadow(0px -2px 5px #55efc4)"/>
                        <circle cx="16" cy="3.5" r="1.5" fill="#55efc4" filter="drop-shadow(0px -2px 5px #55efc4)"/>
                        
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

# --- 6. GIAO DIỆN CHÍNH ---
st.title("⚙️ HỆ THỐNG GIÁM SÁT & ĐỊNH TỰ ĐỘNG AGV")
st.markdown("---")

col1, col2 = st.columns([11, 8])

with col1:
    st.subheader("Bản Đồ Số Lưới Không Gian")
    grid_placeholder = st.empty()
    grid_placeholder.markdown(render_grid(), unsafe_allow_html=True)

with col2:
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
        
        path, exploration_order = solve_astar_with_vis()
        
        # PHA 1: AI chạy đủ hướng để tìm (Hiển thị các ô đã quét qua)
        for cell in exploration_order:
            time.sleep(0.05)
            st.session_state.explored_cells.add(cell)
            grid_placeholder.markdown(render_grid(), unsafe_allow_html=True)
            
        # PHA 2: Chốt đường đi chính xác và Highlight phát sáng rực rỡ
        st.session_state.mode = "PATH LOCKED"
        for node in path:
            st.session_state.final_path_cells.add((node[0], node[1]))
        grid_placeholder.markdown(render_grid(), unsafe_allow_html=True)
        time.sleep(0.6)
        
        # PHA 3: Cho xe chạy tịnh tiến theo lộ trình
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
            st.session_state.cost += COST_TURN
            rerun_page()
