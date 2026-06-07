import streamlit as st
import heapq
import time

# --- 1. CẤU HÌNH GIAO DIỆN ĐẦU FILE (BẮT BUỘC) ---
st.set_page_config(page_title="Hệ Thống Định Tuyến AGV Bãi Đỗ", layout="centered")

def rerun_page():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# --- 2. CẤU HÌNH MA TRẬN BÃI ĐỖ 6X6 NÂNG CAO ---
GRID_SIZE = 6
GRID = [
    [0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0]
]

START_POS = (0, 0, 2)  # Xuất phát tại (0,0), xe hướng về phía Nam
GOAL_POS = (5, 5)      # Đích đỗ xe tại ô (5,5)

COST_FORWARD = 1
COST_TURN = 2
COST_BACKWARD = 3

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)] # 0: Bắc, 1: Đông, 2: Nam, 3: Tây
DIR_NAMES = ["Bắc 🔼", "Đông ▶️", "Nam 🔽", "Tây ◀️"]

# Khớp góc xoay SVG chuẩn: Mặc định đầu xe hướng lên Bắc (0 deg)
DIR_ROTATION = {0: 0, 1: 90, 2: 180, 3: 270}

# --- 3. KHỞI TẠO TRẠNG THÁI HỆ THỐNG ---
if "car_r" not in st.session_state:
    st.session_state.car_r = START_POS[0]
    st.session_state.car_c = START_POS[1]
    st.session_state.car_d = START_POS[2]
    st.session_state.cost = 0
    st.session_state.mode = "MANUAL"
    st.session_state.explored_cells = set()
    st.session_state.final_path_cells = set()

# --- 4. THUẬT TOÁN ĐỊNH TUYẾN LÕI A* ---
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
        
        # 1. TIẾN
        dr, dc = DIRS[d]
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
            heapq.heappush(pq, (g + COST_FORWARD + heuristic(nr, nc, g_goal, r_goal), g + COST_FORWARD, nr, nc, d, path + [(nr, nc, d)]))
            
        # 2. LÙI
        nr, nc = r - dr, c - dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
            heapq.heappush(pq, (g + COST_BACKWARD + heuristic(nr, nc, g_goal, r_goal), g + COST_BACKWARD, nr, nc, d, path + [(nr, nc, d)]))
            
        # 3. RẼ TRÁI / PHẢI
        for next_d in [(d - 1) % 4, (d + 1) % 4]:
            heapq.heappush(pq, (g + COST_TURN + heuristic(r, c, g_goal, r_goal), g + COST_TURN, r, c, next_d, path + [(r, c, next_d)]))
            
    return [], exploration_order

# --- 5. HÀM RENDER ĐỒ HỌA CHUYÊN NGHIỆP (KHÔNG DÙNG EMOJI) ---
def render_grid():
    html = """
    <style>
        .grid-container {
            display: flex;
            justify-content: center;
            background-color: #1e272e;
            padding: 18px;
            border-radius: 14px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        }
        .grid-table { border-collapse: separate; border-spacing: 5px; }
        .grid-cell { 
            width: 65px; 
            height: 65px; 
            text-align: center; 
            vertical-align: middle;
            font-family: 'Segoe UI', Roboto, sans-serif;
            font-size: 16px; 
            font-weight: 700; 
            border-radius: 8px;
            position: relative; 
            background-color: #ffffff;
            box-shadow: inset 0 0 0 1px #e2e8f0;
            transition: background-color 0.15s ease;
        }
        /* Thiết kế vùng cấm/vật cản sọc phản quang công nghiệp */
        .cell-obstacle {
            background: repeating-linear-gradient(45deg, #34495e, #34495e 8px, #2c3e50 8px, #2c3e50 16px) !important;
            box-shadow: none !important;
        }
        .cell-start {
            background-color: #2ecc71 !important;
            color: white !important;
            box-shadow: 0 4px 12px rgba(46, 204, 113, 0.4) !important;
        }
        .cell-goal {
            background-color: #e74c3c !important;
            color: white !important;
            box-shadow: 0 4px 12px rgba(231, 76, 60, 0.4) !important;
        }
        /* Ô hiển thị quét quét mò đường (Màu vàng radar công nghiệp nhẹ) */
        .cell-explored {
            background-color: #ffeaa7 !important;
            box-shadow: inset 0 0 0 2px #f1c40f !important;
        }
        /* Lộ trình tối ưu chốt cuối */
        .cell-path {
            background-color: #e3f2fd !important;
        }
        .path-dot {
            width: 14px;
            height: 14px;
            background-color: #3498db;
            border-radius: 50%;
            margin: auto;
            box-shadow: 0 0 8px #3498db;
        }
        .car-wrapper {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            height: 100%;
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
            
            # Khởi tạo mô hình AGV vector sắc nét đè lên tọa độ hiện tại
            if r == st.session_state.car_r and c == st.session_state.car_c:
                angle = DIR_ROTATION[st.session_state.car_d]
                content = f"""
                <div class='car-wrapper'>
                    <svg width="48" height="48" viewBox="0 0 24 24" style="transform: rotate({angle}deg); transition: transform 0.2s ease;">
                        <rect x="5" y="3" width="14" height="18" rx="4" fill="#0984e3" stroke="#2980b9" stroke-width="1.5"/>
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

# --- 6. GIAO DIỆN DIỄN HOẠ THUYẾT TRÌNH ---
st.title("⚙️ HỆ THỐNG GIÁM SÁT VÀ ĐỊNH TỰ ĐỘNG AGV (A* 6x6)")
st.markdown("---")

col1, col2 = st.columns([11, 8])

with col1:
    st.subheader("Bản Đồ Số Lưới Không Gian")
    grid_placeholder = st.empty()
    grid_placeholder.markdown(render_grid(), unsafe_allow_html=True)

with col2:
    st.subheader("Trạng Thái Hệ Thống")
    st.metric(label="CHẾ ĐỘ HOẠT ĐỘNG", value=st.session_state.mode)
    st.metric(label="TỔNG CHI PHÍ TÍCH LŨY (c)", value=st.session_state.cost)
    st.write(f"**Hướng Vector Xe:** {DIR_NAMES[st.session_state.car_d]}")
    
    st.write("---")
    c_btn1, c_btn2 = st.columns(2)
    
    if c_btn1.button("🤖 KÍCH HOẠT AUTO A*"):
        st.session_state.mode = "AI SEARCHING..."
        st.session_state.explored_cells = set()
        st.session_state.final_path_cells = set()
        
        path, exploration_order = solve_astar_with_vis()
        
        # PHA 1: Minh họa quá trình rà quét mò đường lắt léo
        for cell in exploration_order:
            time.sleep(0.07)
            st.session_state.explored_cells.add(cell)
            grid_placeholder.markdown(render_grid(), unsafe_allow_html=True)
            
        # PHA 2: Xác nhận và hiển thị chuỗi nút đường đi ngắn nhất
        st.session_state.mode = "PATH LOCKED"
        for node in path:
            st.session_state.final_path_cells.add((node[0], node[1]))
        grid_placeholder.markdown(render_grid(), unsafe_allow_html=True)
        time.sleep(0.6)
        
        # PHA 3: Điều khiển mô hình xe dịch chuyển tịnh tiến thực tế
        st.session_state.mode = "AGV EXECUTING"
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
            
        st.success("Điều hướng thành công! Xe đã cập bến an toàn.")

    if c_btn2.button("🔄 RESET MÔ PHỎNG"):
        st.session_state.car_r = START_POS[0]
        st.session_state.car_c = START_POS[1]
        st.session_state.car_d = START_POS[2]
        st.session_state.cost = 0
        st.session_state.mode = "MANUAL"
        st.session_state.explored_cells = set()
        st.session_state.final_path_cells = set()
        rerun_page()

    # Bảng điều hướng thủ công để test mô hình
    if st.session_state.mode == "MANUAL" and (st.session_state.car_r, st.session_state.car_c) != GOAL_POS:
        st.write("**🕹️ Bảng Điều Khiển Thủ Công:**")
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

    if (st.session_state.car_r, st.session_state.car_c) == GOAL_POS and st.session_state.mode == "MANUAL":
        st.balloons()
        st.success(f"Hoàn thành điều khiển thủ công! Tổng chi phí hành trình: {st.session_state.cost}")
