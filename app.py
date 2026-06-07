import streamlit as st
import heapq
import time

# --- 1. LỆNH NÀY PHẢI ĐẶT LÊN ĐẦU TIÊN ĐỂ TRÁNH LỖI ĐỎ ---
st.set_page_config(page_title="AI AGV 6x6 Maze", layout="centered")

# Hàm bổ trợ giúp chạy mượt trên cả Streamlit bản cũ lẫn mới
def rerun_page():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# --- 2. CẤU HÌNH MAP 6X6 "BẪY CỰC KHÓ" CHO AI MÒ ĐƯỜNG ---
GRID_SIZE = 6
GRID = [
    [0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0]
]

START_POS = (0, 0, 2)  # (Dòng 0, Cột 0, Hướng Nam 🔽)
GOAL_POS = (5, 5)      # Đích G ở góc dưới cùng bên phải (5,5)

COST_FORWARD = 1
COST_TURN = 2
COST_BACKWARD = 3

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)] # 0: Bắc, 1: Đông, 2: Nam, 3: Tây
DIR_NAMES = ["Bắc 🔼", "Đông ▶️", "Nam 🔽", "Tây ◀️"]
DIR_ROTATION = {0: -90, 1: 0, 2: 90, 3: 180} # Góc xoay CSS cho xe 🚗

# --- 3. KHỞI TẠO BỘ NHỚ TRẠNG THÁI (SESSION STATE) ---
if "car_r" not in st.session_state:
    st.session_state.car_r = START_POS[0]
    st.session_state.car_c = START_POS[1]
    st.session_state.car_d = START_POS[2]
    st.session_state.cost = 0
    st.session_state.mode = "MANUAL"
    st.session_state.explored_cells = set()
    st.session_state.final_path_cells = set()

# --- 4. THUẬT TOÁN A* QUÉT MAP VÀ LƯU QUÁ TRÌNH MÒ ĐƯỜNG ---
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
        exploration_order.append((r, c)) # Ghi nhận ô AI đang lùng sục thông tin
        
        # 1. TIẾN
        dr, dc = DIRS[d]
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
            heapq.heappush(pq, (g + COST_FORWARD + heuristic(nr, nc, g_goal, r_goal), g + COST_FORWARD, nr, nc, d, path + [(nr, nc, d)]))
            
        # 2. LÙI
        nr, nc = r - dr, c - dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
            heapq.heappush(pq, (g + COST_BACKWARD + heuristic(nr, nc, g_goal, r_goal), g + COST_BACKWARD, nr, nc, d, path + [(nr, nc, d)]))
            
        # 3. RẼ (Trái/Phải)
        for next_d in [(d - 1) % 4, (d + 1) % 4]:
            heapq.heappush(pq, (g + COST_TURN + heuristic(r, c, g_goal, r_goal), g + COST_TURN, r, c, next_d, path + [(r, c, next_d)]))
            
    return [], exploration_order

# --- 5. HÀM VẼ GIAO DIỆN LƯỚI HTML/CSS TỰ ĐỘNG XOAY XE ---
def render_grid():
    html = """
    <style>
        .grid-table { border-collapse: collapse; margin: auto; background-color: #f8f9fa; }
        .grid-cell { width: 65px; height: 65px; text-align: center; font-size: 22px; font-weight: bold; border: 2px solid #bdc3c7; position: relative; }
        .car-icon { font-size: 32px; display: inline-block; transition: transform 0.2s ease-in-out; }
    </style>
    <table class='grid-table'>
    """
    for r in range(GRID_SIZE):
        html += "<tr>"
        for c in range(GRID_SIZE):
            cell_style = "background-color: #ffffff;"
            content = ""
            
            # Đổ màu nền dựa trên trạng thái tìm kiếm
            if GRID[r][c] == 1:
                cell_style = "background-color: #2c3e50;" # Vật cản 
                content = "🧱"
            elif r == START_POS[0] and c == START_POS[1]:
                cell_style = "background-color: #2ecc71; color: white;" # Xuất phát
                content = "S"
            elif r == GOAL_POS[0] and c == GOAL_POS[1]:
                cell_style = "background-color: #e74c3c; color: white;" # Đích
                content = "🏁"
            elif (r, c) in st.session_state.final_path_cells:
                cell_style = "background-color: #dff9fb;" # Đường chốt tối ưu cuối cùng
                content = "🔹"
            elif (r, c) in st.session_state.explored_cells:
                cell_style = "background-color: #ffeaa7;" # Vùng AI đã sục sạo quét qua
            
            # Chèn mô hình xe lên trên cùng nếu trùng tọa độ hiện tại
            if r == st.session_state.car_r and c == st.session_state.car_c:
                angle = DIR_ROTATION[st.session_state.car_d]
                content = f"<div class='car-icon' style='transform: rotate({angle}deg);'>🚗</div>"
                
            html += f"<td class='grid-cell' style='{cell_style}'>{content}</td>"
        html += "</tr>"
    html += "</table>"
    return html

# --- 6. GIAO DIỆN WEB ---
st.title("🚗 LIVE DEMO ")
st.markdown("---")

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Bản Đồ Lưới Bãi Đỗ")
    grid_placeholder = st.empty()
    grid_placeholder.markdown(render_grid(), unsafe_allow_html=True)

with col2:
    st.subheader("Bảng Thông Tin")
    st.metric(label="CHẾ ĐỘ HỆ THỐNG", value=st.session_state.mode)
    st.metric(label="TỔNG CHI PHÍ (c)", value=st.session_state.cost)
    st.write(f"**Hướng xe:** {DIR_NAMES[st.session_state.car_d]}")
    
    st.write("---")
    c_btn1, c_btn2 = st.columns(2)
    
    if c_btn1.button("🤖 CHẠY AUTO A*"):
        st.session_state.mode = "AI SEARCHING..."
        st.session_state.explored_cells = set()
        st.session_state.final_path_cells = set()
        
        path, exploration_order = solve_astar_with_vis()
        
        # GIAI ĐOẠN 1: AI đi mò đường thực tế (Hiển thị các ô màu vàng loang dần)
        for cell in exploration_order:
            time.sleep(0.08) 
            st.session_state.explored_cells.add(cell)
            grid_placeholder.markdown(render_grid(), unsafe_allow_html=True)
            
        # GIAI ĐOẠN 2: Vẽ đường đi tối ưu tìm được (Các dấu chấm xanh biển)
        st.session_state.mode = "PATH FOUND!"
        for node in path:
            st.session_state.final_path_cells.add((node[0], node[1]))
        grid_placeholder.markdown(render_grid(), unsafe_allow_html=True)
        time.sleep(0.6)
        
        # GIAI ĐOẠN 3: Cho xe 🚗 chuyển động chạy bon bon theo lộ trình
        st.session_state.mode = "AGV MOVING..."
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
            
        st.success("AI đã đưa xe cập bến an toàn!")

    if c_btn2.button("🔄 RESET MAP"):
        st.session_state.car_r = START_POS[0]
        st.session_state.car_c = START_POS[1]
        st.session_state.car_d = START_POS[2]
        st.session_state.cost = 0
        st.session_state.mode = "MANUAL"
        st.session_state.explored_cells = set()
        st.session_state.final_path_cells = set()
        rerun_page()

    # Điều khiển bằng tay (Manual Mode)
    if st.session_state.mode == "MANUAL" and (st.session_state.car_r, st.session_state.car_c) != GOAL_POS:
        st.write("**🕹️ Trình Tự Lái Thủ Công:**")
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
        if col_m3.button("↩️ RẼ TRÁI"):
            st.session_state.car_d = (st.session_state.car_d - 1) % 4
            st.session_state.cost += COST_TURN
            rerun_page()
        if col_m4.button("↪️ RẼ PHẢI"):
            st.session_state.car_d = (st.session_state.car_d + 1) % 4
            st.session_state.cost += COST_TURN
            rerun_page()

    if (st.session_state.car_r, st.session_state.car_c) == GOAL_POS and st.session_state.mode == "MANUAL":
        st.balloons()
        st.success(f"Bạn đã tự lái xe về bến! Chi phí: {st.session_state.cost}")
