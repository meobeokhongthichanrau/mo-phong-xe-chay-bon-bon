import streamlit as st
import heapq
import time

# --- CẤU HÌNH MAP BÃI ĐỖ XE (Lưới 6x6 giống hệt ảnh demo của bạn) ---
GRID_SIZE = 6
GRID = [
    [0, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0]
]

START_POS = (0, 0, 2)  # (Dòng 0, Cột 0, Hướng Nam)
GOAL_POS = (5, 5)      # Đích G ở góc dưới phải

# Chi phí hành động theo báo cáo của nhóm bạn
COST_FORWARD = 1
COST_TURN = 2
COST_BACKWARD = 3

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)] # 0: Bắc, 1: Đông, 2: Nam, 3: Tây
DIR_NAMES = ["Bắc 🔼", "Đông ▶️", "Nam 🔽", "Tây ◀️"]
CAR_ARROWS = ["🔼", "▶️", "🔽", "◀️"]

# --- KHỞI TẠO BỘ NHỚ TRẠNG THÁI (SESSION STATE) ---
if "car_r" not in st.session_state:
    st.session_state.car_r = START_POS[0]
    st.session_state.car_c = START_POS[1]
    st.session_state.car_d = START_POS[2]
    st.session_state.cost = 0
    st.session_state.mode = "MANUAL"

# --- THUẬT TOÁN A* TÌM ĐƯỜNG ---
def heuristic(r, c, g_r, g_c):
    return (abs(r - g_r) + abs(c - g_c)) * COST_FORWARD

def solve_astar():
    start_r, start_c, start_d = START_POS
    g_goal, r_goal = GOAL_POS
    
    # Hàng đợi: (f_score, g_score, r, c, d, danh_sách_nút_đã_đi)
    pq = [(heuristic(start_r, start_c, g_goal, r_goal), 0, start_r, start_c, start_d, [(start_r, start_c, start_d)])]
    visited = set()
    
    while pq:
        f, g, r, c, d, path = heapq.heappop(pq)
        if (r, c) == GOAL_POS:
            return path
            
        if (r, c, d) in visited:
            continue
        visited.add((r, c, d))
        
        # 1. TIẾN
        dr, dc = DIRS[d]
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
            pq.append((g + COST_FORWARD + heuristic(nr, nc, g_goal, r_goal), g + COST_FORWARD, nr, nc, d, path + [(nr, nc, d)]))
            
        # 2. LÙI
        nr, nc = r - dr, c - dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
            pq.append((g + COST_BACKWARD + heuristic(nr, nc, g_goal, r_goal), g + COST_BACKWARD, nr, nc, d, path + [(nr, nc, d)]))
            
        # 3. RẼ (Trái/Phải)
        for next_d in [(d - 1) % 4, (d + 1) % 4]:
            pq.append((g + COST_TURN + heuristic(r, c, g_goal, r_goal), g + COST_TURN, r, c, next_d, path + [(r, c, next_d)]))
            
    return []

# --- HÀM VẼ BẢN ĐỒ BẰNG HTML CHO ĐẸP VÀ MƯỢT ---
def render_grid():
    html = """
    <style>
        .grid-table { border-collapse: collapse; margin: auto; }
        .grid-cell { width: 60px; height: 60px; text-align: center; font-size: 24px; font-weight: bold; border: 2px solid #333; }
    </style>
    <table class='grid-table'>
    """
    for r in range(GRID_SIZE):
        html += "<tr>"
        for c in range(GRID_SIZE):
            if r == st.session_state.car_r and c == st.session_state.car_c:
                # Ô chứa xe tự hành
                cell_style = "background-color: #f1c40f;" # Vàng
                content = CAR_ARROWS[st.session_state.car_d]
            elif GRID[r][c] == 1:
                # Vật cản X
                cell_style = "background-color: #7f8c8d; color: white;" # Xám
                content = "❌"
            elif r == START_POS[0] and c == START_POS[1]:
                # Vị trí xuất phát ban đầu
                cell_style = "background-color: #2ecc71;" # Xanh lá
                content = "S"
            elif r == GOAL_POS[0] and c == GOAL_POS[1]:
                # Đích đỗ xe
                cell_style = "background-color: #e74c3c; color: white;" # Đỏ
                content = "G"
            else:
                # Đường đi trống
                cell_style = "background-color: #ffffff;"
                content = ""
            html += f"<td class='grid-cell' style='{cell_style}'>{content}</td>"
        html += "</tr>"
    html += "</table>"
    return html

# --- GIAO DIỆN CHÍNH CỦA TRANG WEB STREAMLIT ---
st.set_page_config(page_title="AGV Routing Demo", layout="centered")
st.title("🚗 LIVE DEMO: BÃI ĐỖ XE THÔNG MINH (AGV)")
st.markdown("---")

# Chia giao diện làm 2 cột: Trái vẽ Map, Phải điều khiển
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Bản Đồ Lưới Mô Phỏng")
    grid_placeholder = st.empty()
    grid_placeholder.markdown(render_grid(), unsafe_allow_html=True)

with col2:
    st.subheader("Bảng Điều Khiển")
    st.metric(label="CHẾ ĐỘ HIỆN TẠI", value=st.session_state.mode)
    st.metric(label="TỔNG CHI PHÍ ($c$)", value=st.session_state.cost)
    st.write(f"**Hướng xe:** {DIR_NAMES[st.session_state.car_d]}")
    
    # Nút chọn chế độ hệ thống
    st.write("---")
    c_btn1, c_btn2 = st.columns(2)
    
    if c_btn1.button("🤖 CHẠY AUTO A*"):
        st.session_state.mode = "AUTONOMOUS (AI)"
        path = solve_astar()
        if path:
            # Chạy hiệu ứng xe di chuyển từng bước một trên web
            current_g = 0
            for idx in range(1, len(path)):
                time.sleep(0.4) # Độ trễ để thấy xe lăn bánh
                prev_r, prev_c, prev_d = path[idx-1]
                r, c, d = path[idx]
                
                # Tính toán chi phí thực tế cộng dồn
                if (r, c) != (prev_r, prev_c):
                    dr, dc = DIRS[prev_d]
                    if prev_r + dr == r and prev_c + dc == c:
                        current_g += COST_FORWARD
                    else:
                        current_g += COST_BACKWARD
                if d != prev_d:
                    current_g += COST_TURN
                    
                # Cập nhật tọa độ mới lên màn hình web
                st.session_state.car_r = r
                st.session_state.car_c = c
                st.session_state.car_d = d
                st.session_state.cost = current_g
                grid_placeholder.markdown(render_grid(), unsafe_allow_html=True)
            st.success("Xe đã về ô đỗ an toàn nhờ thuật toán A*!")
            
    if c_btn2.button("🔄 RESET MAP"):
        st.session_state.car_r = START_POS[0]
        st.session_state.car_c = START_POS[1]
        st.session_state.car_d = START_POS[2]
        st.session_state.cost = 0
        st.session_state.mode = "MANUAL"
        st.rerun
