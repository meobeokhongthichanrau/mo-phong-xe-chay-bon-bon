import streamlit as st
import streamlit.components.v1 as components
import heapq
import time
import graphviz 

# --- 1. CẤU HÌNH GIAO DIỆN HỆ THỐNG ---
st.set_page_config(page_title="Hệ Thống AGV Bãi Đỗ & Step-by-Step Tree", layout="wide")

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
DIR_ARROWS = ["🔼", "▶️", "🔽", "◀️"]
DIR_ROTATION = {0: 0, 1: 90, 2: 180, 3: 270}


# --- 3. KHỞI TẠO CÁC STATE QUẢN LÝ TỪNG BƯỚC ---
if "snapshots" not in st.session_state:
    st.session_state.snapshots = []       # Lưu lại lịch sử từng bước duyệt để tua
    st.session_state.step_index = 0       # Bước hiện tại người dùng đang xem
    st.session_state.final_path = []      # Đường đi chốt cuối cùng
    st.session_state.car_r = START_POS[0]
    st.session_state.car_c = START_POS[1]
    st.session_state.car_d = START_POS[2]
    st.session_state.cost = 0
    st.session_state.mode = "MANUAL"


# --- 4. THUẬT TOÁN A* SINH SNAPSHOTS ---
def heuristic(r, c, g_r, g_c):
    return (abs(r - g_r) + abs(c - g_c)) * COST_FORWARD

def generate_search_snapshots():
    start_r, start_c, start_d = START_POS
    r_goal, c_goal = GOAL_POS

    start_node_id = f"{start_r}-{start_c}-{start_d}"
    pq = [(heuristic(start_r, start_c, r_goal, c_goal), 0, start_r, start_c, start_d, [(start_r, start_c, start_d)], start_node_id)]
    
    visited = {}
    snapshots_list = []
    explored_cells_acc = set()

    # Khởi tạo cây ban đầu
    dot = graphviz.Digraph(comment='Search Tree')
    dot.attr('node', style='filled', color='#e1f5fe', fontcolor='#0288d1', shape='ellipse', fontname='Arial', fontsize='11')
    dot.attr('edge', color='#2c3e50', arrowsize='0.6', penwidth='1.2')
    
    s_h = heuristic(start_r, start_c, r_goal, c_goal)
    dot.node(start_node_id, label=f"S:({start_r},{start_c})\n{DIR_ARROWS[start_d]}\nf={s_h}", color='#0d3b32', fontcolor='#00b894')

    while pq:
        f, g, r, c, d, path, parent_id = heapq.heappop(pq)
        current_state = (r, c, d)
        
        if current_state in visited and visited[current_state] <= g:
            continue
        
        visited[current_state] = g
        current_node_id = f"{r}-{c}-{d}"
        explored_cells_acc.add((r, c))

        # Cập nhật node hiện tại vào cây đồ thị
        if (r, c) == GOAL_POS:
            dot.node(current_node_id, label=f"G:({r},{c})\n{DIR_ARROWS[d]}\ng={g}", color='#4c1c24', fontcolor='#ff7675')
            if parent_id != current_node_id:
                dot.edge(parent_id, current_node_id)
            
            # Lưu snapshot cuối cùng tại Đích và thoát
            snapshots_list.append({
                "explored": set(explored_cells_acc),
                "car": (r, c, d),
                "cost": g,
                "tree": dot.source
            })
            return path, snapshots_list
        else:
            if current_node_id != start_node_id:
                dot.node(current_node_id, label=f"({r},{c})\n{DIR_ARROWS[d]}\nf={f}")
                if parent_id != current_node_id:
                    dot.edge(parent_id, current_node_id)

        # Lưu trạng thái của bước này lại
        snapshots_list.append({
            "explored": set(explored_cells_acc),
            "car": (r, c, d),
            "cost": g,
            "tree": dot.source
        })

        # Kế thừa phát triển các nhánh con
        # 1. Tiến
        dr, dc = DIRS[d]
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
            heapq.heappush(pq, (g + COST_FORWARD + heuristic(nr, nc, r_goal, c_goal), g + COST_FORWARD, nr, nc, d, path + [(nr, nc, d)], current_node_id))

        # 2. Lùi
        nr, nc = r - dr, c - dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
            heapq.heappush(pq, (g + COST_BACKWARD + heuristic(nr, nc, r_goal, c_goal), g + COST_BACKWARD, nr, nc, d, path + [(nr, nc, d)], current_node_id))

        # 3. Xoay
        for next_d in [(d - 1) % 4, (d + 1) % 4]:
            heapq.heappush(pq, (g + COST_TURN + heuristic(r, c, r_goal, c_goal), g + COST_TURN, r, c, next_d, path + [(r, c, next_d)], current_node_id))

    return [], snapshots_list


# --- 5. RENDER SÂN ĐỖ HTML ---
def render_grid_dynamic(explored, car_pos, final_path_cells):
    car_r, car_c, car_d = car_pos
    html = """
    <style>
        body { background-color: transparent; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; overflow: hidden; }
        .grid-container { display: flex; justify-content: center; align-items: center; background-color: #12181b; padding: 12px; border-radius: 14px; border: 3px solid #ffffff; box-shadow: 0 0 22px rgba(255, 255, 255, 0.28); width: 370px; max-width: 100%; margin: 0 auto; box-sizing: border-box; }
        .grid-table { border-collapse: separate; border-spacing: 5px; table-layout: fixed; width: 340px; height: 340px; }
        .grid-cell { width: 50px !important; height: 50px !important; box-sizing: border-box; text-align: center; vertical-align: middle; font-family: 'Segoe UI', sans-serif; font-size: 14px; font-weight: bold; border-radius: 8px; position: relative; background-color: #1e252b; border: 1px dashed #34414c; color: #ffffff; }
        .cell-obstacle { background-color: #3d4d59 !important; border: 1px solid #4f616f !important; }
        .cell-start { background-color: #0d3b32 !important; color: #00b894 !important; border: 1px solid #00b894 !important; }
        .cell-goal { background-color: #4c1c24 !important; color: #ff7675 !important; border: 1px solid #ff7675 !important; }
        .cell-explored { background-color: #103136 !important; border: 1px dashed #00cec9 !important; }
        .cell-path { background-color: #00b894 !important; border: 1px solid #55efc4 !important; box-shadow: inset 0 0 18px rgba(85, 239, 196, 0.85) !important; }
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
                        <rect x="5" y="3" width="14" height="18" rx="4" fill="#0984e3" stroke="#a5d8ff" stroke-width="1.3"/>
                        <path d="M7 8 C 7 6, 17 6, 17 8 L16 11 L8 11 Z" fill="#f1f2f6" opacity="0.92"/>
                        <circle cx="8" cy="3.4" r="1.8" fill="#7bed9f"/>
                        <circle cx="16" cy="3.4" r="1.8" fill="#7bed9f"/>
                    </svg>
                </div>
                """
            html += f"<td class='{cell_class}'>{content}</td>"
        html += "</tr>"
    html += "</table></div>"
    return html


# --- 6. GIAO DIỆN CHÍNH ---
st.subheader("⚙️ HỆ THỐNG MÔ PHỎNG PHÁT TRIỂN NHÁNH CÂY TỪNG BƯỚC (A*)")

# Tạo vùng tương tác thuật toán trước
col_ctrl1, col_ctrl2 = st.columns([3, 7])

with col_ctrl1:
    if st.button("🤖 1. TÍNH TOÁN TOÀN BỘ CÂY A*", use_container_width=True):
        path, snapshots = generate_search_snapshots()
        st.session_state.snapshots = snapshots
        st.session_state.final_path = path
        st.session_state.step_index = 0
        st.session_state.mode = "STEP_VIEW"
        st.success(f"Đã giải xong! Tổng cộng có {len(snapshots)} bước duyệt cây.")

# Nếu đang ở chế độ xem từng bước, hiện thanh điều khiển dòng thời gian
if st.session_state.mode == "STEP_VIEW" and st.session_state.snapshots:
    total_steps = len(st.session_state.snapshots)
    
    with col_ctrl2:
        # Thanh trượt chọn bước
        step_idx = st.slider("Kéo để xem dòng thời gian cây phát triển:", 0, total_steps - 1, st.session_state.step_index, key="slider_step")
        st.session_state.step_index = step_idx
        
        # Cụm nút bấm thủ công tinh chỉnh
        c1, c2, c3 = st.columns(3)
        if c1.button("⬅️ BƯỚC TRƯỚC"):
            if st.session_state.step_index > 0:
                st.session_state.step_index -= 1
                rerun_page()
        if c2.button("BƯỚC TIẾP ➡️"):
            if st.session_state.step_index < total_steps - 1:
                st.session_state.step_index += 1
                rerun_page()
        if c3.button("▶️ CHẠY AUTO"):
            # Chạy tự động từ bước hiện tại đến hết
            for idx in range(st.session_state.step_index, total_steps):
                st.session_state.step_index = idx
                time.sleep(0.15)
                rerun_page()

st.markdown("---")

# LAYOUT HIỂN THỊ CHÍNH (Chia 2 cột cân xứng)
col_left, col_right = st.columns([4, 6], gap="large")

# Lấy dữ liệu snapshot hiện tại
if st.session_state.snapshots:
    current_snap = st.session_state.snapshots[st.session_state.step_index]
    explored_now = current_snap["explored"]
    car_now = current_snap["car"]
    cost_now = current_snap["cost"]
    tree_dot_now = current_snap["tree"]
    
    # Nếu xem tới bước cuối cùng thì hiện đường đi chốt hạ
    is_last_step = (st.session_state.step_index == len(st.session_state.snapshots) - 1)
    path_cells_now = set((n[0], n[1]) for n in st.session_state.final_path) if is_last_step else set()
else:
    explored_now = st.session_state.explored_cells
    car_now = (st.session_state.car_r, st.session_state.car_c, st.session_state.car_d)
    cost_now = st.session_state.cost
    path_cells_now = st.session_state.final_path_cells
    tree_dot_now = None

with col_left:
    st.markdown("### 🗺️ Bản Đồ Lưới Không Gian")
    components.html(render_grid_dynamic(explored_now, car_now, path_cells_now), height=380, scrolling=False)
    
    # Bảng số liệu trạng thái ngay bên dưới bản đồ
    st.markdown("#### Đọc Chỉ Số Hiện Tại:")
    st.write(f"• **Bước tìm kiếm thứ:** {st.session_state.step_index + 1 if st.session_state.snapshots else 0}")
    st.write(f"• **Tọa độ AGV đang xét nhánh:** `({car_now[0]}, {car_now[1]})` - Hướng: {DIR_NAMES[car_now[2]]}")
    st.write(f"• **Chi phí thực tế g(n):** `{cost_now}`")
    
    if st.button("🔄 RESET MÔ PHỎNG", use_container_width=True):
        st.session_state.snapshots = []
        st.session_state.step_index = 0
        st.session_state.final_path = []
        st.session_state.car_r = START_POS[0]
        st.session_state.car_c = START_POS[1]
        st.session_state.car_d = START_POS[2]
        st.session_state.cost = 0
        st.session_state.mode = "MANUAL"
        rerun_page()

with col_right:
    st.markdown("### 🌳 Cây Trạng Thái Duyệt Thuật Toán")
    if tree_dot_now:
        # Hiển thị biểu đồ cây với các mũi tên chỉ rõ nét mối liên kết cha con
        st.graphviz_chart(tree_dot_now)
    else:
        st.info("Hãy nhấn nút 'TÍNH TOÁN TOÀN BỘ CÂY A*' ở trên để khởi tạo mô hình nhánh cây.")
