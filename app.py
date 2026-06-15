import streamlit as st
import streamlit.components.v1 as components
import heapq
import time
import graphviz 

# --- 1. CẤU HÌNH GIAO DIỆN HỆ THỐNG ---
st.set_page_config(page_title="HỆ THỐNG GIÁM SÁT ĐỊNH XE AGV ", layout="wide")

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
    st.session_state.run_mode = "THỦ CÔNG" # Mặc định ban đầu

if "snapshots" not in st.session_state:
    st.session_state.snapshots = []
    st.session_state.step_index = 0
    st.session_state.final_path = []
    
    # State cho chế độ Thủ công
    st.session_state.car_r = START_POS[0]
    st.session_state.car_c = START_POS[1]
    st.session_state.car_d = START_POS[2]
    st.session_state.cost = 0
    st.session_state.explored_cells = set([(START_POS[0], START_POS[1])])


# --- 4. THUẬT TOÁN A* SINH THUẬT TOÁN ĐỒNG BỘ ---
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

    # Cấu hình Graphviz kích thước LỚN, màu sắc tương phản mạnh để nhìn từ xa
    dot = graphviz.Digraph(comment='Large Search Tree')
    dot.attr(rankdir='TB', size='8,8!', ratio='fill', bgcolor='transparent') 
    dot.attr('node', style='filled,rounded', shape='box', 
             fillcolor='#f1f2f6', fontcolor='#2d3436', color='#2d3436',
             fontname='Arial Bold', fontsize='13', penwidth='2.5')
    dot.attr('edge', color='#2d3436', arrowsize='0.8', penwidth='2.0')
    
    # Nút START màu Xanh Lá Đậm - Chữ Trắng tương phản cao
    dot.node(start_node_id, 
             label=f"START ({start_r},{start_c}) {DIR_ARROWS[start_d]}\nf={h_start}\n[g=0, h={h_start}]", 
             fillcolor='#2ed573', fontcolor='#ffffff', color='#26af5f')

    while pq:
        f, g, r, c, d, path, parent_id = heapq.heappop(pq)
        current_state = (r, c, d)
        
        if current_state in visited and visited[current_state] <= g:
            continue
            
        visited[current_state] = g
        current_node_id = f"{r}-{c}-{d}"
        explored_cells_acc.add((r, c))

        h_val = heuristic(r, c, r_goal, c_goal)
        
        # Gặp đích đỗ xe (GOAL) - Nút màu Đỏ Đậm nổi bần bật
        if (r, c) == GOAL_POS:
            dot.node(current_node_id, 
                     label=f"GOAL ({r},{c}) {DIR_ARROWS[d]}\nf={g}\n[g={g}, h=0]", 
                     fillcolor='#ff4757', fontcolor='#ffffff', color='#ff2e44')
            if parent_id != current_node_id:
                dot.edge(parent_id, current_node_id)
                
            snapshots_list.append({
                "explored": set(explored_cells_acc),
                "car": (r, c, d),
                "cost": g,
                "tree": dot.source
            })
            return path, snapshots_list
        else:
            # Các nút trung gian đang duyệt: Đổ màu Vàng Chanh dễ nhìn từ dưới lớp
            if current_node_id != start_node_id:
                dot.node(current_node_id, 
                         label=f"({r},{c}) {DIR_ARROWS[d]}\nf={f}\n[g={g}, h={h_val}]",
                         fillcolor='#ffa502', fontcolor='#000000', color='#eccc68')
                if parent_id != current_node_id:
                    dot.edge(parent_id, current_node_id)

        snapshots_list.append({
            "explored": set(explored_cells_acc),
            "car": (r, c, d),
            "cost": g,
            "tree": dot.source
        })
        
        # Đã duyệt xong bước này, chuyển sang màu Xám Nhạt để nhường tiêu điểm cho bước tiếp theo
        if current_node_id != start_node_id:
            dot.node(current_node_id, 
                     label=f"({r},{c}) {DIR_ARROWS[d]}\nf={f}\n[g={g}, h={h_val}]",
                     fillcolor='#f1f2f6', fontcolor='#57606f', color='#ced6e0')

        # Phát triển các tập con
        dr, dc = DIRS[d]
        # 1. Đi Tiến
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
            heapq.heappush(pq, (g + COST_FORWARD + heuristic(nr, nc, r_goal, c_goal), g + COST_FORWARD, nr, nc, d, path + [(nr, nc, d)], current_node_id))

        # 2. Đi Lùi
        nr, nc = r - dr, c - dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
            heapq.heappush(pq, (g + COST_BACKWARD + heuristic(nr, nc, r_goal, c_goal), g + COST_BACKWARD, nr, nc, d, path + [(nr, nc, d)], current_node_id))

        # 3. Xoay Hướng (Trái / Phải)
        for next_d in [(d - 1) % 4, (d + 1) % 4]:
            heapq.heappush(pq, (g + COST_TURN + heuristic(r, c, r_goal, c_goal), g + COST_TURN, r, c, next_d, path + [(r, c, next_d)], current_node_id))

    return [], snapshots_list


# --- 5. RENDER KHUNG BẢN ĐỒ HTML ---
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


# --- 6. GIAO DIỆN ĐIỀU KHIỂN CHÍNH ---
st.title("📊 MÔ PHỎNG THUẬT TOÁN ĐỊNH TUYẾN AGV BÃI ĐỖ")

# Lựa chọn duy nhất giữa 2 chế độ độc lập
mode_select = st.selectbox("🎯 CHỌN CHẾ ĐỘ HOẠT ĐỘNG:", ["THỦ CÔNG", "TỰ ĐỘNG"], index=0 if st.session_state.run_mode == "THỦ CÔNG" else 1)

if mode_select != st.session_state.run_mode:
    st.session_state.run_mode = mode_select
    # Tự động tính toán cây nếu chuyển sang TỰ ĐỘNG
    if mode_select == "TỰ ĐỘNG" and not st.session_state.snapshots:
        path, snapshots = generate_search_snapshots()
        st.session_state.snapshots = snapshots
        st.session_state.final_path = path
        st.session_state.step_index = 0
    rerun_page()

st.markdown("---")

# CHIA BỐ CỤC 2 CỘT CÂN XỨNG ĐỂ THUYẾT TRÌNH
col_left, col_right = st.columns([4, 6], gap="large")

# Phân tách lấy dữ liệu trạng thái tương ứng theo chế độ được chọn
if st.session_state.run_mode == "TỰ ĐỘNG" and st.session_state.snapshots:
    current_snap = st.session_state.snapshots[st.session_state.step_index]
    explored_now = current_snap["explored"]
    car_now = current_snap["car"]
    cost_now = current_snap["cost"]
    tree_dot_now = current_snap["tree"]
    
    is_last = (st.session_state.step_index == len(st.session_state.snapshots) - 1)
    path_cells_now = set((n[0], n[1]) for n in st.session_state.final_path) if is_last else set()
else:
    # Dữ liệu của chế độ lái tay
    explored_now = st.session_state.explored_cells
    car_now = (st.session_state.car_r, st.session_state.car_c, st.session_state.car_d)
    cost_now = st.session_state.cost
    path_cells_now = set()
    tree_dot_now = None


with col_left:
    st.markdown("### 🗺️ Bản Đồ Sa Bàn Lưới")
    components.html(render_grid_dynamic(explored_now, car_now, path_cells_now), height=370, scrolling=False)
    
    st.markdown("#### 📝 Thông Số Thực Tế:")
    st.info(f"• **Vị trí xe:** Ô `({car_now[0]}, {car_now[1]})` — Hướng: **{DIR_NAMES[car_now[2]]}**\n"
            f"• **Chi phí lũy kế g(n):** `{cost_now}`")
    
    # KHU VỰC 1: ĐIỀU KHIỂN CHẾ ĐỘ THỦ CÔNG (CHỈ HIỆN KHI CHỌN THỦ CÔNG)
    if st.session_state.run_mode == "THỦ CÔNG":
        st.markdown("🕹️ **Cụm Phím Lái Xe Bằng Tay:**")
        cx1, cx2 = st.columns(2)
        if cx1.button("🔼 ĐI TIẾN (+1)", use_container_width=True):
            dr, dc = DIRS[st.session_state.car_d]
            nr, nc = st.session_state.car_r + dr, st.session_state.car_c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
                st.session_state.car_r, st.session_state.car_c = nr, nc
                st.session_state.cost += COST_FORWARD
                st.session_state.explored_cells.add((nr, nc))
                rerun_page()
        if cx2.button("🔽 ĐI LÙI (+3)", use_container_width=True):
            dr, dc = DIRS[st.session_state.car_d]
            nr, nc = st.session_state.car_r - dr, st.session_state.car_c - dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
                st.session_state.car_r, st.session_state.car_c = nr, nc
                st.session_state.cost += COST_BACKWARD
                st.session_state.explored_cells.add((nr, nc))
                rerun_page()
        cx3, cx4 = st.columns(2)
        if cx3.button("🔄 XOAY TRÁI (+2)", use_container_width=True):
            st.session_state.car_d = (st.session_state.car_d - 1) % 4
            st.session_state.cost += COST_TURN
            rerun_page()
        if cx4.button("🔄 XOAY PHẢI (+2)", use_container_width=True):
            st.session_state.car_d = (st.session_state.car_d + 1) % 4
            st.session_state.cost += COST_TURN
            rerun_page()

    # KHU VỰC 2: BỘ ĐIỀU KHIỂN TỰ ĐỘNG (CHỈ HIỆN KHI CHỌN TỰ ĐỘNG)
    if st.session_state.run_mode == "TỰ ĐỘNG" and st.session_state.snapshots:
        st.markdown("🤖 **Bộ Điều Khiển Thuật Toán A*:**")
        total_steps = len(st.session_state.snapshots)
        
        st.write(f"Tiến độ duyệt cây: **Bước {st.session_state.step_index + 1} / {total_steps}**")
        
        c1, c2, c3 = st.columns(3)
        if c1.button("⬅️ LÙI 1 BƯỚC", use_container_width=True):
            if st.session_state.step_index > 0:
                st.session_state.step_index -= 1
                rerun_page()
        if c2.button("TIẾP 1 BƯỚC ➡️", use_container_width=True):
            if st.session_state.step_index < total_steps - 1:
                st.session_state.step_index += 1
                rerun_page()
        if c3.button("▶️ TỰ ĐỘNG CHẠY BƯỚC", use_container_width=True):
            for idx in range(st.session_state.step_index, total_steps):
                st.session_state.step_index = idx
                time.sleep(0.3) # Giãn thời gian ra 0.3s để người xem kịp đối chiếu xe và cây
                rerun_page()

    st.markdown("---")
    if st.button("🔄 RESET MÔ PHỎNG", use_container_width=True):
        st.session_state.snapshots = []
        st.session_state.step_index = 0
        st.session_state.final_path = []
        st.session_state.car_r = START_POS[0]
        st.session_state.car_c = START_POS[1]
        st.session_state.car_d = START_POS[2]
        st.session_state.cost = 0
        st.session_state.explored_cells = set([(START_POS[0], START_POS[1])])
        rerun_page()

with col_right:
    st.markdown("### 🌳 Sơ Đồ Cây Phát Triển Trạng Thái (Kích thước Lớn)")
    if tree_dot_now:
        # Render sơ đồ cây dạng to bản, chữ đậm, nét dầy
        st.graphviz_chart(tree_dot_now, use_container_width=True)
    else:
        st.info("Hệ thống đang ở chế độ lái xe thủ công nên chưa vẽ cây. Hãy chuyển sang chế độ 'TỰ ĐỘNG' ở thanh chọn phía trên để quan sát cây A* nhé!")
