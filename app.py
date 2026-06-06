import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import streamlit as st

# Mở rộng toàn trang
st.set_page_config(layout="wide")

# CSS: Xóa khoảng trắng thừa và ẩn logo Streamlit cho giống App Native
st.markdown("""
    <style>
           .block-container { padding-top: 1rem; padding-bottom: 0rem; }
           header { visibility: hidden; }
           footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

if 'state' not in st.session_state:
    st.session_state.state = {'x': 0, 'y': 0, 'dx': 1, 'dy': 0}

grid = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 3]
])
dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]

def draw_car(ax, x, y, dx, dy, color_body, color_window):
    angle_dict = {(1,0): 0, (0,1): 90, (-1,0): 180, (0,-1): 270}
    angle = angle_dict.get((dx, dy), 0)
    car_width, car_length = 0.5, 0.8
    car_x, car_y = x - car_length/2, y - car_width/2
    car_body = patches.Rectangle((car_x, car_y), car_length, car_width, color=color_body, zorder=5, ec='black', lw=1)
    windshield = patches.Rectangle((x + 0.1, y - 0.2), 0.15, 0.4, color=color_window, zorder=6)
    rear_window = patches.Rectangle((x - 0.35, y - 0.15), 0.1, 0.3, color=color_window, zorder=6)
    light1 = patches.Circle((x + 0.38, y - 0.15), 0.05, color='yellow', zorder=6)
    light2 = patches.Circle((x + 0.38, y + 0.15), 0.05, color='yellow', zorder=6)
    transform = plt.matplotlib.transforms.Affine2D().rotate_deg_around(x, y, angle) + ax.transData
    for patch in [car_body, windshield, rear_window, light1, light2]:
        patch.set_transform(transform)
        ax.add_patch(patch)

def move_car(action):
    x, y, dx, dy = st.session_state.state['x'], st.session_state.state['y'], st.session_state.state['dx'], st.session_state.state['dy']
    dir_idx = dirs.index((dx, dy))
    if action == 'up':
        nx, ny = x + dx, y + dy
        if 0 <= nx < 6 and 0 <= ny < 6 and grid[ny, nx] != 1:
            st.session_state.state['x'], st.session_state.state['y'] = nx, ny
    elif action == 'down':
        nx, ny = x - dx, y - dy
        if 0 <= nx < 6 and 0 <= ny < 6 and grid[ny, nx] != 1:
            st.session_state.state['x'], st.session_state.state['y'] = nx, ny
    elif action == 'left':
        st.session_state.state['dx'], st.session_state.state['dy'] = dirs[(dir_idx - 1) % 4]
    elif action == 'right':
        st.session_state.state['dx'], st.session_state.state['dy'] = dirs[(dir_idx + 1) % 4]

# --- CHIA BỐ CỤC NGANG 16:9 ---
col_map, col_controls = st.columns([2, 1])

# Bên trái: Bản đồ
with col_map:
    fig, ax = plt.subplots(figsize=(5.5, 5.5)) 
    ax.set_facecolor('#0E1621') # Đổi màu nền cho tiệp với slide xanh đen
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(5.5, -0.5)

    for i in range(6):
        for j in range(6):
            ax.add_patch(patches.Rectangle((i-0.45, j-0.45), 0.9, 0.9, fill=False, ec='#ffffff', lw=1.5, ls='--', alpha=0.2))

    goal_box = patches.Rectangle((4.5, 4.5), 1, 1, facecolor='#1a5276', ec='#3498db', lw=3, zorder=2)
    ax.add_patch(goal_box)
    ax.text(5, 5, 'P', color='white', fontsize=24, fontweight='bold', ha='center', va='center', zorder=3)

    for r in range(6):
        for c in range(6):
            if grid[r, c] == 1:
                draw_car(ax, c, r, 1, 0, color_body='#E74C3C', color_window='#1a1a1a') # Xe đỏ nổi hơn

    draw_car(ax, st.session_state.state['x'], st.session_state.state['y'], st.session_state.state['dx'], st.session_state.state['dy'], color_body='#ffffff', color_window='#333333')

    ax.set_xticks(np.arange(0, 6, 1))
    ax.set_yticks(np.arange(0, 6, 1))
    ax.xaxis.tick_top()
    st.pyplot(fig)

# Bên phải: Nút bấm
with col_controls:
    st.markdown("<br><br>", unsafe_allow_html=True) # Đẩy chữ xuống giữa
    if st.session_state.state['x'] == 5 and st.session_state.state['y'] == 5:
        st.success("🎉 ĐỖ XE THÀNH CÔNG!")
    else:
        st.markdown("<h3 style='text-align: center; color: white;'>🕹️ BẢNG ĐIỀU KHIỂN</h3>", unsafe_allow_html=True)

    # Layout nút hình chữ thập (D-pad)
    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c2:
        if st.button("Tiến ⬆️", use_container_width=True): move_car('up')
    
    c4, c5, c6 = st.columns([1.5, 1.2, 1.5])
    with c4:
        if st.button("Trái ↩️", use_container_width=True): move_car('left')
    with c5:
        if st.button("Lùi ⬇️", use_container_width=True): move_car('down')
    with c6:
        if st.button("Phải ↪️", use_container_width=True): move_car('right')
