import pygame
import sys
import heapq
import time

# --- CẤU HÌNH HỆ THỐNG ---
WIDTH, HEIGHT = 600, 700
GRID_SIZE = 6 
CELL_SIZE = 100
FPS = 30

# Màu sắc
WHITE = (245, 245, 245)
BLACK = (30, 30, 30)
GRAY = (120, 120, 120)       # Vật cản X
GREEN = (46, 204, 113)       # Điểm xuất phát S
RED = (231, 76, 60)          # Điểm đích G
BLUE = (52, 152, 219)        # Đường đi của xe
YELLOW = (241, 196, 15)      # Thân xe màu vàng


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_NAMES = ["Bắc", "Đông", "Nam", "Tây"]

GRID = [
    [0, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0]
]

START_POS = (0, 0, 2) 
GOAL_POS = (5, 5)    


COST_FORWARD = 1
COST_TURN = 2
COST_BACKWARD = 3

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("LIVE DEMO: BÃI ĐỖ XE THÔNG MINH - AGV")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 22, bold=True)


def heuristic(r, c, d, g_r, g_c):
   
    manhattan = abs(r - g_r) + abs(c - g_c)
    return manhattan * COST_FORWARD

def solve_astar():
  
    start_r, start_c, start_d = START_POS
    g_goal, r_goal = GOAL_POS
    
    pq = [(heuristic(start_r, start_c, start_d, g_goal, r_goal), 0, start_r, start_c, start_d, [(start_r, start_c)])]
    visited = set()
    
    while pq:
        f, g, r, c, d, path = heapq.heappop(pq)
        
        if (r, c) == GOAL_POS:
            return path, g
            
        if (r, c, d) in visited:
            continue
        visited.add((r, c, d))
        
       
        dr, dc = DIRS[d]
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
            heapq.heappush(pq, (g + COST_FORWARD + heuristic(nr, nc, d, g_goal, r_goal), g + COST_FORWARD, nr, nc, d, path + [(nr, nc)]))
            
      
        nr, nc = r - dr, c - dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
            heapq.heappush(pq, (g + COST_BACKWARD + heuristic(nr, nc, d, g_goal, r_goal), g + COST_BACKWARD, nr, nc, d, path + [(nr, nc)]))
            
   
        for next_d in [(d - 1) % 4, (d + 1) % 4]:
            heapq.heappush(pq, (g + COST_TURN + heuristic(r, c, next_d, g_goal, r_goal), g + COST_TURN, r, c, next_d, path))
            
    return [], 0


def draw_interface(car_r, car_c, car_d, mode, cost, auto_path=[]):
    screen.fill((255, 255, 255))
    
  
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            
            if GRID[r][c] == 1:
                pygame.draw.rect(screen, GRAY, rect) 
            else:
                pygame.draw.rect(screen, BLACK, rect, 1) 
                
          
            if (r, c) == (START_POS[0], START_POS[1]):
                pygame.draw.rect(screen, GREEN, rect, 4)
            elif (r, c) == GOAL_POS:
                pygame.draw.rect(screen, RED, rect)
                
 
    if mode == "AUTO" and len(auto_path) > 1:
        for i in range(len(auto_path) - 1):
            p1 = auto_path[i]
            p2 = auto_path[i+1]
            pygame.draw.line(screen, BLUE, (p1[1]*CELL_SIZE + 50, p1[0]*CELL_SIZE + 50), (p2[1]*CELL_SIZE + 50, p2[0]*CELL_SIZE + 50), 5)

    
    car_center = (car_c * CELL_SIZE + 50, car_r * CELL_SIZE + 50)
    pygame.draw.circle(screen, YELLOW, car_center, 30)
  
    dr, dc = DIRS[car_d]
    arrow_end = (car_center[0] + dc * 25, car_center[1] + dr * 25)
    pygame.draw.line(screen, BLACK, car_center, arrow_end, 5)


    panel_y = GRID_SIZE * CELL_SIZE + 15
    mode_text = font.render(f"CHẾ ĐỘ: {mode}", True, RED if mode=="MANUAL" else BLUE)
    cost_text = font.render(f"TỔNG CHI PHÍ: {cost}", True, BLACK)
    dir_text = font.render(f"HƯỚNG XE: {DIR_NAMES[car_d]}", True, BLACK)
    guide_text = font.render("[M]: Tự lái | [A]: Chạy Auto A* | [R]: Reset", True, GRAY)
    
    screen.blit(mode_text, (20, panel_y))
    screen.blit(cost_text, (250, panel_y))
    screen.blit(dir_text, (20, panel_y + 30))
    screen.blit(guide_text, (20, panel_y + 60))
    
    pygame.display.flip()

def main():
    car_r, car_c, car_d = START_POS
    mode = "MANUAL" 
    cost = 0
    auto_path = []
    auto_idx = 0
    
    running = True
    while running:
        clock.tick(FPS)
        
     
        if mode == "AUTO" and auto_path and auto_idx < len(auto_path):
            time.sleep(0.3)
            next_r, next_c = auto_path[auto_idx]
        
            if (next_r, next_c) != (car_r, car_c):
                dr, dc = DIRS[car_d]
                if car_r + dr == next_r and car_c + dc == next_c:
                    cost += COST_FORWARD
                else:
                    cost += COST_BACKWARD
                car_r, car_c = next_r, next_c
            auto_idx += 1
            
        # Kiểm tra nút bấm 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # RESET
                    car_r, car_c, car_d = START_POS
                    cost = 0
                    auto_path = []
                    mode = "MANUAL"
                elif event.key == pygame.K_m: # Chuyển MANUAL
                    mode = "MANUAL"
                    car_r, car_c, car_d = START_POS
                    cost = 0
                elif event.key == pygame.K_a: # Chuyển AUTO
                    mode = "AUTO"
                    car_r, car_c, car_d = START_POS
                   
                    auto_path, total_astar_cost = solve_astar()
                    cost = 0
                    auto_idx = 0
                    
                # Điều khiển xe bằng tay 
                elif mode == "MANUAL" and (car_r, car_c) != GOAL_POS:
                    dr, dc = DIRS[car_d]
                    if event.key == pygame.K_UP: 
                        nr, nc = car_r + dr, car_c + dc
                        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
                            car_r, car_c = nr, nc
                            cost += COST_FORWARD
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN: 
                        nr, nc = car_r - dr, car_c - dc
                        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and GRID[nr][nc] == 0:
                            car_r, car_c = nr, nc
                            cost += COST_BACKWARD
                    elif event.key == pygame.K_LEFT: 
                        car_d = (car_d - 1) % 4
                        cost += COST_TURN
                    elif event.key == pygame.K_RIGHT: 
                        car_d = (car_d + 1) % 4
                        cost += COST_TURN

        draw_interface(car_r, car_c, car_d, mode, cost, auto_path)
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
