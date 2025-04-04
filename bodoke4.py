import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
from heapq import heappush, heappop
from matplotlib.animation import FuncAnimation

def main():
    # -------------------------
    # 1) Parámetros del robot
    # -------------------------
    l1, l2, l3 = 0.5, 0.5, 0.5
    
    # Obstáculo: línea vertical (o diagonal) 
    pObs1 = np.array([0.1, 0.1])
    pObs2 = np.array([0.9,  0.1])
    
    # -------------------------
    # 2) Discretización del C-Space
    # -------------------------
    N = 50
    theta_vals = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # -------------------------
    # 3) Config inicial y objetivo (en índices) usando IK discreta
    # -------------------------
    start_indices = (0, 0, 0)
    goal_xy = np.array([1.2, 0.3])
    goal_indices = None
    best_dist = float('inf')
    for i in range(N):
        for j in range(N):
            for k in range(N):
                t1, t2, t3 = theta_vals[i], theta_vals[j], theta_vals[k]
                xef, yef = forward_kinematics_end_effector(t1, t2, t3, l1, l2, l3)
                dist = np.hypot(xef - goal_xy[0], yef - goal_xy[1])
                if dist < best_dist:
                    best_dist = dist
                    goal_indices = (i,j,k)
    
    print("Start:", start_indices)
    print("Goal indices (IK discreta):", goal_indices, " Distancia meta:", best_dist)
    
    # -------------------------
    # 4) Buscar colisiones en el C-Space para graficar
    # -------------------------
    coll_points = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                t1, t2, t3 = theta_vals[i], theta_vals[j], theta_vals[k]
                if check_collision_line(t1, t2, t3, l1, l2, l3, pObs1, pObs2):
                    coll_points.append((i, j, k))
    coll_points = np.array(coll_points)
    
    # -------------------------
    # 5) A* para encontrar ruta en (i,j,k)
    # -------------------------
    path = astar_cspace(start_indices, goal_indices, theta_vals, l1, l2, l3, pObs1, pObs2)
    
    # -------------------------
    # 6) Graficar C-Space
    # -------------------------
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_title('Espacio de Configuración con Colisiones (A*)')
    # Colisión en rojo
    if len(coll_points) > 0:
        ax1.scatter(coll_points[:,0], coll_points[:,1], coll_points[:,2],
                    c='r', s=5, label='Colisión')
    
    if path is not None:
        path_array = np.array(path)
        ax1.plot(path_array[:,0], path_array[:,1], path_array[:,2], 'b-o', label='Camino A*')
    ax1.set_xlabel('i (theta1)')
    ax1.set_ylabel('j (theta2)')
    ax1.set_zlabel('k (theta3)')
    ax1.legend()
    
    # -------------------------
    # 7) Animar en 2D (si hay camino)
    # -------------------------
    if path is None:
        print("No se encontró camino libre.")
    else:
        print("Camino encontrado! longitud:", len(path))
        
        fig2, ax2 = plt.subplots()
        ax2.set_aspect('equal', 'box')
        ax2.set_xlim([-0.5, 2.0])
        ax2.set_ylim([-1.0, 1.5])
        ax2.set_title('Animación 2D del Robot (A*)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        
        ax2.plot([pObs1[0], pObs2[0]], [pObs1[1], pObs2[1]], 'k-', lw=3, label='Obstáculo')
        ax2.plot(goal_xy[0], goal_xy[1], 'mx', markersize=8, label='Objetivo')
        ax2.legend()
        
        line1, = ax2.plot([], [], 'b-o', markersize=6)
        line2, = ax2.plot([], [], 'g-o', markersize=6)
        line3, = ax2.plot([], [], 'r-o', markersize=6)
        
        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            line3.set_data([], [])
            return line1, line2, line3
        
        def update(frame):
            i, j, k = path[frame]
            t1, t2, t3 = theta_vals[i], theta_vals[j], theta_vals[k]
            x0, y0 = 0.0, 0.0
            x1 = x0 + l1*np.cos(t1)
            y1 = y0 + l1*np.sin(t1)
            x2 = x1 + l2*np.cos(t1 + t2)
            y2 = y1 + l2*np.sin(t1 + t2)
            x3 = x2 + l3*np.cos(t1 + t2 + t3)
            y3 = y2 + l3*np.sin(t1 + t2 + t3)
            line1.set_data([x0,x1],[y0,y1])
            line2.set_data([x1,x2],[y1,y2])
            line3.set_data([x2,x3],[y2,y3])
            return line1, line2, line3
        
        ani = FuncAnimation(fig2, update, frames=len(path),
                            init_func=init, blit=True, interval=300)
    
    plt.show()

# ----------------------------------------------------------------
# A* en la rejilla (i,j,k)
def astar_cspace(start_idx, goal_idx, theta_vals, l1, l2, l3, pObs1, pObs2):
    N = len(theta_vals)
    visited = np.zeros((N,N,N), dtype=bool)
    parent = {}
    
    # Min-heap
    # Cada elemento: (f, g, (i,j,k))
    #   f = g + h  (g = coste real, h = heurística)
    #   g = coste desde start
    #   h = heurística (dist manhattan o euclidea en el "espacio de índices")
    #   (i,j,k) = nodo
    heap = []
    
    def heuristic(a, b):
        # Distancia euclídea en la rejilla
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)
    
    gscore = np.full((N,N,N), np.inf, dtype=float)
    gscore[start_idx] = 0.0
    
    hstart = heuristic(start_idx, goal_idx)
    fstart = gscore[start_idx] + hstart
    heappush(heap, (fstart, gscore[start_idx], start_idx))
    
    moves = []
    for di in [-1,0,1]:
        for dj in [-1,0,1]:
            for dk in [-1,0,1]:
                if not (di==0 and dj==0 and dk==0):
                    moves.append((di,dj,dk))
    
    while heap:
        fcurr, gcurr, current = heappop(heap)
        if visited[current]:
            continue
        
        visited[current] = True
        if current == goal_idx:
            # Reconstruir
            return reconstruct_path(current, parent)
        
        ci, cj, ck = current
        for (di,dj,dk) in moves:
            ni = (ci+di) % N
            nj = (cj+dj) % N
            nk = (ck+dk) % N
            if not visited[(ni,nj,nk)]:
                # colisión?
                t1, t2, t3 = theta_vals[ni], theta_vals[nj], theta_vals[nk]
                if not check_collision_line(t1,t2,t3,l1,l2,l3,pObs1,pObs2):
                    tentative_g = gcurr + 1.0  # coste unif
                    if tentative_g < gscore[ni,nj,nk]:
                        gscore[ni,nj,nk] = tentative_g
                        hval = heuristic((ni,nj,nk), goal_idx)
                        fval = tentative_g + hval
                        parent[(ni,nj,nk)] = current
                        heappush(heap, (fval, tentative_g, (ni,nj,nk)))
    
    return None

def reconstruct_path(end_idx, parent):
    path = []
    current = end_idx
    while current in parent:
        path.append(current)
        current = parent[current]
    path.append(current)
    path.reverse()
    return path

# ----------------------------------------------------------------
def forward_kinematics_end_effector(t1, t2, t3, l1, l2, l3):
    x0, y0 = 0.0, 0.0
    x1 = x0 + l1*np.cos(t1)
    y1 = y0 + l1*np.sin(t1)
    x2 = x1 + l2*np.cos(t1 + t2)
    y2 = y1 + l2*np.sin(t1 + t2)
    x3 = x2 + l3*np.cos(t1 + t2 + t3)
    y3 = y2 + l3*np.sin(t1 + t2 + t3)
    return x3, y3

def check_collision_line(t1,t2,t3, l1,l2,l3, pObs1,pObs2):
    x0, y0 = 0, 0
    x1 = x0 + l1*np.cos(t1)
    y1 = y0 + l1*np.sin(t1)
    x2 = x1 + l2*np.cos(t1+t2)
    y2 = y1 + l2*np.sin(t1+t2)
    x3 = x2 + l3*np.cos(t1+t2+t3)
    y3 = y2 + l3*np.sin(t1+t2+t3)
    if segments_intersect([x0,y0],[x1,y1], pObs1,pObs2):
        return True
    if segments_intersect([x1,y1],[x2,y2], pObs1,pObs2):
        return True
    if segments_intersect([x2,y2],[x3,y3], pObs1,pObs2):
        return True
    return False

def segments_intersect(p1, p2, p3, p4):
    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)
    
    if (o1 != o2) and (o3 != o4):
        return True
    if o1==0 and on_segment(p1,p3,p2): return True
    if o2==0 and on_segment(p1,p4,p2): return True
    if o3==0 and on_segment(p3,p1,p4): return True
    if o4==0 and on_segment(p3,p2,p4): return True
    return False

def orientation(a,b,c):
    val = (b[0] - a[0])*(c[1] - a[1]) - (b[1]-a[1])*(c[0]-a[0])
    if abs(val) < 1e-12: return 0
    elif val>0: return 1
    else: return 2

def on_segment(p,q,r):
    if (min(p[0],r[0])<=q[0]<=max(p[0],r[0]) and
        min(p[1],r[1])<=q[1]<=max(p[1],r[1])):
        return True
    return False


if __name__=="__main__":
    main()
