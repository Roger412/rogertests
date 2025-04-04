import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import sin, cos

def main():
    # ----- Parámetros del robot -----
    l1, l2, l3 = 0.2, 0.2, 0.2
    
    # ----- Obstáculo: línea diagonal -----
    pObs1 = np.array([0.1, 0.1])
    pObs2 = np.array([0.12, 0.12])
    
    # ----- Configuración inicial -----
    # por ejemplo: (t1=0, t2=0, t3=0)
    thetas = np.array([0.0, 0.0, 0.0])
    
    # ----- Posición objetivo -----
    goal_xy = np.array([0.3, 0.3])
    
    # ----- Resolvemos IK numérica para llegar a goal_xy -----
    # (De forma muy simple y sin verificación de colisiones intermedias)
    thetas_final = inverse_kinematics_numerical(thetas, goal_xy, l1, l2, l3)
    
    # ----- Generamos una trayectoria lineal en el espacio articular -----
    n_steps = 50
    path_thetas = np.linspace(thetas, thetas_final, n_steps)
    
    # ----- Animación del robot y verificación de colisión en cada frame -----
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    ax.set_xlim([-0.1, 1.6])
    ax.set_ylim([-0.1, 1.2])
    
    ax.plot([pObs1[0], pObs2[0]], [pObs1[1], pObs2[1]], 'k-', lw=3, label='Obstáculo')
    ax.plot(goal_xy[0], goal_xy[1], 'mx', markersize=8, label='Objetivo')
    ax.legend()
    
    line1, = ax.plot([], [], 'b-o', markersize=6)
    line2, = ax.plot([], [], 'g-o', markersize=6)
    line3, = ax.plot([], [], 'r-o', markersize=6)
    collision_text = ax.text(0.05, 1.05, "", transform=ax.transAxes, fontsize=12, color='red')
    
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        collision_text.set_text("")
        return line1, line2, line3, collision_text
    
    def update(frame):
        t1, t2, t3 = path_thetas[frame]
        
        # Cinemática directa
        x0, y0 = 0.0, 0.0
        x1 = x0 + l1*cos(t1)
        y1 = y0 + l1*sin(t1)
        x2 = x1 + l2*cos(t1 + t2)
        y2 = y1 + l2*sin(t1 + t2)
        x3 = x2 + l3*cos(t1 + t2 + t3)
        y3 = y2 + l3*sin(t1 + t2 + t3)
        
        line1.set_data([x0, x1], [y0, y1])
        line2.set_data([x1, x2], [y1, y2])
        line3.set_data([x2, x3], [y2, y3])
        
        # Checar colisión en cada frame
        if check_collision_line(x0,y0, x1,y1, x2,y2, x3,y3, pObs1, pObs2):
            collision_text.set_text("¡Colisión!")
        else:
            collision_text.set_text("")
        
        return line1, line2, line3, collision_text
    
    ani = FuncAnimation(fig, update, frames=n_steps, init_func=init,
                        blit=True, interval=300)
    
    plt.show()

# ----------------------------------------------------------------
# Inverse kinematics (numérica) muy simplificada para un RRR 2D
def inverse_kinematics_numerical(thetas_init, goal_xy, l1, l2, l3, lr=0.1, max_iter=200):
    # thetas_init: [t1, t2, t3] inicial
    # goal_xy: [xg, yg] posición deseada
    # lr: learning rate (paso de gradiente)
    # max_iter: número máximo de iteraciones
    thetas = thetas_init.copy()
    
    for _ in range(max_iter):
        # Cinemática directa
        x = f_x(thetas, l1, l2, l3)
        y = f_y(thetas, l1, l2, l3)
        
        # Error
        ex = goal_xy[0] - x
        ey = goal_xy[1] - y
        
        # Gradientes aproximados (derivadas parciales w.r.t t1, t2, t3)
        # E = 1/2*(ex^2 + ey^2)
        # dE/dt1 = -(ex * dx/dt1 + ey * dy/dt1)
        dEdt1 = -(ex * dxdt1(thetas, l1, l2, l3) + ey * dydt1(thetas, l1, l2, l3))
        dEdt2 = -(ex * dxdt2(thetas, l1, l2, l3) + ey * dydt2(thetas, l1, l2, l3))
        dEdt3 = -(ex * dxdt3(thetas, l1, l2, l3) + ey * dydt3(thetas, l1, l2, l3))
        
        # Actualizar ángulos (gradiente descendente)
        thetas[0] -= lr * dEdt1
        thetas[1] -= lr * dEdt2
        thetas[2] -= lr * dEdt3
        
        # Romper si error pequeño
        if np.hypot(ex, ey) < 1e-3:
            break
    
    return thetas

# Estas funciones definen x(t1,t2,t3) = forward kinematics:
def f_x(t, l1, l2, l3):
    return l1*np.cos(t[0]) + l2*np.cos(t[0]+t[1]) + l3*np.cos(t[0]+t[1]+t[2])
def f_y(t, l1, l2, l3):
    return l1*np.sin(t[0]) + l2*np.sin(t[0]+t[1]) + l3*np.sin(t[0]+t[1]+t[2])

# Derivadas parciales w.r.t. t1, t2, t3
def dxdt1(t, l1, l2, l3):
    return -l1*np.sin(t[0]) - l2*np.sin(t[0]+t[1]) - l3*np.sin(t[0]+t[1]+t[2])
def dxdt2(t, l1, l2, l3):
    return -l2*np.sin(t[0]+t[1]) - l3*np.sin(t[0]+t[1]+t[2])
def dxdt3(t, l1, l2, l3):
    return -l3*np.sin(t[0]+t[1]+t[2])

def dydt1(t, l1, l2, l3):
    return  l1*np.cos(t[0]) + l2*np.cos(t[0]+t[1]) + l3*np.cos(t[0]+t[1]+t[2])
def dydt2(t, l1, l2, l3):
    return  l2*np.cos(t[0]+t[1]) + l3*np.cos(t[0]+t[1]+t[2])
def dydt3(t, l1, l2, l3):
    return  l3*np.cos(t[0]+t[1]+t[2])

# ----------------------------------------------------------------
def check_collision_line(x0,y0, x1,y1, x2,y2, x3,y3, pObs1, pObs2):
    # Reutilizamos la lógica de intersección
    if segments_intersect(np.array([x0,y0]), np.array([x1,y1]), pObs1, pObs2):
        return True
    if segments_intersect(np.array([x1,y1]), np.array([x2,y2]), pObs1, pObs2):
        return True
    if segments_intersect(np.array([x2,y2]), np.array([x3,y3]), pObs1, pObs2):
        return True
    return False

def segments_intersect(p1, p2, p3, p4):
    # Igual a la de antes
    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)
    if (o1 != o2) and (o3 != o4): return True
    if o1 == 0 and on_segment(p1, p3, p2): return True
    if o2 == 0 and on_segment(p1, p4, p2): return True
    if o3 == 0 and on_segment(p3, p1, p4): return True
    if o4 == 0 and on_segment(p3, p2, p4): return True
    return False

def orientation(a, b, c):
    val = (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])
    if abs(val) < 1e-12: return 0
    elif val > 0: return 1  # clockwise
    else: return 2  # counterclockwise

def on_segment(p, q, r):
    if (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
        min(p[1], r[1]) <= q[1] <= max(p[1], r[1])):
        return True
    return False

if __name__ == '__main__':
    main()
