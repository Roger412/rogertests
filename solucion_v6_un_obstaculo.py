import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import differential_evolution

def forward_kinematics(theta, link_lengths):
    l1, l2, l3 = link_lengths
    # Base fija en (0,0)
    x0, y0 = 0, 0
    x1 = x0 + l1 * np.cos(theta[0])
    y1 = y0 + l1 * np.sin(theta[0])
    x2 = x1 + l2 * np.cos(theta[0] + theta[1])
    y2 = y1 + l2 * np.sin(theta[0] + theta[1])
    x3 = x2 + l3 * np.cos(theta[0] + theta[1] + theta[2])
    y3 = y2 + l3 * np.sin(theta[0] + theta[1] + theta[2])
    positions = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
    return positions

def segment_circle_collision(p1, p2, obstacle):
    """Verifica si el segmento definido por p1 y p2 intersecta el círculo (obstáculo)."""
    x1, y1 = p1
    x2, y2 = p2
    xc, yc, r = obstacle
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - xc, y1 - yc
    a = dx**2 + dy**2
    b = 2 * (fx * dx + fy * dy)
    c = (fx**2 + fy**2) - r**2
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return False  # No hay intersección
    discriminant = np.sqrt(discriminant)
    t1 = (-b - discriminant) / (2 * a)
    t2 = (-b + discriminant) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)

def is_collision(positions, obstacle):
    """Verifica que ninguno de los 3 eslabones (segmentos) colisione con el obstáculo."""
    if not positions:
        return True
    for i in range(len(positions) - 1):
        if segment_circle_collision(positions[i], positions[i + 1], obstacle):
            return True
    return False

def check_cspace_collision(theta, cspace, resolution):
    index = ((theta + np.pi) / (2 * np.pi) * resolution).astype(int)
    index = np.clip(index, 0, resolution - 1)
    return cspace[tuple(index)] == 1

def generate_cspace(resolution, link_lengths, obstacle):
    theta_range = np.linspace(-np.pi, np.pi, resolution)
    cspace = np.zeros((resolution, resolution, resolution))
    for i, t1 in enumerate(theta_range):
        for j, t2 in enumerate(theta_range):
            for k, t3 in enumerate(theta_range):
                positions = forward_kinematics([t1, t2, t3], link_lengths)
                if is_collision(positions, obstacle):
                    cspace[i, j, k] = 1
    return cspace

def inverse_kinematics(target, link_lengths, obstacle, cspace, resolution, initial_theta):
    best_solution = None
    best_cost = np.inf
    trajectory = []
    
    def objective(theta):
        positions = forward_kinematics(theta, link_lengths)
        if not positions:
            return np.inf
        # Si se detecta colisión en alguno de los 3 eslabones, se penaliza fuertemente.
        if is_collision(positions, obstacle):
            return 1e6
        end_effector = positions[-1]
        dist_to_target = np.linalg.norm(np.array(end_effector) - np.array(target))
        # Penalización extra si el brazo se acerca demasiado al obstáculo (margen de seguridad)
        safety_margin = 0.2
        penalty = 0
        for p in positions:
            distance = np.linalg.norm(np.array(p) - np.array(obstacle[:2]))
            if distance < obstacle[2] + safety_margin:
                penalty += (obstacle[2] + safety_margin - distance) * 50
        return dist_to_target + penalty

    initial_guesses = [
        initial_theta,  
        [np.pi/4, -np.pi/4, np.pi/4],  
        [-np.pi/4, np.pi/4, -np.pi/4],  
        [0, np.pi/2, -np.pi/2],  
        [np.pi/2, 0, -np.pi/2]
    ]

    for guess in initial_guesses:
        result = differential_evolution(objective, bounds=[(-np.pi, np.pi)]*3,
                                        strategy='best1bin', maxiter=1000, polish=True)
        if result.success:
            final_positions = forward_kinematics(result.x, link_lengths)
            cost = np.linalg.norm(np.array(final_positions[-1]) - np.array(target))
            if cost < best_cost and not is_collision(final_positions, obstacle):
                best_solution = result.x
                best_cost = cost
            trajectory.append(result.x)
    
    if best_solution is not None:
        return best_solution, np.array(trajectory)
    else:
        print("⚠ No se encontró una solución válida sin colisión.")
        return None, np.array([])

def plot_cspace(cspace, trajectory):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    theta_range = np.linspace(-np.pi, np.pi, cspace.shape[0])
    x, y, z = np.meshgrid(theta_range, theta_range, theta_range)
    ax.scatter(x[cspace==1], y[cspace==1], z[cspace==1], c='r', marker='o', alpha=0.3, label='Colisión')
    if trajectory.shape[0] > 1:
        ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], 'b-', linewidth=2, label='Trayectoria de Ángulos')
        ax.scatter(trajectory[0,0], trajectory[0,1], trajectory[0,2], c='g', marker='o', s=100, label='Inicio')
        ax.scatter(trajectory[-1,0], trajectory[-1,1], trajectory[-1,2], c='y', marker='o', s=100, label='Final')
    ax.set_xlabel('Theta1')
    ax.set_ylabel('Theta2')
    ax.set_zlabel('Theta3')
    plt.title('C-space con Trayectoria de Ángulos')
    plt.legend()
    plt.show()

def plot_robot(theta, link_lengths, obstacle):
    fig, ax = plt.subplots()
    positions = forward_kinematics(theta, link_lengths)
    if not positions:
        print("⚠ Error en forward_kinematics.")
        return
    x_values, y_values = zip(*positions)
    ax.plot(x_values, y_values, 'bo-', linewidth=3, label='Brazo Robótico')
    ax.scatter(obstacle[0], obstacle[1], color='k', s=100, label='Obstáculo')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.set_title('Configuración final del Brazo')
    ax.grid()
    plt.show(block=False)

def plot_initial_configuration(initial_theta, link_lengths, obstacle):
    positions = forward_kinematics(initial_theta, link_lengths)
    if not positions:
        print("⚠ Error en forward_kinematics.")
        return
    x_values, y_values = zip(*positions)
    plt.figure()
    plt.plot(x_values, y_values, 'bo-', linewidth=3, label='Configuración Inicial')
    plt.scatter(obstacle[0], obstacle[1], color='k', s=100, label='Obstáculo')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Configuración Inicial del Brazo')
    plt.grid()
    plt.show(block=False)

# --- Nuevas funciones para la simulación del movimiento ---

def interpolate_path(theta_start, theta_end, steps):
    """Genera una trayectoria lineal entre dos configuraciones en el espacio articular."""
    return np.linspace(theta_start, theta_end, steps)

def compute_via_point(initial_ee, target, obstacle, margin=0.3):
    """
    Calcula un punto vía en el espacio (x,y) si la línea directa entre la posición
    inicial y el objetivo se acerca demasiado al obstáculo. Se generan dos candidatos y
    se selecciona el que ofrezca mayor alejamiento del obstáculo.
    """
    O = np.array([obstacle[0], obstacle[1]])
    P0 = np.array(initial_ee)
    P1 = np.array(target)
    v = P1 - P0
    if np.linalg.norm(v) == 0:
        return None
    v_norm = v / np.linalg.norm(v)
    t = np.dot(O - P0, v_norm)
    projection = P0 + t * v_norm
    d = np.linalg.norm(O - projection)
    if d < obstacle[2] + margin:
        offset = obstacle[2] + margin - d
        perp = np.array([-v_norm[1], v_norm[0]])
        via1 = projection + perp * offset
        via2 = projection - perp * offset
        return via1 if np.linalg.norm(via1 - O) > np.linalg.norm(via2 - O) else via2
    else:
        return None

def simulate_arm_motion(path, link_lengths, obstacle, target):
    """
    Anima el movimiento del brazo a lo largo de la trayectoria (en espacio articular).
    Dibuja en cada paso el brazo, el obstáculo y el objetivo.
    """
    fig, ax = plt.subplots()
    for theta in path:
        ax.clear()
        positions = forward_kinematics(theta, link_lengths)
        x, y = zip(*positions)
        ax.plot(x, y, 'bo-', linewidth=3)
        # Dibujar obstáculo
        circle = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='r', alpha=0.5)
        ax.add_patch(circle)
        # Dibujar objetivo
        ax.scatter(target[0], target[1], color='g', s=100, label='Objetivo')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Simulación del movimiento del brazo')
        ax.grid()
        plt.pause(0.05)
    plt.show()

# --- Parámetros y ejecución principal ---

link_lengths = [1, 1, 1]
obstacle = (0.5, 0.5, 0.2)
resolution = 30
target_xy = (1, 1)
initial_theta = [-np.pi/4, np.pi/2, -np.pi/3]  # Configuración inicial

# Generar C-space (opcional para visualizar la zona de colisión en el espacio articular)
cspace = generate_cspace(resolution, link_lengths, obstacle)



# Resolver cinemática inversa para el objetivo final
final_angles, trajectory = inverse_kinematics(target_xy, link_lengths, obstacle, cspace, resolution, initial_theta)

if final_angles is not None:
    
    print("Ángulos finales:", final_angles)
    
    # Obtener la posición inicial del efector final
    initial_ee = forward_kinematics(initial_theta, link_lengths)[-1]
    # Calcular un punto vía para evitar el obstáculo
    via_point = compute_via_point(initial_ee, target_xy, obstacle, margin=0.3)
    
    if via_point is not None:
        print("Se usará un punto vía para evitar el obstáculo:", via_point)
        config_via, _ = inverse_kinematics(via_point, link_lengths, obstacle, cspace, resolution, initial_theta)
        if config_via is not None:
            path1 = interpolate_path(initial_theta, config_via, steps=50)
            path2 = interpolate_path(config_via, final_angles, steps=50)
            full_path = np.vstack((path1, path2))
        else:
            full_path = interpolate_path(initial_theta, final_angles, steps=100)
    else:
        full_path = interpolate_path(initial_theta, final_angles, steps=100)
    
    # Simulación animada del movimiento
    simulate_arm_motion(full_path, link_lengths, obstacle, target_xy)
    plot_robot(final_angles, link_lengths, obstacle)
    
else:
    print("⚠ No se encontró una solución válida sin colisión.")

# Graficar configuración inicial
plot_initial_configuration(initial_theta, link_lengths, obstacle)
# (Opcional) Graficar el C-space y la trayectoria en el espacio articular
plot_cspace(cspace, trajectory)
