import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from numba import njit, prange

# ---------------------------
# 🔹 Forward Kinematics
# ---------------------------
@njit
def calcular_cinematica_directa(angulos, longitudes):
    """Calcula la posición de cada articulación."""
    l1, l2, l3 = longitudes
    x0, y0 = 0.0, 0.0
    x1 = x0 + l1 * np.cos(angulos[0])
    y1 = y0 + l1 * np.sin(angulos[0])
    x2 = x1 + l2 * np.cos(angulos[0] + angulos[1])
    y2 = y1 + l2 * np.sin(angulos[0] + angulos[1])
    x3 = x2 + l3 * np.cos(angulos[0] + angulos[1] + angulos[2])
    y3 = y2 + l3 * np.sin(angulos[0] + angulos[1] + angulos[2])
    return [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]

# ---------------------------
# 🔹 Collision Checking
# ---------------------------
def colision_segmento_circulo(puntoA, puntoB, obstaculo):
    """Revisa si un segmento del brazo colisiona con un obstáculo circular."""
    xA, yA = puntoA
    xB, yB = puntoB
    xc, yc, radio = obstaculo[:3]
    dx = xB - xA
    dy = yB - yA
    fx = xA - xc
    fy = yA - yc
    a = dx**2 + dy**2
    b = 2 * (fx * dx + fy * dy)
    c = (fx**2 + fy**2) - radio**2
    disc = b**2 - 4 * a * c
    if disc < 0:
        return False  # No hay colisión
    disc = np.sqrt(disc)
    t1 = (-b - disc) / (2 * a)
    t2 = (-b + disc) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)

def hay_colision(posiciones, obstaculos):
    """Retorna True si alguna parte del brazo toca un obstáculo."""
    for obstaculo in obstaculos:
        for i in range(len(posiciones) - 1):
            if colision_segmento_circulo(posiciones[i], posiciones[i+1], obstaculo):
                return True
    return False

# ---------------------------
# 🔹 Inverse Kinematics (Avoiding Collisions)
# ---------------------------
def calcular_cinematica_inversa(objetivo, longitudes, obstaculos):
    """Encuentra los ángulos óptimos para alcanzar el objetivo evitando colisiones."""
    def funcion_objetivo(angulos):
        pos = calcular_cinematica_directa(angulos, longitudes)
        if hay_colision(pos, obstaculos):
            return 1e6  # Penalización alta para colisiones
        return np.linalg.norm(np.array(pos[-1]) - np.array(objetivo))

    resultado = differential_evolution(funcion_objetivo, bounds=[(-np.pi, np.pi)] * 3,
                                       strategy='best1bin', maxiter=1000, polish=True)
    if resultado.success:
        return resultado.x
    else:
        return None

# ---------------------------
# 🔹 Smooth Trajectory Interpolation
# ---------------------------
def interpolar_trayectoria(angulos_inicio, angulos_fin, pasos=50):
    """Genera una trayectoria suave entre dos configuraciones."""
    ang_ini = np.array(angulos_inicio)
    ang_fin = np.array(angulos_fin)
    diff = (ang_fin - ang_ini + np.pi) % (2 * np.pi) - np.pi
    fracciones = np.linspace(0, 1, pasos).reshape(-1, 1)
    trayectorias = ang_ini + diff * fracciones
    return trayectorias

# ---------------------------
# 🔹 Visualization
# ---------------------------
def graficar_movimiento(trayectoria, longitudes, obstaculos):
    """Muestra la animación del brazo en movimiento y lo deja en la posición final."""
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_title('Movimiento del Brazo Robótico')

    # Dibujar obstáculos
    for obst in obstaculos:
        circ = plt.Circle((obst[0], obst[1]), obst[2], color='r', alpha=0.5)
        ax.add_patch(circ)

    # Animación del movimiento
    for angulos in trayectoria:
        posiciones = calcular_cinematica_directa(angulos, longitudes)
        xs, ys = zip(*posiciones)
        ax.plot(xs, ys, 'bo-', linewidth=3)
        plt.pause(0.1)
        ax.clear()  # Limpiar para la siguiente iteración
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_title('Movimiento del Brazo Robótico')
        for obst in obstaculos:
            circ = plt.Circle((obst[0], obst[1]), obst[2], color='r', alpha=0.5)
            ax.add_patch(circ)

    # 🔹 Mantener el brazo en la posición final 🔹
    posiciones_final = calcular_cinematica_directa(trayectoria[-1], longitudes)
    xs, ys = zip(*posiciones_final)
    ax.plot(xs, ys, 'bo-', linewidth=3)  # Dibuja el brazo en su estado final
    plt.show()

# ---------------------------
# 🔹 Main Execution
# ---------------------------
if __name__ == "__main__":
    # Definir el brazo
    longitudes = [1, 1, 1]
    
    # Definir obstáculos [(x, y, radio)]
    obstaculos = [
        (1.5, 0.5, 0.3),
        (1.0, 1.0, 0.2),
    ]

    # Posición inicial
    angulos_iniciales = [0, 0, 0]

    # Objetivo
    objetivo = (2, 0.5)

    # Calcular cinemática inversa evitando colisiones
    angulos_finales = calcular_cinematica_inversa(objetivo, longitudes, obstaculos)

    if angulos_finales is None:
        print("⚠ No se encontró una solución sin colisiones.")
    else:
        print(f"✅ Solución encontrada: {np.rad2deg(angulos_finales)} grados")

        # Generar trayectoria sin colisiones
        trayectoria = interpolar_trayectoria(angulos_iniciales, angulos_finales, pasos=50)

        # Visualizar el movimiento
        graficar_movimiento(trayectoria, longitudes, obstaculos)
