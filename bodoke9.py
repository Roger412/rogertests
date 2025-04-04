import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import differential_evolution
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from numba import njit, prange

# ---------------------------
# Funciones Base (sin aceleración)
# ---------------------------

def calcular_cinematica_directa(angulos, longitudes):
    """
    Calcula las posiciones (x,y) de cada articulación de un brazo de 3 eslabones.
    Se asume que la base está en (0,0).
    """
    l1, l2, l3 = longitudes
    x0, y0 = 0, 0
    x1 = x0 + l1 * np.cos(angulos[0])
    y1 = y0 + l1 * np.sin(angulos[0])
    x2 = x1 + l2 * np.cos(angulos[0] + angulos[1])
    y2 = y1 + l2 * np.sin(angulos[0] + angulos[1])
    x3 = x2 + l3 * np.cos(angulos[0] + angulos[1] + angulos[2])
    y3 = y2 + l3 * np.sin(angulos[0] + angulos[1] + angulos[2])
    return [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]

def colision_segmento_circulo(puntoA, puntoB, obstaculo):
    """
    Determina si la línea entre 'puntoA' y 'puntoB' cruza el círculo que representa un obstáculo.
    """
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
        return False
    disc = np.sqrt(disc)
    t1 = (-b - disc) / (2 * a)
    t2 = (-b + disc) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)

def indice_colision(posiciones, obstaculos):
    """
    Recorre cada segmento del brazo y para cada obstáculo determina si hay colisión.
    Retorna 0 si no hay colisión, o (índice+1) del primer obstáculo que colisiona.
    """
    for idx, obstaculo in enumerate(obstaculos):
        for i in range(len(posiciones) - 1):
            if colision_segmento_circulo(posiciones[i], posiciones[i+1], obstaculo):
                return idx + 1
    return 0

def hay_colision(posiciones, obstaculos):
    """
    Devuelve True si al menos uno de los obstáculos interfiere con algún segmento del brazo.
    """
    return indice_colision(posiciones, obstaculos) != 0

def revisar_colision_espacio_config(angulos, espacio_config, resolucion):
    """
    Verifica si la configuración dada (angulos) está marcada como colisionante en el C-space.
    """
    indices = ((angulos + np.pi) / (2 * np.pi) * resolucion).astype(int)
    indices = np.clip(indices, 0, resolucion - 1)
    return espacio_config[tuple(indices)] != 0

# ---------------------------
# Funciones aceleradas con Numba para el C-space
# ---------------------------

@njit
def calcular_cinematica_directa_nb(angulos, longitudes):
    l1 = longitudes[0]
    l2 = longitudes[1]
    l3 = longitudes[2]
    x0 = 0.0
    y0 = 0.0
    x1 = x0 + l1 * np.cos(angulos[0])
    y1 = y0 + l1 * np.sin(angulos[0])
    x2 = x1 + l2 * np.cos(angulos[0] + angulos[1])
    y2 = y1 + l2 * np.sin(angulos[0] + angulos[1])
    x3 = x2 + l3 * np.cos(angulos[0] + angulos[1] + angulos[2])
    y3 = y2 + l3 * np.sin(angulos[0] + angulos[1] + angulos[2])
    return np.array([x0, y0, x1, y1, x2, y2, x3, y3])

@njit
def colision_segmento_circulo_nb(puntoA, puntoB, obstaculo):
    xA = puntoA[0]
    yA = puntoA[1]
    xB = puntoB[0]
    yB = puntoB[1]
    xc = obstaculo[0]
    yc = obstaculo[1]
    radio = obstaculo[2]
    dx = xB - xA
    dy = yB - yA
    fx = xA - xc
    fy = yA - yc
    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = (fx * fx + fy * fy) - radio * radio
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    disc = np.sqrt(disc)
    t1 = (-b - disc) / (2 * a)
    t2 = (-b + disc) / (2 * a)
    if (t1 >= 0.0 and t1 <= 1.0) or (t2 >= 0.0 and t2 <= 1.0):
        return True
    else:
        return False

@njit(parallel=True)
def crear_espacio_configuracion_nb(resolucion, longitudes, obstaculos_num):
    espacio_config = np.zeros((resolucion, resolucion, resolucion), dtype=np.int32)
    rango_angulos = np.linspace(-np.pi, np.pi, resolucion)
    for i in prange(resolucion):
        for j in range(resolucion):
            for k in range(resolucion):
                ang = np.array([rango_angulos[i], rango_angulos[j], rango_angulos[k]])
                pos_nb = calcular_cinematica_directa_nb(ang, longitudes)
                pts = np.empty((4, 2))
                pts[0,0] = pos_nb[0]; pts[0,1] = pos_nb[1]
                pts[1,0] = pos_nb[2]; pts[1,1] = pos_nb[3]
                pts[2,0] = pos_nb[4]; pts[2,1] = pos_nb[5]
                pts[3,0] = pos_nb[6]; pts[3,1] = pos_nb[7]
                collision = 0
                for o in range(obstaculos_num.shape[0]):
                    for seg in range(3):
                        if colision_segmento_circulo_nb(pts[seg, :], pts[seg+1, :], obstaculos_num[o, :]):
                            collision = o + 1
                            break
                    if collision != 0:
                        break
                espacio_config[i, j, k] = collision
    return espacio_config

# ---------------------------
# Funciones de cinemática inversa, interpolación y cálculo de punto vía (sin aceleración)
# ---------------------------

def calcular_cinematica_inversa(objetivo, longitudes, obstaculos, espacio_config, resolucion, angulos_iniciales):
    solucion_optima = None
    costo_minimo = np.inf
    trayectorias = []
    def funcion_objetivo(angulos):
        pos = calcular_cinematica_directa(angulos, longitudes)
        if not pos:
            return np.inf
        if hay_colision(pos, obstaculos):
            return 1e6
        efector_final = pos[-1]
        distancia = np.linalg.norm(np.array(efector_final) - np.array(objetivo))
        margen_seguridad = 0.2
        penalizacion = 0
        for punto in pos:
            for obstaculo in obstaculos:
                dist = np.linalg.norm(np.array(punto) - np.array(obstaculo[:2]))
                if dist < obstaculo[2] + margen_seguridad:
                    penalizacion += (obstaculo[2] + margen_seguridad - dist) * 50
        return distancia + penalizacion
    conjeturas_iniciales = [
        angulos_iniciales,
        [np.pi/4, -np.pi/4, np.pi/4],
        [-np.pi/4, np.pi/4, -np.pi/4],
        [0, np.pi/2, -np.pi/2],
        [np.pi/2, 0, -np.pi/2]
    ]
    for guess in conjeturas_iniciales:
        resultado = differential_evolution(funcion_objetivo, bounds=[(-np.pi, np.pi)] * 3,
                                           strategy='best1bin', maxiter=1000, polish=True)
        if resultado.success:
            pos_final = calcular_cinematica_directa(resultado.x, longitudes)
            costo_actual = np.linalg.norm(np.array(pos_final[-1]) - np.array(objetivo))
            if costo_actual < costo_minimo and not hay_colision(pos_final, obstaculos):
                solucion_optima = resultado.x
                costo_minimo = costo_actual
            trayectorias.append(resultado.x)
    if solucion_optima is not None:
        return solucion_optima, np.array(trayectorias)
    else:
        print("⚠ No se pudo hallar una solución sin colisiones.")
        return None, np.array([])

def interpolar_trayectoria(angulos_inicio, angulos_fin, pasos):
    ang_ini = np.array(angulos_inicio)
    ang_fin = np.array(angulos_fin)
    diff = (ang_fin - ang_ini + np.pi) % (2 * np.pi) - np.pi
    fracciones = np.linspace(0, 1, pasos).reshape(-1, 1)
    trayectorias = ang_ini + diff * fracciones
    return trayectorias

def calcular_punto_via(efector_inicial, objetivo, obstaculos, margen=0.3):
    P0 = np.array(efector_inicial)
    P1 = np.array(objetivo)
    vector_dir = P1 - P0
    if np.linalg.norm(vector_dir) == 0:
        return None
    vector_dir_norm = vector_dir / np.linalg.norm(vector_dir)
    for obstaculo in obstaculos:
        centro_obs = np.array([obstaculo[0], obstaculo[1]])
        t = np.dot(centro_obs - P0, vector_dir_norm)
        punto_proy = P0 + t * vector_dir_norm
        distancia = np.linalg.norm(centro_obs - punto_proy)
        if distancia < obstaculo[2] + margen:
            compensacion = obstaculo[2] + margen - distancia
            perpendicular = np.array([-vector_dir_norm[1], vector_dir_norm[0]])
            candidato1 = punto_proy + perpendicular * compensacion
            candidato2 = punto_proy - perpendicular * compensacion
            return candidato1 if np.linalg.norm(candidato1 - centro_obs) > np.linalg.norm(candidato2 - centro_obs) else candidato2
    return None

# ---------------------------
# Clase de Simulación Interactiva con Tkinter (Workspace y C-space)
# ---------------------------
class RealTimeSimulation:
    def __init__(self, master):
        self.master = master
        self.master.title("Simulación en Tiempo Real del Brazo Robótico")
        
        # Panel de controles para definir el nuevo objetivo
        control_frame = tk.Frame(master)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        tk.Label(control_frame, text="Objetivo X:").pack(side=tk.LEFT, padx=5)
        self.entry_x = tk.Entry(control_frame, width=5)
        self.entry_x.pack(side=tk.LEFT)
        
        tk.Label(control_frame, text="Objetivo Y:").pack(side=tk.LEFT, padx=5)
        self.entry_y = tk.Entry(control_frame, width=5)
        self.entry_y.pack(side=tk.LEFT)
        
        self.button = tk.Button(control_frame, text="Actualizar Objetivo", command=self.update_target)
        self.button.pack(side=tk.LEFT, padx=5)
        
        # Frame para el espacio de trabajo (2D)
        self.frame_workspace = tk.Frame(master)
        self.frame_workspace.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig_workspace, self.ax_workspace = plt.subplots()
        self.canvas_workspace = FigureCanvasTkAgg(self.fig_workspace, master=self.frame_workspace)
        self.canvas_workspace.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Frame para el C-space (3D)
        self.frame_cspace = tk.Frame(master)
        self.frame_cspace.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig_cspace = plt.figure()
        self.ax_cspace = self.fig_cspace.add_subplot(111, projection='3d')
        self.canvas_cspace = FigureCanvasTkAgg(self.fig_cspace, master=self.frame_cspace)
        self.canvas_cspace.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Parámetros fijos
        self.longitudes = [1, 1, 1]
        self.obstaculos = [
            (2.2, 0.4, 0.3, 'm'),
            (2.0, 0.7, 0.25, 'c'),
            (1.8, 0.9, 0.2, 'y')
        ]
        self.resolucion = 30
        self.angulos_iniciales = [-np.pi/4, np.pi/2, -np.pi/3]
        # Para acelerar el cálculo del C-space, se utiliza la versión JIT con Numba.
        # Se crea un arreglo numérico de obstáculos (sin la parte de color)
        obstaculos_num = np.array([[obs[0], obs[1], obs[2]] for obs in self.obstaculos])
        self.espacio_config = crear_espacio_configuracion_nb(self.resolucion, np.array(self.longitudes), obstaculos_num)
        self.objetivo = (1, 1)  # Objetivo inicial
        
        # Dibujar inicialmente tanto el workspace como el C-space
        self.update_workspace([])
        self.update_cspace(np.array([]))
    
    def update_target(self):
        try:
            x = float(self.entry_x.get())
            y = float(self.entry_y.get())
            self.objetivo = (x, y)
            print("Nuevo objetivo:", self.objetivo)
            self.simulate_movement()
        except Exception as e:
            print("Error en los valores ingresados:", e)
    
    def simulate_movement(self):
        ang_final, tray = calcular_cinematica_inversa(self.objetivo, self.longitudes, self.obstaculos,
                                                       self.espacio_config, self.resolucion, self.angulos_iniciales)
        if ang_final is None:
            print("No se encontró solución sin colisiones.")
            return
        efector_inicial = calcular_cinematica_directa(self.angulos_iniciales, self.longitudes)[-1]
        punto_via = calcular_punto_via(efector_inicial, self.objetivo, self.obstaculos, margen=0.3)
        if punto_via is not None:
            config_via, _ = calcular_cinematica_inversa(punto_via, self.longitudes, self.obstaculos,
                                                         self.espacio_config, self.resolucion, self.angulos_iniciales)
            if config_via is not None:
                tray1 = interpolar_trayectoria(self.angulos_iniciales, config_via, pasos=50)
                tray2 = interpolar_trayectoria(config_via, ang_final, pasos=50)
                trayectoria_completa = np.vstack((tray1, tray2))
            else:
                trayectoria_completa = interpolar_trayectoria(self.angulos_iniciales, ang_final, pasos=100)
        else:
            trayectoria_completa = interpolar_trayectoria(self.angulos_iniciales, ang_final, pasos=100)
        
        self.update_workspace(trayectoria_completa)
        self.update_cspace(trayectoria_completa)
        self.angulos_iniciales = ang_final

    def update_workspace(self, trayectoria):
        self.ax_workspace.clear()
        # Prender la cuadrícula en la gráfica 2D
        self.ax_workspace.grid(True)
        if len(trayectoria) > 0:
            pos = calcular_cinematica_directa(trayectoria[-1], self.longitudes)
        else:
            pos = calcular_cinematica_directa(self.angulos_iniciales, self.longitudes)
        xs, ys = zip(*pos)
        self.ax_workspace.plot(xs, ys, 'bo-', linewidth=3, label="Brazo")
        for obst in self.obstaculos:
            circ = plt.Circle((obst[0], obst[1]), obst[2], color=obst[3], alpha=0.5)
            self.ax_workspace.add_patch(circ)
        self.ax_workspace.scatter(self.objetivo[0], self.objetivo[1], color='g', s=100, label="Objetivo")
        self.ax_workspace.set_xlim(-3, 3)
        self.ax_workspace.set_ylim(-3, 3)
        self.ax_workspace.set_xlabel("X")
        self.ax_workspace.set_ylabel("Y")
        self.ax_workspace.set_title("Espacio de Trabajo")
        self.ax_workspace.legend()
        self.canvas_workspace.draw()
    
    def update_cspace(self, trayectoria):
        self.ax_cspace.clear()
        rango_angulos = np.linspace(-np.pi, np.pi, self.resolucion)
        X, Y, Z = np.meshgrid(rango_angulos, rango_angulos, rango_angulos)
        for idx, obstaculo in enumerate(self.obstaculos):
            color = obstaculo[3] if len(obstaculo) >= 4 else 'r'
            mask = (self.espacio_config == (idx + 1))
            self.ax_cspace.scatter(X[mask], Y[mask], Z[mask], c=color, marker='o', alpha=0.3, label=f"Obs {idx+1}")
        if trayectoria.size > 0:
            self.ax_cspace.plot(trayectoria[:,0], trayectoria[:,1], trayectoria[:,2], 'b-', linewidth=2, label="Trayectoria")
            self.ax_cspace.scatter(trayectoria[0,0], trayectoria[0,1], trayectoria[0,2], c='g', marker='o', s=100, label="Inicio")
            self.ax_cspace.scatter(trayectoria[-1,0], trayectoria[-1,1], trayectoria[-1,2], c='y', marker='o', s=100, label="Final")
        self.ax_cspace.set_xlabel("θ₁")
        self.ax_cspace.set_ylabel("θ₂")
        self.ax_cspace.set_zlabel("θ₃")
        self.ax_cspace.set_title("Espacio de Configuración")
        self.ax_cspace.legend()
        self.canvas_cspace.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeSimulation(root)
    root.mainloop()
