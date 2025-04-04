import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Initial parameters
W1_V = 1
W2_V = 1
w_radius = 1
l = 1
time_values = [0]
x_dot_values = [0]
y_dot_values = [0]
theta_dot_values = [0]
t = 0  # Time counter

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

# Box properties
box_size = 50
x_pos = 375  # Starting position (center of the screen)
y_pos = 275
theta = 0

# Create figure and axis for the plots
fig, ax = plt.subplots(3, 1, figsize=(6, 8))
plt.subplots_adjust(left=0.1, bottom=0.3)

# Initialize empty plots
line_x_dot, = ax[0].plot(time_values, x_dot_values, 'r-', label="x_dot")
line_y_dot, = ax[1].plot(time_values, y_dot_values, 'g-', label="y_dot")
line_theta_dot, = ax[2].plot(time_values, theta_dot_values, 'b-', label="theta_dot")

for a in ax:
    a.legend()
    a.set_xlim(0, 10)  # Initial x-axis range
    a.set_ylim(-2, 2)  # Initial y-axis range

# Add sliders for real-time control of variables
ax_w1 = plt.axes([0.2, 0.15, 0.65, 0.03])
ax_w2 = plt.axes([0.2, 0.1, 0.65, 0.03])
ax_w_radius = plt.axes([0.2, 0.05, 0.65, 0.03])
ax_theta = plt.axes([0.2, 0.0, 0.65, 0.03])

slider_w1 = Slider(ax_w1, 'W1_V', -100, 100, valinit=W1_V)
slider_w2 = Slider(ax_w2, 'W2_V', -100, 100, valinit=W2_V)
slider_w_radius = Slider(ax_w_radius, 'w_radius', 0.1, 50, valinit=w_radius)

def draw_grass_background():
    """Draw a grassy background with alternating shades of green."""
    grass_color_1 = (34, 139, 34)  # Forest green
    grass_color_2 = (0, 128, 0)    # Green
    stripe_height = 20  # Height of each green stripe

    for y in range(0, 600, stripe_height * 2):
        pygame.draw.rect(screen, grass_color_1, pygame.Rect(0, y, 800, stripe_height))
        pygame.draw.rect(screen, grass_color_2, pygame.Rect(0, y + stripe_height, 800, stripe_height))

def update_plot(frame):
    global t, W1_V, W2_V, w_radius, theta, x_pos, y_pos

    # Get updated values from sliders
    W1_V = slider_w1.val
    W2_V = slider_w2.val
    w_radius = slider_w_radius.val

    # Compute inertial velocities
    x_dot = w_radius * ((W1_V + W2_V) / 2) * np.cos(theta)
    y_dot = w_radius * ((W1_V + W2_V) / 2) * np.sin(theta)
    theta_dot = w_radius * ((W1_V - W2_V) / 2) / l

    # Update time
    t += 0.1
    time_values.append(t)
    x_dot_values.append(x_dot)
    y_dot_values.append(y_dot)
    theta_dot_values.append(theta_dot)

    # Limit to last 100 values for performance
    time_values[:] = time_values[-100:]
    x_dot_values[:] = x_dot_values[-100:]
    y_dot_values[:] = y_dot_values[-100:]
    theta_dot_values[:] = theta_dot_values[-100:]

    # Update plots
    line_x_dot.set_xdata(time_values)
    line_x_dot.set_ydata(x_dot_values)
    
    line_y_dot.set_xdata(time_values)
    line_y_dot.set_ydata(y_dot_values)
    
    line_theta_dot.set_xdata(time_values)
    line_theta_dot.set_ydata(theta_dot_values)

    # Adjust axis limits dynamically
    for i, (line, values) in enumerate(zip([line_x_dot, line_y_dot, line_theta_dot], 
                                           [x_dot_values, y_dot_values, theta_dot_values])):
        ax[i].set_xlim(max(0, t - 10), t)  # Keep last 10 seconds visible
        if values:  # Avoid empty list errors
            ax[i].set_ylim(min(values) - 0.5, max(values) + 0.5)  # Add padding

    plt.draw()
    plt.pause(0.1)

    # Move the box based on inertial velocities
    x_pos += x_dot * 0.1  # 0.1 is the time step
    y_pos += y_dot * 0.1
    theta += theta_dot * 0.1

    # Draw the grassy background
    draw_grass_background()

    # Draw the box in the Pygame window
    pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(x_pos, y_pos, box_size, box_size))  # Draw the box

    # Update the display
    pygame.display.flip()

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    update_plot(0)
    clock.tick(30)  # Control frame rate to 30 FPS

pygame.quit()
plt.close(fig)
