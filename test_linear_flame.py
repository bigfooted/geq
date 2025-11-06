import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def initial_horizontal_flame(X, Y, y_flame):
    """Create level set with G = -(Y - y_flame) so G>0 below, G<0 above"""
    return -(Y - y_flame)


def analytical_flame_position(t, y0, U):
    """Returns y0 + U*t"""
    return y0 + U * t


def extract_flame_position(G, X, Y):
    """Extract mean y-position of G=0 contour"""
    contours = plt.contour(X, Y, G, levels=[0])
    y_positions = [c.get_paths()[0].vertices[:, 1].mean() for c in contours]
    return np.mean(y_positions) if y_positions else None


def create_plots(G, X, Y, analytical_positions, time_steps):
    """Visualization function creating contour plots, position comparison, and 3D surface"""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, G, levels=50)
    plt.colorbar(label='G value')
    plt.title('Level Set Contour')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.subplot(1, 2, 2)
    plt.plot(time_steps, analytical_positions, label='Analytical Position', color='red')
    plt.title('Flame Position Comparison')
    plt.xlabel('Time')
    plt.ylabel('Flame Position (y)')
    plt.legend()

    plt.show()


def test_linear_flame(t_final=2.0, time_scheme='rk2', use_reinit=True, verbose=True):
    """Main test function"""
    # Parameters
    L = 1.0
    nx = 101
    ny = 101
    dt = 0.001
    save_interval = 50
    reinit_interval = 50
    y0 = 0.3
    U = 0.2 - 0.1  # Expected flame velocity

    # Create grid
    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    X, Y = np.meshgrid(x, y)

    # Initial condition
    G = initial_horizontal_flame(X, Y, y0)
    analytical_positions = []
    time_steps = []

    for t in range(int(t_final / dt)):
        if t % save_interval == 0:
            time_steps.append(t * dt)
            analytical_y = analytical_flame_position(t * dt, y0, U)
            analytical_positions.append(analytical_y)

        # Update G here with your numerical scheme (not implemented)

        if use_reinit and t % reinit_interval == 0:
            G = initial_horizontal_flame(X, Y, y0)  # Reinitialize

    # Extract flame position
    mean_flame_position = extract_flame_position(G, X, Y)

    if verbose:
        print(f'Mean flame position: {mean_flame_position}')

    # Create plots
    create_plots(G, X, Y, analytical_positions, time_steps)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test Linear Flame Propagation')
    parser.add_argument('--scheme', choices=['euler', 'rk2'], default='rk2', help='Time integration scheme')
    parser.add_argument('--t_final', type=float, default=2.0, help='Final time for the simulation')
    parser.add_argument('--no_reinit', action='store_true', help='Disable reinitialization')
    args = parser.parse_args()

    test_linear_flame(t_final=args.t_final, time_scheme=args.scheme, use_reinit=not args.no_reinit, verbose=True)