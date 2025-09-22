#Third order covnergence on 2D brusselator
#Third order 15 stage ESRK implemented on the  2D brussealtor this too produce the graph in mathmatically background of the paper
import numpy as np
import time
import tracemalloc
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Configure NumPy to raise exceptions on floating-point errorsa
np.seterr(over='raise', invalid='raise', divide='raise', under='warn')

def kahan_sum(values):
    """
    Kahan summation for improved numerical accuracy.
    """
    sum_ = np.float128(0.0)
    compensation = np.float128(0.0)
    for value in values:
        y = np.float128(value) - compensation
        temp = sum_ + y
        compensation = (temp - sum_) - y
        sum_ = temp
    return sum_

def brusselator_1d(t, y, N, dx, D_u, D_v, a_param, b_param):
    """
    2D Brusselator reaction-diffusion system.
    """
    u = y[:N]
    v = y[N:]

    du_dt = np.zeros(N, dtype=np.float128)
    dv_dt = np.zeros(N, dtype=np.float128)
    u_xx = np.zeros(N, dtype=np.float128)
    v_xx = np.zeros(N, dtype=np.float128)

    # Internal points (finite difference)
    u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    v_xx[1:-1] = (v[2:] - 2 * v[1:-1] + v[:-2]) / dx**2

    # Neumann boundary conditions
    u_xx[0] = (u[1] - 2 * u[0] + u[2]) / dx**2
    u_xx[-1] = (u[-2] - 2 * u[-1] + u[-3]) / dx**2
    v_xx[0] = (v[1] - 2 * v[0] + v[2]) / dx**2
    v_xx[-1] = (v[-2] - 2 * v[-1] + v[-3]) / dx**2

    reaction_u = a_param - (b_param + 1) * u + (u**2) * v
    reaction_v = b_param * u - (u**2) * v

    du_dt = D_u * u_xx + reaction_u
    dv_dt = D_v * v_xx + reaction_v

    return np.concatenate([du_dt, dv_dt])

def runge_kutta_general(f, t_span, y0, h, A, b, args=()):
    """
    General s-stage explicit Runge-Kutta integrator using Butcher tableau (A, b).

    Parameters
    ----------
    f : callable
        The ODE right-hand side (RHS).
    t_span : tuple (t0, tf)
        Start and end times.
    y0 : ndarray
        Initial condition.
    h : float
        Time step size.
    A : ndarray of shape (s, s)
        The coefficient matrix of the Butcher tableau.
    b : ndarray of shape (s,)
        The weights for the final combination of stages.
    args : tuple
        Additional parameters to pass to f.

    Returns
    -------
    t_values : ndarray
        Array of time points.
    y_values : ndarray
        Array of solutions at each time point.
    """
    t0, tf = t_span
    t = np.float128(t0)
    y = y0.copy()

    # Number of stages
    s = len(b)
    # c_i = sum of row i of A up to i-1 (standard definition for explicit RK)
    c = np.array([kahan_sum(A[i, :i]) for i in range(s)], dtype=np.float128)

    t_values = [t]
    y_values = [y.copy()]

    # Figure out how many steps are required
    # (or step less if it overshoots tf)
    num_steps = int(np.ceil((tf - t0) / h))

    for step in range(num_steps):
        # If very close to tf, adjust h to hit tf exactly
        if t + h > tf + np.float128(1e-14):
            h = tf - t
        if h <= 0:
            break

        # Store stage slopes here
        k_stages = np.zeros((s, y.size), dtype=np.float128)

        # --- Compute all s stages ---
        for i in range(s):
            # Construct the stage value y_stage
            y_stage = y.copy()
            for j in range(i):
                y_stage += h * A[i, j] * k_stages[j]

            t_stage = t + c[i] * h
            k_stages[i] = f(t_stage, y_stage, *args)

        # --- Combine stages to get y_{n+1} ---
        for i in range(s):
            y += h * b[i] * k_stages[i]

        t += h
        t_values.append(t)
        y_values.append(y.copy())

        if t >= tf - np.float128(1e-14):
            # We have reached or exceeded the final time
            break

    return np.array(t_values, dtype=np.float128), np.array(y_values, dtype=np.float128)

def l2_norm_error(y_numerical, y_reference):
    """
    L2 norm of the difference between
    y_numerical and y_reference at final time.
    """
    error = y_numerical - y_reference
    l2_norm = np.linalg.norm(error)
    return np.sqrt(np.mean(l2_norm**2, dtype=np.float128))

def calculate_order_of_convergence(errors, hs):
    """
    Given a list of errors and corresponding step sizes,
    compute the observed order of convergence between consecutive steps.
    """
    orders = []
    for i in range(1, len(errors)):
        if errors[i-1] == 0 or hs[i-1] == hs[i]:
            orders.append(np.nan)
        else:
            order = np.log(errors[i] / errors[i-1]) / np.log(hs[i] / hs[i-1])
            orders.append(order)
    return orders

if __name__ == "__main__":
    # Problem setup
    t_span = (0, 10)
    L = 10.0
    N = 100
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]

    D_u = 0.1
    D_v = 0.05
    a_param = 1.0
    b_param = 3.0

    np.random.seed(0)
    u0 = a_param + 0.1 * np.random.rand(N)
    v0 = b_param / a_param + 0.1 * np.random.rand(N)
    y0 = np.concatenate([u0, v0]).astype(np.float128)

    # Stability limit for an explicit scheme ~ dx^2 / (2 * D_max)
    D_max = max(D_u, D_v)
    dt_max = dx**2 / (2 * D_max)
    print(f"Maximum allowable Δt for stability: {dt_max}")

    # Time-step array for convergence study
    hs = np.linspace(dt_max * 0.9, 0.001, 120)

    # Butcher tableau (A, b). Make sure shape matches s x s with s = len(b)
    A_list = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.0243586417803786, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.0358989324994081, 0.0258303808904268, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.0358989324994081, 0.0358989324994081, 0.0667956303329210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0140960387721938, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0412105997557866, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0149469583607297, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.414086419082813, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00395908281378477, 0, 0, 0, 0, 0, 0, 0],
        [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.480561088337756, 0, 0, 0, 0, 0, 0],
        [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.319660987317690, 0, 0, 0, 0, 0],
        [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0.00668808071535874, 0, 0, 0, 0],
        [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0, 0.0374638233561973, 0, 0, 0],
        [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0, 0.422645975498266, 0.439499983548480, 0, 0],
        [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0, 0.422645975498266, 0.0327614907498598, 0.367805790222090, 0],
    ]
    b_list = [
        0.035898932499408134, 0.035898932499408134, 0.035898932499408134,
        0.035898932499408134, 0.035898932499408134, 0.035898932499408134,
        0.035898932499408134, 0.035898932499408134, 0.006612457947210495,
        0.21674686949693006, 0.0, 0.42264597549826616,
        0.03276149074985981, 0.0330623263939421, 0.0009799086295048407
    ]

    # Pad A so it's exactly s x s if needed
    A_raw = np.array(A_list, dtype=np.float128)
    s = len(b_list)
    if A_raw.shape[0] < s or A_raw.shape[1] < s:
        A_tmp = np.zeros((s, s), dtype=np.float128)
        A_tmp[:A_raw.shape[0], :A_raw.shape[1]] = A_raw
        A_raw = A_tmp

    b_raw = np.array(b_list, dtype=np.float128)

    print(f"Adjusted A shape: {A_raw.shape}, b length: {len(b_raw)}")

    print("Generating reference solution with h=0.0001...")
    t_values_ref, y_values_ref = runge_kutta_general(
        brusselator_1d, t_span, y0, 0.0001, A_raw, b_raw,
        args=(N, dx, D_u, D_v, a_param, b_param)
    )
    print("Reference solution generated.")

    # Convergence study
    errors = []
    for h in hs:
        t_values, y_values = runge_kutta_general(
            brusselator_1d, t_span, y0, h, A_raw, b_raw,
            args=(N, dx, D_u, D_v, a_param, b_param)
        )
        errors.append(l2_norm_error(y_values[-1], y_values_ref[-1]))

    orders = calculate_order_of_convergence(errors, hs)

    plt.figure()
    plt.loglog(hs, errors, 'o-', label='Numerical Error')
    # Plot a reference line for 3rd-order
    expected_order = 3
    plt.loglog(
        hs,
        [errors[0] * (h_ / hs[0])**expected_order for h_ in hs],
        'k--', label=f'{expected_order}rd-Order Reference'
    )
    plt.xlabel('Time Step Size (Δt)')
    plt.ylabel('Error Norm')
    plt.title('Convergence Study 2D Brusselator 3rd 15 stage')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()


