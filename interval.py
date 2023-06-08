import numpy as np
from scipy import linalg as scla
from numpy import pi as PI, sin, cos
from matplotlib import pyplot as plt

L = PI # interval length

# size of FEM space (number of grid points - 1)
N_h = 2 ** 3
# N_h = 2 ** 4
# N_h = 2 ** 5

# basis functions
x_h = np.linspace(0, 1, N_h + 1)
plt.figure()
plt.plot(x_h, np.eye(N_h + 1)[:, 1:-1])
plt.xlabel("x"), plt.ylabel("phi(x)")

plt.figure()
plt.plot(x_h, np.pad(np.random.rand(N_h - 1, 3), [(1, 1), (0, 0)]))
plt.xlabel("x"), plt.ylabel("some random f(x)")
plt.show()

#%% system
A = (2 * np.eye(N_h - 1) - np.diag(np.ones(N_h - 2), 1)
                         - np.diag(np.ones(N_h - 2), -1)) * N_h / L
B = (2./3 * np.eye(N_h - 1) + 1./6 * np.diag(np.ones(N_h - 2), 1)
                            + 1./6 * np.diag(np.ones(N_h - 2), -1)) * L / N_h
if N_h < 20:
    print("A:\n", np.trunc(A * 100)/100) # print A in nice format
    print("B:\n", np.trunc(B * 100)/100) # print B in nice format
eigs = scla.eigvalsh(A, B)
print("approx. eigenvalues:\n", eigs)

#%% closed-form expressions
# define exact eigenvalues and eigenfunctions
gen_vals = lambda n: (n * PI / L) ** 2
gen_vecs = lambda x, n: (2 * L) ** .5 / n / PI * sin(n * PI / L * x)

lambda_max = 1e4
n, vals_exact = 1, []
while not len(vals_exact) or vals_exact[-1] < lambda_max:
    vals_exact += [gen_vals(n)]
    n += 1

def get_vals(N_h, lambda_max = np.inf):
    # this is the closed-form expression of the numerical eigenvalues
    vals_app = []
    for n in range(1, N_h):
        vals_app += [3 * (N_h / L) ** 2 * (1 - cos(n * PI / N_h))
                                        / (1 + .5 * cos(n * PI / N_h))]
        if vals_app[-1] >= lambda_max: break
    return vals_app

def H10norm(N_h, vec):
    return (N_h / L * vec[1:-1].dot(- vec[:-2] + 2 * vec[1:-1] - vec[2:])) ** .5

def get_valsvecs(N_h, lambda_max = np.inf):
    # this is the closed-form expression of the numerical eigenvectors
    x_h = np.linspace(0., L, N_h + 1)
    vals_app, vecs_app = [], []
    for n in range(1, N_h):
        vals_app += [3 * (N_h / L) ** 2 * (1 - cos(n * PI / N_h))
                                        / (1 + .5 * cos(n * PI / N_h))]
        vec = sin(n * PI / L * x_h)
        vecs_app += [vec / H10norm(N_h, vec)]
        if vals_app[-1] >= lambda_max: break
    return vals_app, vecs_app

#%% vals
idxs_plot = [6, 25]
plt.figure(figsize = (5 * len(idxs_plot), 4))
vals_app = get_vals(N_h)
for j, idx_plot in enumerate(idxs_plot):
    plt.subplot(1, len(idxs_plot), 1 + j)
    plt.plot(1 + np.arange(len(vals_exact)), vals_exact, 'kx')
    plt.plot(1 + np.arange(len(vals_app)), vals_app, 'b+')
    plt.xlim(.5, idx_plot+.5), plt.ylim(0, vals_exact[idx_plot+1])
    if j == 0: plt.ylabel('eigenvalues')
    plt.xlabel('index'), plt.legend(['exact', 'approx'])
plt.tight_layout(), plt.show()

#%% vecs
idxs_plot = list(range(N_h - 1))
plt_j = int(len(idxs_plot) ** .5)
plt_i = int(np.ceil(len(idxs_plot) / plt_j))
plt.figure(figsize = (5 * plt_j, 4 * plt_i))
vecs_app = get_valsvecs(N_h)[1]
x_h = np.linspace(0, L, N_h + 1)
xx = np.linspace(0, L, 10001)
for j, idx_plot in enumerate(idxs_plot):
    plt.subplot(plt_i, plt_j, 1 + j)
    plt.plot(x_h, vecs_app[idx_plot], 'r')
    plt.plot(xx, gen_vecs(xx, idx_plot + 1), 'k:')
    plt.title("eigval - exact:{}, approx:{}".format(gen_vals(idx_plot + 1),
                                                    vals_app[idx_plot]))
    plt.xlabel(" "), plt.legend(['exact', 'approx'])
plt.tight_layout(), plt.show()
