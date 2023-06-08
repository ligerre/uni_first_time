import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from matplotlib import pyplot as plt
import fenics as fen, mshr # note: FEniCS must be installed!

example = "disk"
# example = "disk_quadrant"
# example = "disk_noquadrant"
# example = "disk_hole"
# example = "ring"
# example = "moon"

# define mesh, function space
if example == "disk":
    mesh = mshr.generate_mesh(mshr.Circle(fen.Point(0., 0.), 1.), 25)
if example == "disk_hole":
    mesh = mshr.generate_mesh(mshr.Circle(fen.Point(0., 0.), 1.)
                            - mshr.Rectangle(fen.Point(-.5, -.3),
                                             fen.Point(.2, .7)), 35)
if example == "disk_quadrant":
    mesh = mshr.generate_mesh(mshr.Circle(fen.Point(0., 0.), 1.)
                            - mshr.Rectangle(fen.Point(-1., -1.),
                                             fen.Point(0., 1.))
                            - mshr.Rectangle(fen.Point(-1., -1.),
                                             fen.Point(1., 0.)), 25)
if example == "disk_noquadrant":
    mesh = mshr.generate_mesh(mshr.Circle(fen.Point(0., 0.), 1.)
                            - mshr.Rectangle(fen.Point(-1., -1.),
                                             fen.Point(0., 0.)), 35)
if example == "ring":
    mesh = mshr.generate_mesh(mshr.Circle(fen.Point(0., 0.), 1.)
                            - mshr.Circle(fen.Point(0., 0.), .4), 30)
if example == "moon":
    mesh = mshr.generate_mesh(mshr.Circle(fen.Point(0., 0.), 1.)
                            - mshr.Circle(fen.Point(-.4, .1), .6), 30)
V = fen.FunctionSpace(mesh, "P", 1)

# plot mesh
plt.figure(figsize = (7, 7))
fen.plot(mesh)
plt.jet(), plt.show()

# assemble bilinear forms
u = fen.TrialFunction(V)
v = fen.TestFunction(V)
a = fen.dot(fen.grad(u), fen.grad(v)) * fen.dx
b = u * v * fen.dx
Amat = fen.as_backend_type(fen.assemble(a)).mat()
Bmat = fen.as_backend_type(fen.assemble(b)).mat()
# convert to sparse scipy format
Ar, Ac, Av = Amat.getValuesCSR()
Br, Bc, Bv = Bmat.getValuesCSR()
A = csr_matrix((Av, Ac, Ar), shape = Amat.size, dtype = float)
B = csr_matrix((Bv, Bc, Br), shape = Bmat.size, dtype = float)

# BCs
Dbdr = lambda x, on_b: on_b
bad_idxs = fen.DirichletBC(V, 0, Dbdr).get_boundary_values()
good_idxs = [i for i in range(A.shape[0]) if i not in bad_idxs]
A, B = A[good_idxs][:, good_idxs], B[good_idxs][:, good_idxs]

plt.figure(figsize = (15, 15))
plt.subplot(221)
plt.spy(A, markersize = 1)
plt.title("spy A")
plt.subplot(222)
plt.spy(A[:50, :50], markersize = 4)
plt.title("spy A[:50, :50]")
plt.subplot(223)
plt.spy(B, markersize = 1)
plt.title("spy B")
plt.subplot(224)
plt.spy(B[:50, :50], markersize = 4)
plt.title("spy B[:50, :50]")
plt.show()

# compute eigenvalues
vals, vecs = eigsh(A, 21, B, which = 'SM')

# plot eigenvalues
plt.figure()
plt.plot(vals, 'o')
plt.xlabel("index"), plt.ylabel("eigenvalue")
plt.show()

# plot some eigenfunctions
u = fen.Function(V)
for val, vec in zip(vals, vecs.T):
    vecs_eff = np.zeros(V.dim())
    vecs_eff[good_idxs] = vec
    u.vector().set_local(vecs_eff)
    # Plot eigenfunction
    plt.figure(figsize = (7, 7))
    p = fen.plot(u)
    plt.title("eigval = {}".format(val))
    plt.colorbar(p)
    plt.show()
