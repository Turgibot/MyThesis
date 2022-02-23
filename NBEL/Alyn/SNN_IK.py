import matplotlib.pyplot as plt
import numpy as np

import nengo
from nengo.processes import WhiteSignal
from nengo.dists import Distribution, UniformHypersphere
from scipy.special import beta, betainc, betaincinv
from scipy.linalg import svd

class Rd(Distribution):

    def __repr__(self):
        return "%s()" % (type(self).__name__)

    def sample(self, n, d=1, rng=np.random):
        """Samples ``n`` points in ``d`` dimensions."""
        if d == 1:
            # Tile the points optimally. 
            return np.linspace(1.0 / n, 1, n)[:, None]
        if d is None or not isinstance(d, (int, np.integer)) or d < 1:
            raise ValueError("d (%d) must be positive integer" % d)
        return _rd_generate(n, d)

class ScatteredHypersphere(UniformHypersphere):

    def __init__(self, surface, base=Rd()):
        super(ScatteredHypersphere, self).__init__(surface)
        self.base = base

    def __repr__(self):
        return "%s(surface=%r, base=%r)" % (
            type(self).__name__,
            self.surface,
            self.base
        )

    def sample(self, n, d, rng=np.random):
        """Samples ``n`` points in ``d`` dimensions."""
        if d == 1:
            return super(ScatteredHypersphere, self).sample(n, d, rng)

        if self.surface:
            samples = self.base.sample(n, d - 1, rng)
            radius = 1.0
        else:
            samples = self.base.sample(n, d, rng)
            samples, radius = samples[:, :-1], samples[:, -1:] ** (1.0 / d)

        mapped = spherical_transform(samples)

        # radius adjustment for ball versus sphere, and a random rotation
        rotation = random_orthogonal(d, rng=rng)
        return np.dot(mapped * radius, rotation)

class SphericalCoords(Distribution):

    def __init__(self, m):
        super(SphericalCoords, self).__init__()
        self.m = m

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.m)

    def sample(self, n, d=None, rng=np.random):
        """Samples ``n`` points in ``d`` dimensions."""
        shape = self._sample_shape(n, d)
        y = rng.uniform(size=shape)
        return self.ppf(y)

    def pdf(self, x):
        """Evaluates the PDF along the values ``x``."""
        return np.pi * np.sin(np.pi * x) ** (self.m - 1) / beta(self.m / 2.0, 0.5)

    def cdf(self, x):
        """Evaluates the CDF along the values ``x``."""
        y = 0.5 * betainc(self.m / 2.0, 0.5, np.sin(np.pi * x) ** 2)
        return np.where(x < 0.5, y, 1 - y)

    def ppf(self, y):
        """Evaluates the inverse CDF along the values ``x``."""
        y_reflect = np.where(y < 0.5, y, 1 - y)
        z_sq = betaincinv(self.m / 2.0, 0.5, 2 * y_reflect)
        x = np.arcsin(np.sqrt(z_sq)) / np.pi
        return np.where(y < 0.5, x, 1 - x)


def random_orthogonal(d, rng=None):

    rng = np.random if rng is None else rng
    m = UniformHypersphere(surface=True).sample(d, d, rng=rng)
    u, s, v = svd(m)
    return np.dot(u, v)

def _rd_generate(n, d, seed=0.5):

    def gamma(d, n_iter=20):
        """Newton-Raphson-Method to calculate g = phi_d."""
        x = 1.0
        for _ in range(n_iter):
            x -= (x ** (d + 1) - x - 1) / ((d + 1) * x ** d - 1)
        return x

    g = gamma(d)
    alpha = np.zeros(d)
    for j in range(d):
        alpha[j] = (1 / g) ** (j + 1) % 1

    z = np.zeros((n, d))
    z[0] = (seed + alpha) % 1
    for i in range(1, n):
        z[i] = (z[i - 1] + alpha) % 1

    return z

def spherical_transform(samples):

    samples = np.asarray(samples)
    samples = samples[:, None] if samples.ndim == 1 else samples
    coords = np.empty_like(samples)
    n, d = coords.shape

    # inverse transform method (section 1.5.2)
    for j in range(d):
        coords[:, j] = SphericalCoords(d - j).ppf(samples[:, j])

    # spherical coordinate transform
    mapped = np.ones((n, d + 1))
    i = np.ones(d)
    i[-1] = 2.0
    s = np.sin(i[None, :] * np.pi * coords)
    c = np.cos(i[None, :] * np.pi * coords)
    mapped[:, 1:] = np.cumprod(s, axis=1)
    mapped[:, :-1] *= c
    return mapped
    
encoders_dist = ScatteredHypersphere(surface=True)


def get_intercepts(n_neurons, dimensions):

    triangular = np.random.triangular(left=0.35, 
                                      mode=0.45, 
                                      right=0.55, 
                                      size=n_neurons)
                                      
    intercepts = nengo.dists.CosineSimilarity(dimensions + 2).ppf(1 - triangular)
    return intercepts

class SNN_IK :
    
    def __init__ (self, calc_T, calc_J, target_xyz, n_scale = 500):
        
        self.current_q  = np.zeros(5)
        self.error      = 10
        self.target_xyz = np.array(target_xyz)
        self.output     = np.zeros(5)
      
        self.calc_J = calc_J
        self.calc_T = calc_T
        
        print('Generating model')
        
        self.model = nengo.Network(seed=42)
        with self.model:

            q_in = nengo.Node(self.current_q)
            q_c = nengo.Ensemble(n_scale*5, dimensions=5, seed=42,
                                 intercepts=get_intercepts(n_scale*5, 5),
                                 encoders = encoders_dist.sample(n_scale*5, 5))
            nengo.Connection(q_in, q_c)

            q_t = nengo.Ensemble(n_scale*20, dimensions=5, seed=42,
                                 intercepts=get_intercepts(n_scale*20, 5),
                                 encoders = encoders_dist.sample(n_scale*20, 5))
            conn = nengo.Connection(q_c, q_t, synapse=0.01)

            xyz_t = nengo.Ensemble(n_scale*3, dimensions=3,seed=42,
                                   intercepts=get_intercepts(n_scale*3, 3),
                                   encoders = encoders_dist.sample(n_scale*3, 3))

            def q2xyz(q):
                t = self.calc_T(q)
                return t[0], t[1], t[2]

            nengo.Connection(q_t, xyz_t, function=q2xyz)

            xyz_in = nengo.Node(self.target_xyz)
            nengo.Connection(xyz_in, xyz_t, transform=-1)

            error_q = nengo.Ensemble(n_scale*8, dimensions=8,seed=42,
                                    intercepts=get_intercepts(n_scale*8, 8),
                                    encoders = encoders_dist.sample(n_scale*8, 8))

            def combine(error_q):
                J_x = self.calc_J(error_q[0:5])
                return np.dot(np.linalg.pinv(J_x), error_q[5:])

            nengo.Connection(q_t, error_q[0:5])
            nengo.Connection(xyz_t, error_q[5:])

            error_combined = nengo.Ensemble(n_scale*5, dimensions=5, seed=42,
                                            intercepts=get_intercepts(n_scale*5, 5),
                                            encoders = encoders_dist.sample(n_scale*5, 5))
            
            nengo.Connection(error_q, error_combined, function=combine, synapse=0.01)

            conn.learning_rule_type = nengo.PES(learning_rate=1e-3)
            nengo.Connection(error_combined, conn.learning_rule)

            def comp_error(error_combined):
                return np.sqrt(sum(np.power(error_combined, 2)))

            error_out = nengo.Node(size_in=1, size_out=0)
            nengo.Connection(error_combined, error_out, function=comp_error)
                   
            def output_func(t, x):
                self.output = np.copy(x)

            output = nengo.Node(output_func, size_in=5, size_out=0)
            nengo.Connection(q_t, output, synapse=0.1)
            
            def error_func(t, x):
                self.error = np.copy(x)

            err = nengo.Node(error_func, size_in=1, size_out=0)
            nengo.Connection(error_out, err, synapse=0.1)
                  
            self.q_c_probe = nengo.Probe(q_c, synapse=0.01)
            self.q_t_probe = nengo.Probe(q_t, synapse=0.1)
            self.xyz_in_probe = nengo.Probe(xyz_in, synapse=0.01)
            self.xyz_t_probe = nengo.Probe(xyz_t, synapse=0.01)
            self.error_q_probe = nengo.Probe(error_q, synapse=0.01)
            self.error_combined_probe = nengo.Probe(error_combined, synapse=0.01)

            self.N = self.model.n_neurons
        
        nengo.rc.set("decoder_cache", "enabled", "False")
        self.sim = nengo.Simulator(self.model, dt=0.001)
           
        print('Model generated with {} neurons'.format(self.N))
    
    def generate(self, current_q):
    
        self.current_q  = current_q
        self.sim.run(time_in_seconds = 1, progress_bar=False)
        return self.output, self.error
        
        
        
        