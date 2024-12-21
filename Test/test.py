import time
from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Union, Dict

import jax
# not allocate all gpu memory
import os

import numpy as np
from jax import profiler
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # may affect performance
import optax
from flax import linen as nn, jax_utils
from flax.core.frozen_dict import freeze
from flax.training import train_state

from jax import random, jit, vmap, lax, tree, pmap, grad, jacrev
import jax.numpy as jnp
from jax._src.flatten_util import ravel_pytree
from jax._src.tree_util import tree_reduce, tree_map, tree_leaves, tree_map_with_path
from jax.nn.initializers import glorot_normal, normal, zeros, constant, glorot_uniform
import ml_collections
from matplotlib import pyplot as plt

import jaxfit
from examples.ldc.utils import sample_points_on_square_boundary, get_dataset
from jaxpi.archs import _weight_fact
from jaxpi.evaluator import BaseEvaluator
from jaxpi.logging import Logger
from jaxpi.samplers import UniformSampler
from jaxpi.utils import ntk_fn

from line_profiler import profile, LineProfiler

class TrainState(train_state.TrainState):
    weights: Dict
    momentum: float

    def apply_weights(self, weights, **kwargs):
        """Updates `weights` using running average  in return value.

        Returns:
          An updated instance of `self` with new weights updated by applying `running_average`,
          and additional attributes replaced as specified by `kwargs`.
        """

        running_average = (
            lambda old_w, new_w: old_w * self.momentum + (1 - self.momentum) * new_w
        )
        weights = tree.map(running_average, self.weights, weights)
        weights = lax.stop_gradient(weights)

        return self.replace(
            step=self.step,
            params=self.params,
            opt_state=self.opt_state,
            weights=weights,
            **kwargs,
        )


activation_fn = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "swish": nn.swish,
    "sigmoid": nn.sigmoid,
    "tanh": jnp.tanh,
    "sin": jnp.sin,
}

def _get_activation(str):
    if str in activation_fn:
        return activation_fn[str]
    else:
        raise NotImplementedError(f"Activation {str} not supported yet!")

def flatten_pytree(pytree):
    return ravel_pytree(pytree)[0]

class DenseRP(nn.Module):
    features: int
    kernel_init: Callable = jax.nn.initializers.variance_scaling(1, "fan_avg", "uniform")
    bias_init: Callable = zeros
    reparam: Union[None, Dict] = None

    @nn.compact
    def __call__(self, x):
        if self.reparam is None:
            kernel = self.param(
                "kernel", self.kernel_init, (x.shape[-1], self.features)
            )

        elif self.reparam["type"] == "weight_fact":
            g, v = self.param(
                "kernel",
                _weight_fact(
                    self.kernel_init,
                    mean=self.reparam["mean"],
                    stddev=self.reparam["stddev"],
                ),
                (x.shape[-1], self.features),
            )
            kernel = g * v

        bias = self.param("bias", self.bias_init, (self.features,))

        y = jnp.dot(x, kernel) + bias

        return y

class Dense(nn.Module):
    features: int
    kernel_init: Callable = jax.nn.initializers.variance_scaling(1, "fan_in", "uniform")
    bias_init: Callable = zeros

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel", self.kernel_init, (x.shape[-1], self.features)
        )
        bias = self.param("bias", self.bias_init, (self.features,))
        y = jnp.dot(x, kernel) + bias
        return y

class FourierEmbs(nn.Module):
    embed_scale: float
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel", normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2)
        )
        y = jnp.concatenate(
            [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
        )
        return y


class MLP(nn.Module):
    arch_name: Optional[str] = "MLP"
    num_layers: int = 5
    hidden_dim: int = 512
    out_dim: int = 3
    activation: str = "sin"
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        # 设置激活函数
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        if self.fourier_emb:
           x = FourierEmbs(**self.fourier_emb)(x)

        for _ in range(self.num_layers):
            x = DenseRP(features=self.hidden_dim, reparam=self.reparam)(x)
            x = self.activation_fn(x)
        x = Dense(features=self.out_dim)(x)
        return x

    @nn.compact
    def subspace_call(self, x):
        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)

        for _ in range(self.num_layers):
            x = DenseRP(features=self.hidden_dim, reparam=self.reparam)(x)
            x = self.activation_fn(x)
        return x


def _create_optimizer(config):
    lr = optax.exponential_decay(
        init_value=config.learning_rate,
        transition_steps=config.decay_steps,
        decay_rate=config.decay_rate,
    )
    tx = optax.adam(
        learning_rate=lr, b1=config.beta1, b2=config.beta2, eps=config.eps
    )

    return tx

def _create_train_state(config):
    # Initialize network
    arch = MLP(**config.arch)
    x = jnp.ones((1, config.input_dim))
    params = arch.init(random.PRNGKey(config.seed), x)

    # Initialize optax optimizer
    tx = _create_optimizer(config.adam)


    # Convert config dict to dict
    init_weights = dict(config.weighting.init_weights)

    state = TrainState.create(
        apply_fn=arch.apply,
        params=params,
        tx=tx,
        weights=init_weights,
        momentum=config.weighting.momentum,
    )

    return state

config = ml_collections.ConfigDict()
config.input_dim = 2
config.seed = 42

# Arch
config.arch = arch = ml_collections.ConfigDict()
arch.arch_name = "MLP"
arch.num_layers = 2
arch.hidden_dim = 2048
arch.out_dim = 3
arch.activation = "sin"
arch.fourier_emb = ml_collections.ConfigDict(
    {"embed_scale": 10.0, "embed_dim": 50}
)

arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}
    )

config.adam = adam = ml_collections.ConfigDict()
adam.beta1 = 0.9
adam.beta2 = 0.999
adam.eps = 1e-8
adam.learning_rate = 2e-3
adam.decay_rate = 0.9
adam.decay_steps = 500
adam.grad_accum_steps = 0

# Weighting
config.weighting = weighting = ml_collections.ConfigDict()
weighting.scheme = "grad_norm" #"ntk" #"grad_norm"
weighting.init_weights = ml_collections.ConfigDict(
    {
        #"u_bc": 1.0,
        #"v_bc": 1.0,
        "ru": 1.0,
        "rv": 1.0,
        "rc": 1.0}
)
weighting.momentum = 0.9
weighting.update_every_steps = 100
config.logging = logging = ml_collections.ConfigDict()
logging.log_every_steps = 500
logging.log_errors = True
logging.log_losses = True
logging.log_weights = False
logging.log_grads = False
logging.log_ntk = False
logging.log_preds = True



class PINN:
    def __init__(self, config):
        self.config = config
        self.state = _create_train_state(config)
        self.num_layers = config.arch.num_layers

    def u_net(self, params, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def r_net(self, params, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def losses(self, params, batch, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def compute_diag_ntk(self, params, batch, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def is_last_layer_param(self, path):
        path_str = '/'.join(str(p) for p in path)
        return f"Dense_0" in path_str

    #@partial(jit, static_argnums=(0,))
    def filter_grads(self, grads):
        def filter_fn(path):
            # Return 0 for last layer params, 1 otherwise
            return 0.0 if self.is_last_layer_param(path) else 1.0

        # Create a mask with the same structure as grads
        grad_mask = tree_map_with_path(lambda p, _: filter_fn(p), grads)

        # Multiply gradients by mask (0 for last layer, 1 otherwise)
        filtered_grads = tree_map(lambda x, y: x * y, grads, grad_mask)
        return filtered_grads

    #@partial(jit, static_argnums=(0,))
    def loss(self, params, weights, batch, *args):
        # Compute losses
        losses = self.losses(params, batch, *args)
        # Compute weighted loss
        weighted_losses = tree_map(lambda x, y: x * y, losses, weights)
        # Sum weighted losses
        loss = tree_reduce(lambda x, y: x + y, weighted_losses)
        return loss

    #@partial(jit, static_argnums=(0,))
    def compute_weights(self, params, batch, *args):
        if self.config.weighting.scheme == "grad_norm":
            # Compute the gradient of each loss w.r.t. the parameters
            grads = jacrev(self.losses)(params, batch, *args)

            # Compute the grad norm of each loss
            grad_norm_dict = {}
            for key, value in grads.items():
                flattened_grad = flatten_pytree(value)
                grad_norm_dict[key] = jnp.linalg.norm(flattened_grad)

            # Compute the mean of grad norms over all losses
            mean_grad_norm = jnp.mean(jnp.stack(tree_leaves(grad_norm_dict)))
            # Grad Norm Weighting
            w = tree_map(lambda x: (mean_grad_norm / x), grad_norm_dict)

        elif self.config.weighting.scheme == "ntk":
            # Compute the diagonal of the NTK of each loss
            ntk = self.compute_diag_ntk(params, batch, *args)

            # Compute the mean of the diagonal NTK corresponding to each loss
            mean_ntk_dict = tree_map(lambda x: jnp.mean(x), ntk)

            # Compute the average over all ntk means
            mean_ntk = jnp.mean(jnp.stack(tree_leaves(mean_ntk_dict)))
            # NTK Weighting
            w = tree_map(lambda x: (mean_ntk / x), mean_ntk_dict)

        return w

    #@partial(jit, static_argnums=(0,))
    def update_weights(self, state, batch, *args):
        weights = self.compute_weights(state.params, batch, *args)
        state = state.apply_weights(weights=weights)
        return state

    #@partial(jit, static_argnums=(0,))
    def step(self, state, batch, *args):
        grads = grad(self.loss)(state.params, state.weights, batch, *args)
        #filtered_grads = self.filter_grads(grads)
        state = state.apply_gradients(grads=grads)
        return state


class ForwardBVP(PINN):
    def __init__(self, config):
        super().__init__(config)


class NavierStokes2D(ForwardBVP):
    def __init__(self, config):
        super().__init__(config)

        # Sample boundary points uniformly
        self.num_pts = 256
        self.x_bc1 = sample_points_on_square_boundary(
            self.num_pts, eps=0.01
        )  # avoid singularity a right corner for u velocity
        self.x_bc2 = sample_points_on_square_boundary(self.num_pts, eps=0.01)

        # Boundary conditions
        self.v_bc = jnp.zeros((self.num_pts * 4,))
        self.u_bc = self.v_bc.at[:self.num_pts].set(1.0)

        # Predictions over a grid
        self.u_pred_fn = vmap(self.u_net, (None, 0, 0))
        self.v_pred_fn = vmap(self.v_net, (None, 0, 0))
        self.p_pred_fn = vmap(self.p_net, (None, 0, 0))
        self.r_pred_fn = vmap(self.r_net, (None, None, 0, 0))
        self.n_fn = vmap(self.neural_net, (None, 0, 0))
        #self.subspace_n_fn = vmap(self.subspace_neural_net, (None, 0, 0))
        self.subspace_r_pred_fn = vmap(self.subspace_r_net, (None, None, 0, 0))

        self.opt = jaxfit.LeastSquares()
        self.v = []

        self.lv, self.lv_x, self.lv_y, self.lv_xx, self.lv_yy = None, None, None, None, None
        self.blv = None

    @partial(jit, static_argnums=(0,))
    def neural_net(self, params, x, y):
        z = jnp.stack([x, y])
        outputs = self.state.apply_fn(params, z)
        u = outputs[0]
        v = outputs[1]
        p = outputs[2]
        return u, v, p

    @partial(jit, static_argnums=(0,))
    def subspace_neural_net(self, params, x, y):
        z = jnp.stack([x, y])
        return self.state.apply_fn(params, z, method=MLP.subspace_call)

    @partial(jit, static_argnums=(0,))
    def u_net(self, params, x, y):
        u, _, _ = self.neural_net(params, x, y)
        return u

    @partial(jit, static_argnums=(0,))
    def v_net(self, params, x, y):
        _, v, _ = self.neural_net(params, x, y)
        return v

    @partial(jit, static_argnums=(0,))
    def p_net(self, params, x, y):
        _, _, p = self.neural_net(params, x, y)
        return p

    @partial(jit, static_argnums=(0,))
    def r_net(self, params, nu, x, y):
        def fu_x(x, y):
            return jax.jvp(lambda x, y: self.u_net(params, x, y), (x, y), (1.0, 0.0))

        def fu_y(x, y):
            return jax.jvp(lambda x, y: self.u_net(params, x, y), (x, y), (0.0, 1.0))

        def fv_x(x, y):
            return jax.jvp(lambda x, y: self.v_net(params, x, y), (x, y), (1.0, 0.0))

        def fv_y(x, y):
            return jax.jvp(lambda x, y: self.v_net(params, x, y), (x, y), (0.0, 1.0))

        def fp_x(x, y):
            return jax.jvp(lambda x, y: self.p_net(params, x, y), (x, y), (1.0, 0.0))

        def fp_y(x, y):
            return jax.jvp(lambda x, y: self.p_net(params, x, y), (x, y), (0.0, 1.0))

        def fu_xx(x, y):
            return jax.jvp(lambda x, y: fu_x(x, y)[1], (x, y), (1.0, 0.0))[1]

        def fu_yy(x, y):
            return jax.jvp(lambda x, y: fu_y(x, y)[1], (x, y), (0.0, 1.0))[1]

        def fv_xx(x, y):
            return jax.jvp(lambda x, y: fv_x(x, y)[1], (x, y), (1.0, 0.0))[1]

        def fv_yy(x, y):
            return jax.jvp(lambda x, y: fv_y(x, y)[1], (x, y), (0.0, 1.0))[1]

        _, u_x = fu_x(x, y)
        u, u_y = fu_y(x, y)
        _, v_x = fv_x(x, y)
        v, v_y = fv_y(x, y)
        _, p_x = fp_x(x, y)
        p, p_y = fp_y(x, y)
        u_xx = fu_xx(x, y)
        u_yy = fu_yy(x, y)
        v_xx = fv_xx(x, y)
        v_yy = fv_yy(x, y)

        ru = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        rv = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        rc = u_x + v_y

        return ru, rv, rc

    @partial(jit, static_argnums=(0,))
    def subspace_r_net(self, params, nu, x, y):
        def f_lv_x(x, y):
            tangents_x = jnp.ones_like(x)
            tangents_y = jnp.zeros_like(y)
            return jax.jvp(lambda x, y: self.subspace_neural_net(params, x, y), (x, y), (tangents_x, tangents_y))

        def f_lv_y(x, y):
            tangents_x = jnp.zeros_like(x)
            tangents_y = jnp.ones_like(y)
            return jax.jvp(lambda x, y: self.subspace_neural_net(params, x, y), (x, y), (tangents_x, tangents_y))

        def f_lv_xx(x, y):
            tangents_x = jnp.ones_like(x)
            tangents_y = jnp.zeros_like(y)
            return jax.jvp(lambda x, y: f_lv_x(x, y)[1], (x, y), (tangents_x, tangents_y))[1]

        def f_lv_yy(x, y):
            tangents_x = jnp.zeros_like(x)
            tangents_y = jnp.ones_like(y)
            return jax.jvp(lambda x, y: f_lv_y(x, y)[1], (x, y), (tangents_x, tangents_y))[1]


        lv, lv_x = f_lv_x(x, y)
        _, lv_y = f_lv_y(x, y)
        lv_xx = f_lv_xx(x, y)
        lv_yy = f_lv_yy(x, y)

        return lv, lv_x, lv_y, lv_xx, lv_yy

    def ru_net(self, params, nu, x, y):
        ru, _, _ = self.r_net(params, nu, x, y)
        return ru

    def rv_net(self, params, nu, x, y):
        _, rv, _ = self.r_net(params, nu, x, y)
        return rv

    def rc_net(self, params, nu, x, y):
        _, _, rc = self.r_net(params, nu, x, y)
        return rc

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch, nu):
        # boundary condition losses
        # Compute forward pass of u and v
        u_pred = self.u_pred_fn(params, self.x_bc1[:, 0], self.x_bc1[:, 1])
        v_pred = self.v_pred_fn(params, self.x_bc2[:, 0], self.x_bc2[:, 1])

        # Compute losses
        #u_bc_loss = jnp.mean((u_pred - self.u_bc) ** 2)
        #v_bc_loss = jnp.mean(v_pred**2)

        # Compute forward pass of residual
        ru_pred, rv_pred, rc_pred = self.r_pred_fn(params, nu, batch[:, 0], batch[:, 1])
        # Compute losses
        ru_loss = jnp.mean(ru_pred**2)
        rv_loss = jnp.mean(rv_pred**2)
        rc_loss = jnp.mean(rc_pred**2)

        loss_dict = {
            #"u_bc": u_bc_loss,
            #"v_bc": v_bc_loss,
            "ru": ru_loss,
            "rv": rv_loss,
            "rc": rc_loss,
        }

        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch, nu):
        u_bc_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.u_net, params, self.x_bc1[:, 0], self.x_bc1[:, 1]
        )
        v_bc_ntk = vmap(ntk_fn, (None, None, 0, 0))(
            self.v_net, params, self.x_bc2[:, 0], self.x_bc2[:, 1]
        )

        ru_ntk = vmap(ntk_fn, (None, None, None, 0, 0))(
            self.ru_net, params, nu, batch[:, 0], batch[:, 1]
        )
        rv_ntk = vmap(ntk_fn, (None, None, None, 0, 0))(
            self.rv_net, params, nu, batch[:, 0], batch[:, 1]
        )
        rc_ntk = vmap(ntk_fn, (None, None, None, 0, 0))(
            self.rc_net, params, nu, batch[:, 0], batch[:, 1]
        )

        ntk_dict = {
            #"u_bc": u_bc_ntk,
            #"v_bc": v_bc_ntk,
            "ru": ru_ntk,
            "rv": rv_ntk,
            "rc": rc_ntk,
        }

        return ntk_dict

    @partial(jit, static_argnums=(0,))
    def update_weights(self, state, batch, nu):
        weights = self.compute_weights(state.params, batch, nu)
        state = state.apply_weights(weights=weights)
        return state

    @partial(jit, static_argnums=(0,))
    def step(self, state, batch, nu):
        grads = grad(self.loss)(state.params, state.weights, batch, nu)
        #filtered_grads = self.filter_grads(grads)
        state = state.apply_gradients(grads=grads)
        return state

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, x_star, y_star, U_test):
        u_pred = vmap(vmap(self.u_net, (None, None, 0)), (None, 0, None))(
            params, x_star, y_star
        )
        v_pred = vmap(vmap(self.v_net, (None, None, 0)), (None, 0, None))(
            params, x_star, y_star
        )

        U_pred = jnp.sqrt(u_pred**2 + v_pred**2)
        l2_error = jnp.linalg.norm(U_pred - U_test) / jnp.linalg.norm(U_test)

        return l2_error

    def replace_last_layer_params(self, params, new_weights):
        def replace_fn(path, param):
            path_str = '/'.join(str(p) for p in path)
            if f"Dense_0" in path_str:
                # If this is the last hidden layer parameter, return the new weights
                if param.shape == new_weights.shape:
                    return new_weights
            return param

        return tree_map_with_path(replace_fn, params)

    @partial(jit, static_argnums=(0,))
    def calculate_lv(self, state, batch, nu):
        lv, lv_x, lv_y, lv_xx, lv_yy = self.subspace_r_pred_fn(state.params, nu, batch[:, 0], batch[:, 1])
        blv, _, _, _, _ = self.subspace_r_pred_fn(state.params, nu, self.x_bc1[:, 0], self.x_bc1[:, 1])
        return lv, lv_x, lv_y, lv_xx, lv_yy, blv

    @partial(jit, static_argnums=(0,))
    def cost(self, weights, state, batch, nu):
        weights = weights.reshape(state.params['params'][f'Dense_0']['kernel'].shape)
        wu, wv, wp = weights[:, 0].reshape(-1, 1), weights[:, 1].reshape(-1,1), weights[:, 2].reshape(-1,1)
        
        # Pre-compute common terms to avoid redundant calculations
        lv_wu = jnp.matmul(self.lv, wu)
        lv_wv = jnp.matmul(self.lv, wv)

        # Compute components with minimal intermediate tensors
        ru = (lv_wu * jnp.matmul(self.lv_x, wu) + 
              lv_wv * jnp.matmul(self.lv_y, wu) + 
              jnp.matmul(self.lv_x, wp) - 
              nu * (jnp.matmul(self.lv_xx, wu) + jnp.matmul(self.lv_yy, wu)))

        rv = (lv_wu * jnp.matmul(self.lv_x, wv) + 
              lv_wv * jnp.matmul(self.lv_y, wv) + 
              jnp.matmul(self.lv_y, wp) - 
              nu * (jnp.matmul(self.lv_xx, wv) + jnp.matmul(self.lv_yy, wv)))

        rc = jnp.matmul(self.lv_x, wu) + jnp.matmul(self.lv_y, wv)
        bu = jnp.matmul(self.blv, wu) - self.u_bc.reshape(-1, 1)
        bv = jnp.matmul(self.blv, wv)
    
        return jnp.concatenate([ru, rv, rc, bu, bv], axis=0).reshape(-1)

    def monitor(self, weights, state, batch, nu):
        weights = weights.reshape(state.params['params'][f'Dense_0']['kernel'].shape)
        wu, wv, wp = weights[:, 0].reshape(-1, 1), weights[:, 1].reshape(-1, 1), weights[:, 2].reshape(-1, 1)
        lvb, _, _, _, _ = self.subspace_r_pred_fn(state.params, nu, self.x_bc1[:, 0], self.x_bc1[:, 1])
        return float(jax.device_get(jnp.mean((lvb @ wu)[:self.num_pts])))

    def optimize(self, state, batch, nu):
        weights = jax.device_get(state.params['params'][f'Dense_0']['kernel'])
        flat_weights = weights.flatten()
        self.lv, self.lv_x, self.lv_y, self.lv_xx, self.lv_yy, self.blv = self.calculate_lv(state, batch, nu)
        opt = self.opt.least_squares(lambda w: self.cost(w, state, batch, nu)
                                     , flat_weights, verbose=2, method='trf', max_nfev=100,
                                    tr_options={'method': 'exact'})
        optimal_weights, cost = opt.x, opt.cost

        self.monitor(optimal_weights, state, batch, nu)

        print(f"Cost: {cost}")
        return optimal_weights.reshape(weights.shape)

    def optimize_NLSQ_perturb(self, state, batch, nu, delta=0.5, iter=5, eval=50):
        weights = jax.device_get(state.params['params'][f'Dense_0']['kernel'])
        flat_weights = weights.flatten()
        self.calculate_lv(state, batch, nu)
        opt = self.opt.least_squares(lambda w: self.cost(w, state, batch, nu)
                                     , flat_weights, verbose=2, method='trf', max_nfev=eval,
                                    tr_options={'method': 'exact'})
        optimal_weights, cost = opt.x, opt.cost

        if cost >= 1e-13:
            for i in range(iter):
                key1, key2 = random.split(random.PRNGKey(0))
                delta1 = delta * random.uniform(key1)
                optimal_weights = (1-delta1) * optimal_weights + delta1 * 2 * (random.uniform(key2, optimal_weights.shape) - 1/2)
                opt = self.opt.least_squares(lambda w: self.cost(w, state, batch, nu), optimal_weights, verbose=2, method='trf', max_nfev=eval,
                                    tr_options={'method': 'exact'})
                optimal_weights, cost = opt.x, opt.cost
                print(f"Cost: {cost}")

        return optimal_weights.reshape(weights.shape)


class NavierStokesEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, x_star, y_star, U_ref):
        l2_error = self.model.compute_l2_error(params, x_star, y_star, U_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params, x_star, y_star, U_ref):
        u_pred = vmap(vmap(self.model.u_net, (None, None, 0)), (None, 0, None))(
            params, x_star, y_star
        )
        v_pred = vmap(vmap(self.model.v_net, (None, None, 0)), (None, 0, None))(
            params, x_star, y_star
        )
        U_pred = jnp.sqrt(u_pred**2 + v_pred**2)

        fig, ax = plt.subplots(3, 1, figsize=(5, 16), dpi=300)
        vmin = U_ref.min()
        vmax = U_ref.max()
        levels = np.linspace(vmin, vmax, 100)
        c1 = ax[0].contourf(x_star, y_star, U_ref.T, cmap="viridis", levels=levels)
        ax[0].set_title("Reference")
        ax[0].set_aspect('equal', 'box')
        fig.colorbar(c1, ax=ax[0], ticks=np.linspace(vmin, vmax, 10))
        c2 = ax[1].contourf(x_star, y_star, U_pred.T, cmap="viridis", levels=levels)
        ax[1].set_title("Prediction")
        ax[1].set_aspect('equal', 'box')
        fig.colorbar(c2, ax=ax[1], ticks=np.linspace(vmin, vmax, 10))
        c3 = ax[2].contourf(x_star, y_star, U_ref.T - U_pred.T, cmap="rainbow", levels=np.linspace(-0.25, 0.25, 100))
        ax[2].set_title("Error")
        ax[2].set_aspect('equal', 'box')
        fig.colorbar(c3, ax=ax[2], ticks=np.linspace(-0.5, 0.5, 10))
        plt.show()
        # fig.close()

    def __call__(self, state, batch, x_star, y_star, U_ref, nu):
        self.log_dict = super().__call__(state, batch, nu)

        if self.config.logging.log_errors:
            self.log_errors(state.params, x_star, y_star, U_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params, x_star, y_star, U_ref)

        return self.log_dict



def loop(model):
    u_ref, v_ref, x_star, y_star, nu = get_dataset(100)
    nu = nu[0,0]
    U_ref = jnp.sqrt(u_ref ** 2 + v_ref ** 2)

    x0 = x_star[0]
    x1 = x_star[-1]

    y0 = y_star[0]
    y1 = y_star[-1]

    # Define domain
    dom = jnp.array([[x0, x1], [y0, y1]])

    # Initialize  residual sampler
    res_sampler = iter(UniformSampler(dom, 2048))
    logger = Logger()
    evaluator = NavierStokesEvaluator(config, model)

    batch = next(res_sampler)[0]

    start_time = time.time()
    steps = 2000
    for step in range(steps+1):

        model.state = model.step(model.state, batch, nu)

        # Update weights if necessary
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            model.state = model.update_weights(model.state, batch, nu)

        if step % config.logging.log_every_steps == 0:
            log_dict = evaluator(model.state, batch, x_star, y_star, U_ref, nu)
            end_time = time.time()
            logger.log_iter(step, start_time, end_time, log_dict)
            start_time = end_time

    start_time = time.time()
    weights = model.optimize(model.state, batch, nu)
    #weights = model.optimize_NLSQ_perturb(model.state, batch, nu, delta=0.5, iter=5, eval=50)
    new_params = model.replace_last_layer_params(model.state.params, weights)
    model.state = model.state.replace(params=new_params)
    end_time = time.time()
    log_dict = evaluator(model.state, batch, x_star, y_star, U_ref, nu)
    logger.log_iter(steps+1, start_time, end_time, log_dict)

if __name__ == '__main__':
    # lp = LineProfiler()
    # model = NavierStokes2D(config)
    #
    # lpw = lp(loop)
    # #lp.add_function(model.cost)
    # lp.add_function(model.step.__wrapped__)
    # lp.add_function(model.losses.__wrapped__)
    # lp.add_function(model.r_net.__wrapped__)
    # lpw(model)
    # lp.print_stats()
    #profiler.start_trace(log_dir='./tmp')
    model = NavierStokes2D(config)
    loop(model)
    
    #profiler.stop_trace()
    print(model.v)
