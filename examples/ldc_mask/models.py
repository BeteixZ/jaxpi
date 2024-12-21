import io
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image
from jax._src.tree_util import tree_reduce

import wandb
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian, tree_map
from jax.flatten_util import ravel_pytree

from jaxpi.models import ForwardBVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn

from utils import sample_points_on_square_boundary

from matplotlib import pyplot as plt


class NavierStokes2D(ForwardBVP):
    def __init__(self, config):
        super().__init__(config)

        # Sample boundary points uniformly
        num_pts = 256
        self.x_bc1 = sample_points_on_square_boundary(
            num_pts, eps=0.01
        )  # avoid singularity a right corner for u velocity
        self.x_bc2 = sample_points_on_square_boundary(num_pts, eps=0.01)

        # Boundary conditions
        self.v_bc = jnp.zeros((num_pts * 4,))
        self.u_bc = self.v_bc.at[:num_pts].set(1.0)

        # Predictions over a grid
        self.u_pred_fn = vmap(self.u_net, (None, 0, 0))
        self.v_pred_fn = vmap(self.v_net, (None, 0, 0))
        self.p_pred_fn = vmap(self.p_net, (None, 0, 0))
        self.r_pred_fn = vmap(self.r_net, (None, None, 0, 0))
        self.r_pred_fn_ne = vmap(self.r_net, (None, None, 0, 0))

    def neural_net(self, params, x, y):
        z = jnp.stack([x, y])
        outputs = self.state.apply_fn(params, z)
        u = outputs[0]
        v = outputs[1]
        p = outputs[2]
        return u, v, p

    def u_net(self, params, x, y):
        u, _, _ = self.neural_net(params, x, y)
        return u

    def v_net(self, params, x, y):
        _, v, _ = self.neural_net(params, x, y)
        return v

    def p_net(self, params, x, y):
        _, _, p = self.neural_net(params, x, y)
        return p

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
    def losses_ne(self, params, batch, nu):
        """Modified loss function for masked training with proper normalization"""
        u_pred = self.u_pred_fn(params, self.x_bc1[:, 0], self.x_bc1[:, 1])
        v_pred = self.v_pred_fn(params, self.x_bc2[:, 0], self.x_bc2[:, 1])

        # Compute losses
        u_bc_loss = jnp.mean((u_pred - self.u_bc) ** 2)
        v_bc_loss = jnp.mean(v_pred**2)

        # Compute forward pass of residual
        ru_pred, rv_pred, rc_pred = self.r_pred_fn_ne(params, nu, batch[:, 0], batch[:, 1])
        # Compute losses
        ru_loss = jnp.mean(ru_pred**2)
        rv_loss = jnp.mean(rv_pred**2)
        rc_loss = jnp.mean(rc_pred**2)

        return {
            'u_bc': u_bc_loss,
            'v_bc': v_bc_loss,
            'ru': ru_loss,
            'rv': rv_loss,
            'rc': rc_loss
        }

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch, nu):
        # boundary condition losses
        # Compute forward pass of u and v
        u_pred = self.u_pred_fn(params, self.x_bc1[:, 0], self.x_bc1[:, 1])
        v_pred = self.v_pred_fn(params, self.x_bc2[:, 0], self.x_bc2[:, 1])

        # Compute losses
        u_bc_loss = jnp.mean((u_pred - self.u_bc) ** 2)
        v_bc_loss = jnp.mean(v_pred**2)

        # Compute forward pass of residual
        ru_pred, rv_pred, rc_pred = self.r_pred_fn(params, nu, batch[:, 0], batch[:, 1])
        # Compute losses
        ru_loss = jnp.mean(ru_pred**2)
        rv_loss = jnp.mean(rv_pred**2)
        rc_loss = jnp.mean(rc_pred**2)

        return {
            "u_bc": u_bc_loss,
            "v_bc": v_bc_loss,
            "ru": ru_loss,
            "rv": rv_loss,
            "rc": rc_loss,
        }

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
            "u_bc": u_bc_ntk,
            "v_bc": v_bc_ntk,
            "ru": ru_ntk,
            "rv": rv_ntk,
            "rc": rc_ntk,
        }

        return ntk_dict

    def compute_residual_norms(self, params, batch, nu):
        """Compute the residual norms for each point in the batch."""
        ru, rv, rc = vmap(self.r_net, (None, None, 0, 0))(params, nu, batch[:, 0], batch[:, 1])

        # Compute L2 norm of residuals at each point
        ru_norm = jnp.sqrt(ru ** 2)
        rv_norm = jnp.sqrt(rv ** 2)

        return ru_norm, rv_norm

    def identify_bad_subspace(self, params, batch, nu, threshold_percentile=90):
        """Identify points in the bad subspace based on residual norms."""
        ru_norm, rv_norm = self.compute_residual_norms(params, batch, nu)

        total_norm = ru_norm + rv_norm
        threshold = jnp.percentile(total_norm, threshold_percentile)

        combined_mask = total_norm > threshold

        return batch[combined_mask], combined_mask

    @partial(jit, static_argnums=(0,))
    def loss_ne(self, params, weights, batch, nu):
        # Compute losses
        losses = self.losses_ne(params, batch, nu)
        # Compute weighted loss
        weighted_losses = tree_map(lambda x, y: x * y, losses, weights)
        # Sum weighted losses
        loss = tree_reduce(lambda x, y: x + y, weighted_losses)
        return loss

    @partial(pmap, axis_name="batch_ne", static_broadcasted_argnums=(0, 3))
    def mask_step(self, state, batch_ne, nu):
        grads = grad(self.loss_ne)(state.params, state.weights, batch_ne, nu)
        grads = lax.pmean(grads, "batch_ne")
        state = state.apply_gradients(grads=grads)
        return state

    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0, 3, 4))
    def step(self, state, batch, nu, solver):
        # Check if the optimizer is L-BFGS by checking the name of the chain
        if solver == "lbfgs":
            # Flatten batch for each device
            batch = batch.reshape(-1, batch.shape[-1])

            # L-BFGS specific step
            value = self.loss(state.params, state.weights, batch, nu)
            grads = grad(self.loss)(state.params, state.weights, batch, nu)
            grads = lax.pmean(grads, "batch")

            updates, new_opt_state = state.tx.update(
                updates=grads,
                state=state.opt_state,
                params=state.params,
                value=value,
                grad=grads,
                value_fn=lambda p: self.loss(p, state.weights, batch, nu)
            )
            params = optax.apply_updates(state.params, updates)
            return state.replace(
                step=state.step + 1,
                params=params,
                opt_state=new_opt_state,
            )
        else:
            # ADAM step (original code)
            grads = grad(self.loss)(state.params, state.weights, batch, nu)
            grads = lax.pmean(grads, "batch")
            state = state.apply_gradients(grads=grads)
            return state

    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0, 3))
    def update_weights(self, state, batch, nu):
        weights = self.compute_weights(state.params, batch, nu)
        weights = lax.pmean(weights, "batch")
        state = state.apply_weights(weights=weights)
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


class NavierStokesEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, x_star, y_star, U_ref):
        l2_error = self.model.compute_l2_error(params, x_star, y_star, U_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params, x_star, y_star):
        u_pred = vmap(vmap(self.model.u_net, (None, None, 0)), (None, 0, None))(
            params, x_star, y_star
        )
        v_pred = vmap(vmap(self.model.v_net, (None, None, 0)), (None, 0, None))(
            params, x_star, y_star
        )
        U_pred = jnp.sqrt(u_pred ** 2 + v_pred ** 2)

        # Create mesh grid for proper plotting
        XX, YY = jnp.meshgrid(x_star, y_star, indexing='ij')

        # Create figure and plot
        fig, ax = plt.subplots(figsize=(8, 6))
        pc = ax.pcolormesh(XX, YY, U_pred, cmap="jet")
        plt.colorbar(pc, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Velocity Magnitude')

        # Save the plot to a temporary buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)

        # Convert to PIL Image
        image = Image.open(buf)

        # Create wandb Image from PIL Image
        self.log_dict["U_pred"] = wandb.Image(image, caption='Velocity Magnitude')

        # Close the buffer and figure
        buf.close()
        plt.close(fig)
        
    def log_residuals(self, params, nu, x_star, y_star):
        ru_pred = vmap(vmap(self.model.ru_net, (None, None, None, 0)), (None, None, 0, None))(
            params, nu, x_star, y_star
        )
        rv_pred = vmap(vmap(self.model.rv_net, (None, None, None, 0)), (None, None, 0, None))(
            params, nu, x_star, y_star
        )
        rc_pred = vmap(vmap(self.model.rc_net, (None, None, None, 0)), (None, None, 0, None))(
            params, nu, x_star, y_star
        )

        # Create mesh grid for proper plotting
        XX, YY = jnp.meshgrid(x_star, y_star, indexing='ij')

        # Create figure and plot
        fig, ax = plt.subplots(figsize=(8, 6))
        pc = ax.pcolormesh(XX, YY, ru_pred, cmap="jet", vmax= 2, vmin=-2)
        plt.colorbar(pc, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Residual ru')

        # Save the plot to a temporary buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)

        # Convert to PIL Image
        image = Image.open(buf)

        # Create wandb Image from PIL Image
        self.log_dict["ru_pred"] = wandb.Image(image, caption='Residual ru')

        # Close the buffer and figure
        buf.close()
        plt.close(fig)

        # Create figure and plot
        fig, ax = plt.subplots(figsize=(8, 6))
        pc = ax.pcolormesh(XX, YY, rv_pred, cmap="jet",vmax= 2, vmin=-2)
        plt.colorbar(pc, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Residual rv')

        # Save the plot to a temporary buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)

        # Convert to PIL Image
        image = Image.open(buf)

        # Create wandb Image from PIL Image
        self.log_dict["rv_pred"] = wandb.Image(image, caption='Residual rv')

        buf.close()
        plt.close(fig)

    def __call__(self, state, batch, x_star, y_star, U_ref, nu):
        self.log_dict = super().__call__(state, batch, nu)

        if self.config.logging.log_errors:
            self.log_errors(state.params, x_star, y_star, U_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params, x_star, y_star)

        if self.config.logging.log_residuals:
            self.log_residuals(state.params, nu, x_star, y_star)

        return self.log_dict

