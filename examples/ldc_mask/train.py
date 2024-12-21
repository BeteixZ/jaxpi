import io
import time
import os

from PIL import Image
from absl import logging

import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import vmap, jacrev
from jax.tree_util import tree_map

from flax import jax_utils

import ml_collections
import matplotlib.pyplot as plt
from sympy.physics.vector.printing import params

import wandb
from jaxpi.models import TrainState, _create_optimizer

from jaxpi.samplers import UniformSampler, LHSSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
from utils import get_dataset




def train_curriculum(config, workdir, model, step_offset, max_steps, Re):
    # Get dataset
    u_ref, v_ref, x_star, y_star, nu = get_dataset(Re)
    U_ref = jnp.sqrt(u_ref**2 + v_ref**2)

    x0 = x_star[0]
    x1 = x_star[-1]

    y0 = y_star[0]
    y1 = y_star[-1]

    # Define domain
    dom = jnp.array([[x0, x1], [y0, y1]])

    # Initialize residual sampler
    res_sampler = iter(UniformSampler(dom, config.training.batch_size))

    # Initialize evaluator
    evaluator = models.NavierStokesEvaluator(config, model)

    # Initialize logger
    logger = Logger()

    # Update  viscosity
    nu = 1 / Re

    steps = 0

    if config.training.sequential_opt.use_sequential:
        # 1. ADAM Training Phase
        print("Starting ADAM training phase...")
        adam_steps = config.training.sequential_opt.adam_steps

        # Set ADAM optimizer
        model.state = _create_train_state_with_optimizer(config.optim.adam, model)
        start_time = time.time()
        print("Waiting for JIT...")

        # Calculate number of devices
        num_devices = jax.device_count()

        # Adjust batch size to be divisible by number of devices
        device_batch_size = config.optim.lbfgs.batch_size // num_devices
        total_batch_size = device_batch_size * num_devices

        # Create new sampler with adjusted batch size
        res_sampler = iter(LHSSampler(dom, total_batch_size))

        # Get batch and reshape for devices
        batch = next(res_sampler)
        # batch = batch.reshape(num_devices, device_batch_size, -1)

        for step in range(adam_steps):
            model.state = model.step(model.state, batch, nu, "adam")
            # Update weights if necessary
            if config.weighting.scheme in ["grad_norm", "ntk"]:
                if step % config.weighting.update_every_steps == 0:
                    model.state = model.update_weights(model.state, batch, nu)

            # Check if we should do masked training
            if step % config.masked_training.mask_training_interval == 0 and step != 0:
                print(f"Starting masked training at step {step}")

                # Get the first replica of state for mask computation
                local_state = jax.device_get(tree_map(lambda x: x[0], model.state))
                local_batch = jax.device_get(tree_map(lambda x: x[0], batch))

                # Compute mask using the current state
                batch_ne, mask = model.identify_bad_subspace(local_state.params, local_batch,
                                                             nu, config.masked_training.threshold_percentile)

                # Replicate mask for all devices
                batch_ne = jax_utils.replicate(batch_ne)
                mask = jax_utils.replicate(mask)

                # Create visualization of masked points
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

                # Calculate residuals for colormap
                ru_norm, rv_norm = model.compute_residual_norms(local_state.params, local_batch, nu)
                total_residual = ru_norm + rv_norm

                # First subplot: Scatter plot with residual magnitude as color
                scatter = ax1.scatter(local_batch[:, 0], local_batch[:, 1],
                                      c=total_residual, cmap='viridis',
                                      alpha=0.6)
                plt.colorbar(scatter, ax=ax1, label='Residual Magnitude')
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                ax1.set_title('Residual Distribution')
                ax1.grid(True, linestyle='--', alpha=0.3)

                # Second subplot: Masked points
                ax2.scatter(local_batch[:, 0], local_batch[:, 1],
                            color='lightgray', alpha=0.3, label='Regular points')
                # Only plot masked points where mask is True
                masked_indices = jnp.where(mask)  # Get indices where mask is True
                ax2.scatter(local_batch[masked_indices, 0], local_batch[masked_indices, 1],
                            color='red', alpha=0.6, label='Bad subspace')
                ax2.set_xlabel('x')
                ax2.set_ylabel('y')
                ax2.set_title(f'Bad Subspace Points (top {config.masked_training.threshold_percentile}%)')
                ax2.legend()
                ax2.grid(True, linestyle='--', alpha=0.3)

                # Set axis limits for both plots
                for ax in [ax1, ax2]:
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)

                plt.tight_layout()

                # Save the plot to a temporary buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)

                # Convert to PIL Image
                image = Image.open(buf)

                original_state = model.state
                model.state = _create_train_state_with_optimizer(config.optim.adam, model)
                model.state.replace(params=original_state.params)
                # Perform masked training steps
                for mask_step in range(config.masked_training.mask_training_steps):
                    model.state = model.mask_step(model.state, batch_ne, nu)

                    # Update weights if necessary
                    if config.weighting.scheme in ["grad_norm", "ntk"]:
                        if steps % config.weighting.update_every_steps == 0:
                            model.state = model.update_weights(model.state, batch_ne, nu)

                    # Log during masked training if needed
                    if jax.process_index() == 0 and mask_step % config.logging.log_every_steps_adam_masked == 0:
                        state_out = jax.device_get(tree_map(lambda x: x[0], model.state))
                        batch_out = jax.device_get(tree_map(lambda x: x[0], batch_ne))
                        log_dict = evaluator(state_out, batch_out, x_star, y_star, U_ref, nu)
                        log_dict["training_phase"] = 1
                        log_dict["masked"] = wandb.Image(image, caption='masked')
                        wandb.log(log_dict, steps)

                        end_time = time.time()
                        # Report training metrics
                        logger.log_iter(steps, start_time, end_time, log_dict)
                        start_time = end_time
                    steps += 1
                print(f"Completed masked training sequence")
                step += config.masked_training.mask_training_steps

            # Log training metrics
            if jax.process_index() == 0:
                if steps % config.logging.log_every_steps_adam == 0:
                    state_out = jax.device_get(tree_map(lambda x: x[0], model.state))
                    batch_out = jax.device_get(tree_map(lambda x: x[0], batch))
                    log_dict = evaluator(state_out, batch_out, x_star, y_star, U_ref, nu)
                    log_dict["training_phase"] = 0
                    wandb.log(log_dict, steps + step_offset)
                    end_time = time.time()
                    # Report training metrics
                    logger.log_iter(steps, start_time, end_time, log_dict)
                    start_time = end_time

            # Save ADAM checkpoint
            if config.saving.save_every_steps is not None:
                if (steps % config.saving.save_every_steps == 0) or \
                        (step == config.training.max_steps):
                    ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt", f"Re{Re}_adam")
                    save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts, step=steps)

            steps += 1

        # 2. L-BFGS Training Phase
        print("\nStarting L-BFGS training phase...")
        lbfgs_steps = config.training.sequential_opt.lbfgs_steps

        # Calculate number of devices
        num_devices = jax.device_count()

        # Adjust batch size to be divisible by number of devices
        device_batch_size = config.optim.lbfgs.batch_size // num_devices
        total_batch_size = device_batch_size * num_devices

        # Create new sampler with adjusted batch size
        res_sampler = iter(LHSSampler(dom, total_batch_size))

        # Get batch and reshape for devices
        # batch = next(res_sampler)
        # batch = batch.reshape(num_devices, device_batch_size, -1)

        # Initialize L-BFGS optimizer
        trained_params = jax.device_get(tree_map(lambda x: x[0], model.state.params))
        model.state = _create_train_state_with_optimizer(config.optim.lbfgs, model)
        model.state = model.state.replace(params=jax_utils.replicate(trained_params))

        for step in range(steps, steps+lbfgs_steps):
            model.state = model.step(model.state, batch, nu, "lbfgs")

            if jax.process_index() == 0:
                if step % config.logging.log_every_steps_lbfgs == 0:
                    state = jax.device_get(tree_map(lambda x: x[0], model.state))
                    batch_for_eval = jax.device_get(tree_map(lambda x: x[0], batch))
                    log_dict = evaluator(state, batch_for_eval, x_star, y_star, U_ref, nu)
                    log_dict["training_phase"] = 2
                    wandb.log(log_dict, steps + step_offset)

                    end_time = time.time()
                    # Report training metrics
                    logger.log_iter(steps, start_time, end_time, log_dict)
                    start_time = end_time

            # Update weights if necessary
            if config.weighting.scheme in ["grad_norm", "ntk"]:
                if step % config.weighting.update_every_steps == 0:
                    model.state = model.update_weights(model.state, batch, nu)

            if config.saving.save_every_steps is not None:
                if step % config.saving.save_every_steps == 0 or (
                        step + 1
                ) == config.training.max_steps:
                    ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt", f"Re{Re}_adam")
                    save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts, step=steps)

            steps += 1
    else:
        # jit warm up
        print("Waiting for JIT...")
        start_time = time.time()
        print("Start ADAM training...")
        for step in range(max_steps):
            batch = next(res_sampler)   # should modify batch with mask
            model.state = model.step(model.state, batch, nu)    # train at here

            # Update weights if necessary
            if config.weighting.scheme in ["grad_norm", "ntk"]:
                if step % config.weighting.update_every_steps == 0:
                    model.state = model.update_weights(model.state, batch, nu)

            # Log training metrics, only use host 0 to record results
            if jax.process_index() == 0:
                if step % config.logging.log_every_steps_adam == 0:
                    # Get the first replica of the state and batch
                    state = jax.device_get(tree_map(lambda x: x[0], model.state))
                    batch = jax.device_get(tree_map(lambda x: x[0], batch))
                    log_dict = evaluator(state, batch, x_star, y_star, U_ref, nu)
                    wandb.log(log_dict, step + step_offset)

                    end_time = time.time()
                    # Report training metrics
                    logger.log_iter(step, start_time, end_time, log_dict)
                    start_time = end_time

            # Save checkpoint
            if config.saving.save_every_steps is not None:
                if (step + 1) % config.saving.save_every_steps == 0 or (
                    step + 1
                ) == config.training.max_steps:
                    ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt", "Re{}".format(Re))
                    save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts, step=steps)


    return model, step_offset + max_steps


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Initialize model
    model = models.NavierStokes2D(config)

    # Curriculum training
    step_offset = 0

    assert len(config.training.max_steps) == len(config.training.Re)
    num_Re = len(config.training.Re)

    for idx in range(num_Re):
        # Set Re and maximum number of training steps
        Re = config.training.Re[idx]
        max_steps = config.training.max_steps[idx]
        print("Training for Re = {}".format(Re))
        model, step_offset = train_curriculum(
            config, workdir, model, step_offset, max_steps, Re
        )

    return model


def _create_train_state_with_optimizer(optim_config, model):
    """Helper function to create train state with specific optimizer"""
    # Unreplicate the state to get the original params
    unreplicated_state = jax.device_get(tree_map(lambda x: x[0], model.state))

    # Create new optimizer
    tx = _create_optimizer(optim_config)

    # Create new state with unreplicated params
    state = TrainState.create(
        apply_fn=unreplicated_state.apply_fn,
        params=unreplicated_state.params,  # Now using unreplicated params
        tx=tx,
        weights=unreplicated_state.weights,
        momentum=unreplicated_state.momentum,
    )

    # Replicate the new state
    return jax_utils.replicate(state)