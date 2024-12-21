import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-LDC"
    wandb.name = "default-VP-400"
    wandb.tag = None

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "ModifiedMlp"
    arch.num_layers = 8
    arch.hidden_dim = 100
    arch.out_dim = 3
    arch.activation = "swish"
    arch.periodicity = False
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 10.0, "embed_dim": 100})
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}
    )

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    # ADAM settings
    optim.adam = adam = ml_collections.ConfigDict()
    adam.optimizer = "Adam"
    adam.beta1 = 0.9
    adam.beta2 = 0.999
    adam.eps = 1e-8
    adam.learning_rate = 1e-3
    adam.decay_rate = 0.9
    adam.decay_steps = 2000
    adam.grad_accum_steps = 0

    # L-BFGS settings
    optim.lbfgs = lbfgs = ml_collections.ConfigDict()
    lbfgs.optimizer = "L-BFGS"
    lbfgs.max_backtracking_steps = 30
    lbfgs.batch_size = 1024 * 5  # Larger batch size for L-BFGS
    lbfgs.grad_accum_steps = 0

    optim.grad_accum_steps = 0

    # Training

    config.masked_training = masked = ml_collections.ConfigDict()
    masked.mask_training_interval = 100  # Run masked training every x steps
    masked.mask_training_steps = 100    # Number of masked training steps
    masked.threshold_percentile = 95  # Percentile for identifying bad subspace


    # Sequential optimization settings
    config.training = training = ml_collections.ConfigDict()
    training.sequential_opt = sequential = ml_collections.ConfigDict()
    sequential.use_sequential = True  # Whether to use ADAM + L-BFGS
    sequential.adam_steps = 1000  # Number of ADAM steps
    sequential.lbfgs_steps = 1000  # Number of L-BFGS steps

    training.Re =[400] # [100, 400, 1000]
    training.max_steps = [sequential.adam_steps + sequential.lbfgs_steps +
                          masked.mask_training_steps * (sequential.adam_steps // masked.mask_training_interval)
                          ] # [20000, 40000, 140000]
    training.batch_size = 1024 * 5

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict(
        {"u_bc": 1.0, "v_bc": 1.0, "ru": 1.0, "rv": 1.0, "rc": 1.0}
    )
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps_adam = 100
    logging.log_every_steps_lbfgs = 100
    logging.log_every_steps_adam_masked = 10
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_grads = False
    logging.log_ntk = False
    logging.log_preds = True
    logging.log_residuals = True

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 1000
    saving.num_keep_ckpts = 5

    # Input shape for initializing Flax models
    config.input_dim = 2

    # Integer for PRNG random seed.
    config.seed = 42

    return config
