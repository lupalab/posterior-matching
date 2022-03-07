from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.data = ConfigDict()
    config.data.dataset = "mnist16"
    config.data.train_split = "train"
    config.data.validation_split = "test"
    config.data.train_batch_size = 32
    config.data.val_batch_size = 32
    config.data.mask_generator = "UniformMaskGenerator"
    config.data.mask_generator_kwargs = ConfigDict()
    config.data.mask_generator_kwargs.bounds = (0.0, 0.20)

    # Replace this with a path to your own VAE model directory.
    # This should be a directory that was created by the `train_pm_vae.py` script.
    config.pm_vae_dir = "runs/pm-vae-mnist16-20220302-160842"

    config.model = ConfigDict()
    config.model.lookahead_subsample = 16
    config.model.model_samples = 64

    config.steps = 40000
    config.validation_freq = 5000

    config.lr_schedule = ConfigDict()
    config.lr_schedule.init_value = 0.001
    config.lr_schedule.decay_rate = 0.9
    config.lr_schedule.transition_steps = 5000

    return config
