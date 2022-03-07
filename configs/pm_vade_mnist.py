from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.data = ConfigDict()
    config.data.dataset = "mnist"
    config.data.train_split = "train"
    config.data.validation_split = "test"
    config.data.train_batch_size = 128
    config.data.val_batch_size = 128

    # Replace this with a path to your own VADE model directory.
    # This should be a directory that was created by the `train_vade.py` script.
    config.vade_dir = "runs/vade-mnist-20220305-121540"

    config.model = ConfigDict()
    config.model.encoder_net = "ConvEncoder"
    config.model.decoder_net = "ConvDecoder"
    config.model.decoder_dist = "Bernoulli"
    config.model.latent_dim = 10
    config.model.num_components = 10

    config.model.partial_posterior_dist = "AutoregressiveGMM"
    config.model.partial_posterior_dist_config = ConfigDict()
    config.model.partial_posterior_dist_config.num_components = 10
    config.model.partial_posterior_dist_config.residual_blocks = 2
    config.model.partial_posterior_dist_config.hidden_units = 256

    config.model.encoder_net_config = ConfigDict()
    config.model.encoder_net_config.conv_layers = [
        (32, 5, 1),
        (32, 5, 2),
        (64, 5, 1),
        (64, 5, 2),
        (128, 7, 1),
    ]

    config.model.decoder_net_config = ConfigDict()
    config.model.decoder_net_config.conv_layers = [
        (64, 7, 1),
        (64, 5, 2),
        (32, 5, 1),
        (32, 5, 2),
        (32, 5, 1),
        (1, 5, 1),
    ]

    config.steps = 160000
    config.validation_freq = 5000

    config.lr_schedule = ConfigDict()
    config.lr_schedule.init_value = 0.001
    config.lr_schedule.decay_rate = 0.9
    config.lr_schedule.staircase = False
    config.lr_schedule.transition_steps = int(60000 / config.data.train_batch_size * 10)

    return config
