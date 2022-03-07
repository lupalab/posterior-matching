from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.data = ConfigDict()
    config.data.dataset = "mnist"
    config.data.train_split = "train"
    config.data.validation_split = "test"
    config.data.train_batch_size = 256
    config.data.val_batch_size = 256
    config.data.mask_generator = "MNISTMaskGenerator"

    config.model = ConfigDict()
    config.model.latent_dim = 32
    config.model.encoder_net = "ConvEncoder"
    config.model.decoder_net = "ConvDecoder"
    config.model.posterior_dist = "TriLGaussian"
    config.model.partial_posterior_dist = "AutoregressiveGMM"
    config.model.decoder_dist = "Bernoulli"

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

    config.steps = 80000
    config.validation_freq = 1000

    config.lr_schedule = ConfigDict()
    config.lr_schedule.init_value = 0.001
    config.lr_schedule.decay_rate = 0.9
    config.lr_schedule.transition_steps = 5000

    return config
