from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.data = ConfigDict()
    config.data.dataset = "mnist16"
    config.data.train_split = "train"
    config.data.validation_split = "test"
    config.data.train_batch_size = 128
    config.data.val_batch_size = 128
    config.data.mask_generator = "UniformMaskGenerator"
    config.data.mask_generator_kwargs = ConfigDict()
    config.data.mask_generator_kwargs.bounds = (0.0, 0.2)

    config.model = ConfigDict()
    config.model.latent_dim = 10
    config.model.encoder_net = "ConvEncoder"
    config.model.decoder_net = "ConvDecoder"
    config.model.posterior_dist = "TriLGaussian"
    config.model.decoder_dist = "Bernoulli"

    config.model.encoder_net_config = ConfigDict()
    config.model.encoder_net_config.conv_layers = [
        (32, 3, 1),
        (32, 3, 2),
        (64, 3, 2),
        (64, 1, 1),
    ]

    config.model.decoder_net_config = ConfigDict()
    config.model.decoder_net_config.conv_layers = [
        (64, 8, 1),
        (64, 5, 2),
        (32, 5, 1),
        (32, 5, 1),
        (1, 3, 1),
    ]

    config.steps = 200000
    config.validation_freq = 10000

    config.lr_schedule = ConfigDict()
    config.lr_schedule.init_value = 0.001
    config.lr_schedule.decay_rate = 0.9
    config.lr_schedule.transition_steps = 5000

    return config
