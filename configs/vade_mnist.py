from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.data = ConfigDict()
    config.data.dataset = "mnist"
    config.data.train_split = "train"
    config.data.validation_split = "test"
    config.data.train_batch_size = 128
    config.data.val_batch_size = 128

    config.model = ConfigDict()
    config.model.encoder_net = "ConvEncoder"
    config.model.decoder_net = "ConvDecoder"
    config.model.decoder_dist = "Bernoulli"
    config.model.latent_dim = 10
    config.model.num_components = 10

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

    config.pretrain_steps = int(60000 / config.data.train_batch_size * 150)
    config.steps = int(60000 / config.data.train_batch_size * 300)
    config.validation_freq = 1000
    config.cluster_pred_num_samples = 50

    config.pretrain_lr = 0.002

    config.lr_schedule = ConfigDict()
    config.lr_schedule.init_value = 0.002
    config.lr_schedule.decay_rate = 0.9
    config.lr_schedule.staircase = False
    config.lr_schedule.transition_steps = int(60000 / config.data.train_batch_size * 10)

    config.adam = ConfigDict()
    config.adam.eps = 1e-4

    return config
