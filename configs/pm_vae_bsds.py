from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.data = ConfigDict()
    config.data.dataset = "bsds"
    config.data.train_split = "train"
    config.data.validation_split = "val"
    config.data.train_batch_size = 512
    config.data.val_batch_size = 512
    config.data.training_noise = 0.001
    config.data.mask_generator = "BernoulliMaskGenerator"

    config.model = ConfigDict()
    config.model.latent_dim = 64
    config.model.encoder_net = "ResidualMLP"
    config.model.decoder_net = "ResidualMLP"
    config.model.decoder_dist = "IdentityGaussian"
    config.model.posterior_dist = "TriLGaussian"
    config.model.decoder_dist_config = ConfigDict()
    config.model.decoder_dist_config.event_size = 63
    config.model.masked_posterior_dist = "AutoregressiveGMM"
    config.model.masked_posterior_config = ConfigDict()
    config.model.masked_posterior_config.hidden_units = 256
    config.model.masked_posterior_config.residual_blocks = 3

    config.model.encoder_net_config = ConfigDict()
    config.model.encoder_net_config.residual_blocks = 5
    config.model.encoder_net_config.hidden_units = 256
    config.model.encoder_net_config.layer_norm = True

    config.model.decoder_net_config = ConfigDict()
    config.model.decoder_net_config.residual_blocks = 5
    config.model.decoder_net_config.hidden_units = 256
    config.model.decoder_net_config.layer_norm = True

    config.model.matching_ll_stop_gradients = True

    config.beta = ConfigDict()
    config.beta.schedule = "monotonic"
    config.beta.low_value = 0.0
    config.beta.high_value = 1.0
    config.beta.transition_steps = 200000
    config.beta.transition_begin = 30000

    config.steps = 200000
    config.validation_freq = 1000
    config.save_final_state = True

    config.weight_decay = 0.00001

    config.lr_schedule = ConfigDict()
    config.lr_schedule.init_value = 0.001
    config.lr_schedule.decay_rate = 0.9
    config.lr_schedule.transition_steps = 5000

    return config
