from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.data = ConfigDict()
    config.data.dataset = "celeb_a"
    config.data.train_split = "train"
    config.data.validation_split = "validation"
    config.data.train_batch_size = 64
    config.data.val_batch_size = 64

    config.model = ConfigDict()
    config.model.embedding_dim = 64
    config.model.num_embeddings = 512
    config.model.hidden_units = 128
    config.model.residual_hidden_units = 32
    config.model.residual_blocks = 2
    config.model.decay = 0.99
    config.model.use_ema = True
    config.model.commitment_cost = 0.25
    config.model.output_channels = 3

    config.steps = 100000
    config.validation_freq = 1000

    config.learning_rate = 3e-4

    return config
