from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.data = ConfigDict()
    config.data.dataset = "mnist"
    config.data.train_split = "train"
    config.data.validation_split = "test"
    config.data.train_batch_size = 32
    config.data.val_batch_size = 32
    config.data.mask_generator = "MNISTMaskGenerator"

    # Replace this with a path to your own VQVAE model directory.
    # This should be a directory that was created by the `train_vqvae.py` script.
    config.vqvae_dir = "runs/vqvae-mnist-20220227-111235"

    config.pixel_cnn = ConfigDict()
    config.pixel_cnn.image_shape = (7, 7)
    config.pixel_cnn.num_resnet = 8
    config.pixel_cnn.num_hierarchies = 1
    config.pixel_cnn.num_filters = 128
    config.pixel_cnn.dropout = 0.5

    config.conditional_dim = 512

    config.steps = 120000
    config.validation_freq = 1000

    config.lr_schedule = ConfigDict()
    config.lr_schedule.init_value = 3e-4
    config.lr_schedule.decay_rate = 0.999995
    config.lr_schedule.transition_steps = 1

    return config
