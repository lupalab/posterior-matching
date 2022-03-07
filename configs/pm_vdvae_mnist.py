from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.data = ConfigDict()
    config.data.dataset = "mnist"
    config.data.train_split = "train"
    config.data.validation_split = "test"
    # Note that this is the per-device batch size. Models in the paper were trained on
    # 8 TPUv3 cores, so the total batch size was 128. The batch size here may need to
    # be adjusted when using a different number of accelerators.
    config.data.train_batch_size = 16
    config.data.val_batch_size = 16
    config.data.mask_generator = "MNISTMaskGenerator"

    config.model = ConfigDict()
    config.model.image_shape = (28, 28, 1)
    config.model.encoder_blocks = "28x6,28d2,14x4,14d2,7x2,7d2,3x2,3d2,1x2"
    config.model.decoder_blocks = "1x2,3m1,3x2,7m3,7x2,14m7,14x4,28m14,28x6"
    config.model.latent_dim = 16
    config.model.width = 192
    config.model.bottleneck_multiple = 0.25
    config.model.no_bias_above = 64
    config.model.num_mixtures = 10
    config.model.custom_width_string = None

    config.ema_rate = 0.999
    config.gradient_clip = 200.0
    config.lr = 0.00015

    config.steps = 500000
    config.validation_freq = 5000

    return config