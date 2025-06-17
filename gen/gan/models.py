from gan.layers import ImageConversion, ImageConversionMode, PixelNorm, ScaledConv2d, ScaledDense, ScaledLeakyRelu, Upsample, UpsampleConv2d
import tensorflow as tf
from typing import Optional


def validate_resolution(resolution: int) -> None:
    assert resolution in [4, 8, 16, 32, 64, 128, 256, 512, 1024]


def activate(
        x,
        activation = lambda: ScaledLeakyRelu()):
    if activation is not None:
        x = activation()(x)
    return x


def conv_2d(
        x,
        channel_count: int,
        kernel_size: int = 3,
        activation = lambda: ScaledLeakyRelu(),
        padding: str = 'same',
        strides: int = 1,
        pre_blur: bool = False,
        name: Optional[str] = None):
    x = ScaledConv2d(
        channel_count,
        kernel_size,
        padding=padding,
        strides=strides,
        pre_blur=pre_blur,
        name=name)(x)
    return activate(x, activation=activation)


def create_generator_body(resolution, latent_vector):
    def to_rgb(x):
        rgb = conv_2d(x, 3, 1, activation=None, name=f'to_rgb_{resolution}x{resolution}')
        return rgb

    resolution_to_channel_counts = {
        4: 512, 8: 512, 16: 512, 32: 512, 64: 512, 128: 256, 256: 128, 512: 64, 1024: 32}
    channel_count = resolution_to_channel_counts[resolution]
    if resolution == 4:
        block = ScaledDense(
            channel_count*4*4,
            name='latent_to_4x4')(latent_vector)
        block = tf.keras.layers.Reshape((4, 4, channel_count))(block)
        block = activate(block)
        block = conv_2d(block, channel_count, name='conv_4x4')
        rgb = to_rgb(block)
        return rgb, block
    else:
        lower_res_rgb, lower_res_block = create_generator_body(resolution // 2, latent_vector)
        name_base = f'conv_{resolution}x{resolution}'

        block = lower_res_block
        block = activate(UpsampleConv2d(channel_count, name=f'{name_base}_1')(block))
        block = conv_2d(block, channel_count, name=f'{name_base}_2')
        rgb = to_rgb(block)

        lower_res_rgb = Upsample()(lower_res_rgb)
        rgb = tf.keras.layers.Add()([lower_res_rgb, rgb])
        return rgb, block


def create_generator(output_resolution: int, latent_vector_size: int ) -> tf.keras.Model:
    validate_resolution(output_resolution)

    latent_vector = tf.keras.layers.Input((latent_vector_size,))
    normalized_latent_vector = PixelNorm()(latent_vector)

    model_rgb, _ = create_generator_body(output_resolution, normalized_latent_vector)
    tensorflow_rgb = ImageConversion(ImageConversionMode.MODEL_TO_TENSORFLOW)(model_rgb)

    # Cast the output to float32. Needed when using mixed_float16.
    if tensorflow_rgb.dtype != tf.float32:
        tensorflow_rgb = tf.keras.layers.Activation(None, dtype=tf.float32)(tensorflow_rgb)

    return tf.keras.Model(inputs=latent_vector, outputs=tensorflow_rgb)














