import functools
from gan.models import create_generator
from gan.serialize import deserialize_model, serialize_model
import tensorflow as tf


class TrainingOptions:
    def __init__(
            self,
            resolution: int,
            latent_size = 512
            ):

        self.resolution = resolution
        self.latent_size = latent_size


class TrainingState:
    def __init__(
            self,
            options: TrainingOptions,
            generator: tf.keras.Model = None,
            epoch_i: int = 0):
        self.options = options
        self.generator = generator
        self.epoch_i = epoch_i

    def __getstate__(self):
        state = self.__dict__.copy()
        state['generator'] = serialize_model(self.generator)
        return state

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        self.generator = deserialize_model(
            self.generator,
            functools.partial(create_generator, self.options.resolution, self.options.latent_size))














