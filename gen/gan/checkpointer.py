import os
import tensorflow as tf
from typing import List, Optional, Union


def read_binary_file(file_path, open_mode='rb') -> bytes:
    with open(file_path, open_mode) as f:
        return f.read()
    

def write_binary_file(file_path, content, open_mode='wb') -> None:
    with open(file_path, open_mode) as f:
        f.write(content)


class Checkpointer:
	def __init__(self, model_path):
		super().__init__()
		self.model_path = model_path

	def path_for_checkpoint(self, checkpoint_i):
		return os.path.join(self.out, f'{checkpoint_i}.checkpoint')

	def save_checkpoint(self, checkpoint_i, content) -> None:
		write_binary_file(self.path_for_checkpoint(checkpoint_i), content)

	def load_checkpoint(self) -> bytes:
		return read_binary_file(self.model_path)
