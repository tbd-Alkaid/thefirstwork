import numpy as np

class Config(object):
    n_layer = 4
    batch_size = 64
    valid_size = 256
    step_boundaries = [2000, 4000]
    num_iterations = 6000
    logging_frequency = 5
    verbose = True
    y_init_range = [0, 1]


class AllenCahnConfig(Config):
    total_time = 0.3
    num_time_interval = 20
    dim = 5
    lr_values = list(np.array([5e-4, 5e-4]))
    lr_boundaries = [2000]
    num_iterations = 1000
    num_hiddens = [dim, dim + 10, dim + 10, dim]
    y_init_range = [0.3, 0.6]

def get_config(name):
    try:
        return globals()[name+'Config']
    except KeyError:
        raise KeyError("Config for the required problem not found.")
