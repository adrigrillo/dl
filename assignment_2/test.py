import math
import random

import numpy as np


def random_sine(batch_size, steps_per_epoch,
                input_sequence_length, target_sequence_length,
                min_frequency=0.1, max_frequency=10,
                min_amplitude=0.1, max_amplitude=1,
                min_offset=-0.5, max_offset=0.5,
                num_signals=3, seed=43):
    """Produce a batch of signals.
    The signals are the sum of randomly generated sine waves.
    Arguments
    ---------
    batch_size: Number of signals to produce.
    steps_per_epoch: Number of batches of size batch_size produced by the
        generator.
    input_sequence_length: Length of the input signals to produce.
    target_sequence_length: Length of the target signals to produce.
    min_frequency: Minimum frequency of the base signals that are summed.
    max_frequency: Maximum frequency of the base signals that are summed.
    min_amplitude: Minimum amplitude of the base signals that are summed.
    max_amplitude: Maximum amplitude of the base signals that are summed.
    min_offset: Minimum offset of the base signals that are summed.
    max_offset: Maximum offset of the base signals that are summed.
    num_signals: Number of signals that are summed together.
    seed: The seed used for generating random numbers

    Returns
    -------
    signals: 2D array of shape (batch_size, sequence_length)
    """
    num_points = input_sequence_length + target_sequence_length
    x = np.arange(num_points) * 2 * np.pi / 30

    while True:
        # Reset seed to obtain same sequences from epoch to epoch
        np.random.seed(seed)

        for _ in range(steps_per_epoch):
            signals = np.zeros((batch_size, num_points))
            for _ in range(num_signals):
                # Generate random amplitude, frequence, offset, phase
                amplitude = (np.random.rand(batch_size, 1) *
                             (max_amplitude - min_amplitude) +
                             min_amplitude)
                frequency = (np.random.rand(batch_size, 1) *
                             (max_frequency - min_frequency) +
                             min_frequency)
                offset = (np.random.rand(batch_size, 1) *
                          (max_offset - min_offset) +
                          min_offset)
                phase = np.random.rand(batch_size, 1) * 2 * np.pi

                signals += amplitude * np.sin(frequency * x + phase)
            signals = np.expand_dims(signals, axis=2)

            encoder_input = signals[:, :input_sequence_length, :]
            decoder_output = signals[:, input_sequence_length:, :]

            # The output of the generator must be ([encoder_input, decoder_input], [decoder_output])
            decoder_input = np.zeros((decoder_output.shape[0], decoder_output.shape[1], 1))

            yield ([encoder_input, decoder_input], decoder_output)


def generate_x_y_data_v1(isTrain, batch_size):
    """
    Data for exercise 1.
    returns: tuple (X, Y)
        X is a sine and a cosine from 0.0*pi to 1.5*pi
        Y is a sine and a cosine from 1.5*pi to 3.0*pi
    Therefore, Y follows X. There is also a random offset
    commonly applied to X an Y.
    The returned arrays are of shape:
        (seq_length, batch_size, output_dim)
        Therefore: (10, batch_size, 2)
    For this exercise, let's ignore the "isTrain"
    argument and test on the same data.
    """
    seq_length = 10

    batch_x = []
    batch_y = []
    for _ in range(batch_size):
        rand = random.random() * 2 * math.pi

        sig1 = np.sin(np.linspace(0.0 * math.pi + rand,
                                  3.0 * math.pi + rand, seq_length * 2))
        sig2 = np.cos(np.linspace(0.0 * math.pi + rand,
                                  3.0 * math.pi + rand, seq_length * 2))
        x1 = sig1[:seq_length]
        y1 = sig1[seq_length:]
        x2 = sig2[:seq_length]
        y2 = sig2[seq_length:]

        x_ = np.array([x1, x2])
        y_ = np.array([y1, y2])
        x_, y_ = x_.T, y_.T

        batch_x.append(x_)
        batch_y.append(y_)

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    # shape: (batch_size, seq_length, output_dim)

    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))
    # shape: (seq_length, batch_size, output_dim)

    return batch_x, batch_y


if __name__ == '__main__':
    generator = random_sine(batch_size=2, steps_per_epoch=2, input_sequence_length=3, target_sequence_length=1)
    datas = list()
    for i, data in enumerate(generator):
        datas.append(data)
        if i == 10:
            break

    data = generate_x_y_data_v1(True, 5)
