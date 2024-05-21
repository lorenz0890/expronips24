import torch
import numpy as np
import random
import torch.nn.functional as F

import struct


def flip_bits_across_layers(model, layer_names, num_bits, part='mantissa'):
    """
    Flip a total number of bits, randomly distributed across the trainable parameters
    of specified layers in a model, either in the exponent, mantissa part of the
    floating-point representation, or the sign bit.

    Parameters:
    - model: The PyTorch model whose parameters will be manipulated.
    - layer_names: A list of layer names (strings) whose parameters will be targeted.
    - num_bits: The total number of bits to flip across all selected layers.
    - part: The part of the floating-point representation to flip bits in ('exponent', 'mantissa', or 'sign').
    """
    bit_positions = range(0, 32)  # all
    if part == 'exponent':
        bit_positions = range(1, 9)  # Exponent bits
    elif part == 'mantissa':
        bit_positions = range(9, 32)  # Mantissa bits
    elif part == 'sign':
        bit_positions = [31]  # Sign bit only

    # Collect all parameters from the specified layers
    targeted_params = []
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names) and param.requires_grad:
            targeted_params.append(param)

    if not targeted_params:
        raise ValueError("No matching layers found or no parameters require gradients.")

    flipped_bits = 0
    for _ in range(num_bits):
        # Randomly select a parameter to attack
        param_to_attack = random.choice(targeted_params)
        param_numpy = param_to_attack.detach().cpu().numpy()

        # Randomly select an element in the parameter array
        total_elements = param_numpy.size
        random_index = np.unravel_index(random.randint(0, total_elements - 1), param_numpy.shape)

        # Convert the selected element to its IEEE 754 binary32 representation
        value = param_numpy[random_index]
        bits = np.frombuffer(np.float32(value).tobytes(), dtype=np.uint32)[0]

        # Handle sign bit flipping differently for true random flips
        #print('was', param_numpy[random_index], np.float32(value).tobytes())
        if part == 'sign':
            # Flip the sign bit (32nd bit)
            #bits ^= 1 << 31  # XOR operation to flip the sign bit regardles of state
            #print(print(f"OK bit 32-bit representation: {bits:032b}"))
            bits |= 1 << 31  # XOR operation to flip the sign bit to 1
            #print(f"Flipped 32-bit representation: {bits:032b}")
            flipped_bits+=1
        else:
            # Flip a bit from 1 to 0 for exponent or mantissa
            for flip_position in bit_positions:
                if bits & (1 << (32 - flip_position)):
                    bits &= ~(1 << (32 - flip_position))
                    flipped_bits += 1
                    break

        # Convert the manipulated bits back to a floating-point number and update the selected element
        param_numpy[random_index] = np.frombuffer(np.array([bits], dtype=np.uint32).tobytes(), dtype=np.float32)
        #print('is', param_numpy[random_index], np.array([bits], dtype=np.uint32).tobytes(), '\n')
        # Update the original parameter tensor
        with torch.no_grad():
            param_to_attack.copy_(torch.from_numpy(param_numpy).to(param_to_attack.device))

        if part != 'sign' and flipped_bits >= num_bits:
            break  # Exit if the target number of bits has been flipped for exponent/mantissa

    return flipped_bits
'''


def flip_bits_across_layers(model, layer_names, num_bits, part='mantissa'):
    """
    Iterates through every target parameter and then through every column,
    applying the flipping strategy to the row with the maximum value until the desired number of bits is flipped.

    Parameters:
    - model: The PyTorch model whose parameters will be manipulated.
    - layer_names: A list of layer names (strings) whose parameters will be targeted.
    - num_bits: The total number of bits to flip across all selected layers.
    - part: Part of the floating-point representation to flip bits in. Not directly used as flips are uniformly distributed.
    """

    targeted_params = [param for name, param in model.named_parameters()
                       if any(layer_name in name for layer_name in layer_names) and param.requires_grad]

    if not targeted_params:
        raise ValueError("No matching layers found or no parameters require gradients.")

    flipped_bits = 0

    for param in targeted_params:
        param_numpy = param.detach().cpu().numpy()

        # Iterate over columns
        for col in range(param_numpy.shape[-1]):
            if flipped_bits >= num_bits:
                break  # Stop if the desired number of bit flips is reached

            # Find the row with the maximum value in this column
            if param_numpy.ndim == 2:
                row = np.argmax(param_numpy[:, col])
                value = param_numpy[row, col]
            else:
                # For tensors with more dimensions, adjust accordingly
                row = np.argmax(param_numpy[..., col])
                value = param_numpy[..., col][row]

            if value <= 0:
                continue  # Skip non-positive values

            bits = np.frombuffer(np.float32(value).tobytes(), dtype=np.uint32)[0]

            # Flip the largest unset bit in the exponent
            for bit_position in range(30, 22, -1):
                if not (bits & (1 << bit_position)):
                    bits |= 1 << bit_position
                    flipped_bits += 1
                    break

            # Ensure the number is negative by setting the sign bit
            bits |= 1 << 31
            flipped_bits += 1

            # Update the parameter
            if param_numpy.ndim == 2:
                param_numpy[row, col] = np.frombuffer(np.array([bits], dtype=np.uint32).tobytes(), dtype=np.float32)[0]
            else:
                # For tensors with more dimensions, adjust the indexing as needed
                param_numpy[..., col][row] = np.frombuffer(np.array([bits], dtype=np.uint32).tobytes(), dtype=np.float32)[0]

            with torch.no_grad():
                param.copy_(torch.from_numpy(param_numpy).to(param.device))
            #print(param)
            #print(param_numpy, '\n')
            if flipped_bits >= num_bits:
                break  # Exit the loop once the desired number of flips is reached

    return flipped_bits
'''