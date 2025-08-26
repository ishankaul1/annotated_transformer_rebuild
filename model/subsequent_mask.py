import torch

# Just a math trick to get diagonal a matrix shifted diagonal 1 up; 
# Easily zero's out anything past your value
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0