import torch

from model.subsequent_mask import subsequent_mask
from model.model_example import make_model

def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)
    print('src_mask', src_mask)

    # First - encode once
    memory = test_model.encode(src, src_mask)
    print('memory', memory)
    print('memory shape:', memory.shape)

    # Outputs
    ys = torch.zeros(1, 1).type_as(src)
    print('ys before -', ys)

    for i in range(9):
        print(f"\nInference iter {i}")
        print("=" * 20)
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        print('out', out)
        prob = test_model.generator(out[:, -1])
        print('prob', prob)
        _, next_word = torch.max(prob, dim=1)
        print('next_word', next_word)
        next_word = next_word.data[0]
        print('next_word pred', next_word)
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
        print('ys', ys)

    print("Example Untrained Model Prediction:", ys)


inference_test()