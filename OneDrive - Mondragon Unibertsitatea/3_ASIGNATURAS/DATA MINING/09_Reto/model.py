

import torch

INPUT_SIZE = 28 * 28


def build_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(INPUT_SIZE, 20),
        torch.nn.ReLU(20),
        torch.nn.Linear(20, 10)
    )
    return model


if __name__ == "__main__":
    model = build_model()
    print(model)
