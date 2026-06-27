import torch
import numpy as np

class Boop:
    def __init__(self, config):
        self.config = config

    def beep(self):
        return self.config["FOOD"]

class Beep:
    def __init__(self, config):
        self.config = config

    def boop(self):
        self.config["FOOD"] = "New Beep"

def main():
    config = {"FOOD": "Beep"}
    boop = Boop(config)
    beep = Beep(config)
    print(boop.beep())
    beep.boop()
    print(boop.beep())

    ten1 = torch.tensor(np.ndarray((64, 2, 22, 10)))
    ten2 = torch.tensor(np.ndarray((64, 2, 22, 10)))

    print(f"ten1: {ten1.shape}")
    print(f"ten2: {ten2.shape}")

    ten3 = torch.tensor(np.ndarray((64, 128, 1, 10)))
    ten4 = torch.tensor(np.ndarray((64, 128, 22, 1)))

    print(f"ten3: {ten3.shape}")
    print(f"ten4: {ten4.shape}")

    flat1 = ten1.flatten(start_dim=1)
    flat2 = ten2.flatten(start_dim=1)

    print(f"flat1: {flat1.shape}")
    print(f"flat2: {flat2.shape}")

    flat3 = ten3.flatten(start_dim=1)
    flat4 = ten4.flatten(start_dim=1)

    print(f"flat3: {flat3.shape}")
    print(f"flat4: {flat4.shape}")

    cat12 = torch.cat([flat1, flat2], dim=1)
    cat34 = torch.cat([flat3, flat4], dim=1)

    print(cat12.shape)
    print(cat34.shape)

    ten_shorted = torch.tensor(np.ndarray((32, 2, 22, 10)))
    a_batched = ten_shorted[5]

    print(a_batched.shape)

    ten_shorted1 = torch.zeros((12))
    print(ten_shorted1.shape)
    print(ten_shorted1[4].shape)

    ten_shorted2 = torch.zeros((12, 1))
    print(ten_shorted2.shape)
    print(ten_shorted2[3].shape)


if __name__ == "__main__":
    main()