<<<<<<< HEAD
import torch

if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    print(f"CUDA enabled: {cuda}")
    if cuda:
=======
import torch

if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    print(f"CUDA enabled: {cuda}")
    if cuda:
>>>>>>> 1d4db2f081cb7f024eed821f55ecde73b820a964
        print(f"Number of GPUs: {torch.cuda.device_count()}")