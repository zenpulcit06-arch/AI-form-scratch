import numpy as np
import struct

def load_pytorch(filename):
    with open(filename, 'rb') as f:
        N = np.frombuffer(f.read(8), dtype=np.int64)[0]
        layers = []
        for _ in range(N):
            out_features = np.frombuffer(f.read(8), dtype=np.int64)[0]
            in_features  = np.frombuffer(f.read(8), dtype=np.int64)[0]
            W = np.frombuffer(f.read(8 * out_features * in_features), dtype=np.float64).reshape(out_features, in_features)
            b = np.frombuffer(f.read(8 * out_features), dtype=np.float64)
            layers.append((W, b))
    return layers

def print_layer_info(layers):
    for i, (W, b) in enumerate(layers):
        print(f"Layer {i+1}: W={W.shape}, b={b.shape}")
    


if __name__ == "__main__":
    filename = input("Enter the filename to load: ")
    layers = load_pytorch(filename)
    print_layer_info(layers)