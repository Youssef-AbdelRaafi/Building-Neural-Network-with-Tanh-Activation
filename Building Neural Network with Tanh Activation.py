import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

def initialize_parameters():
    np.random.seed(0) 
    weights = {
        'w2': np.random.uniform(-0.5, 0.5),
        'w3': np.random.uniform(-0.5, 0.5),
        'w4': np.random.uniform(-0.5, 0.5),
        'w6': np.random.uniform(-0.5, 0.5),
        'w7': np.random.uniform(-0.5, 0.5),
        'w8': np.random.uniform(-0.5, 0.5)
    }
    biases = {
        'b1': 0.5,
        'b2': 0.7
    }
    return weights, biases

def forward_pass(inputs, weights, biases):
    H1_1, H1_2, H1_3, H2_1, H2_2 = inputs
    
    H5_input = H1_1 * weights['w2'] + H1_2 * weights['w3'] + H1_3 * weights['w4'] + biases['b1']
    H5_output = tanh(H5_input)
    
    H2_input = H2_1 * weights['w7'] + H2_2 * weights['w8'] + H5_output * weights['w6'] + biases['b2']
    H2_output = tanh(H2_input)
    
    O1_output = tanh(H2_output * 0.01)
    O2_output = tanh(H2_output * 99)
    
    cache = {
        'H1_1': H1_1, 'H1_2': H1_2, 'H1_3': H1_3,
        'H2_1': H2_1, 'H2_2': H2_2,
        'H5_input': H5_input, 'H5_output': H5_output,
        'H2_input': H2_input, 'H2_output': H2_output,
        'O1_output': O1_output, 'O2_output': O2_output
    }
    
    return O1_output, O2_output, cache

def main():
    weights, biases = initialize_parameters()
    
    inputs = (20, 30, 45, 50, 55)
    
    O1, O2, cache = forward_pass(inputs, weights, biases)
    
    print(f"Output O1: {O1}")
    print(f"Output O2: {O2}")
    print("\nIntermediate Values:")
    for key, value in cache.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()