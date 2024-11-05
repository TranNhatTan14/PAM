import torch
import time
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

class PerformanceTest:
    def __init__(self):
        # Check CUDA availability
        self.device_cpu = torch.device('cpu')
        self.device_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Print device information
        print(f"CPU Device: {self.device_cpu}")
        if torch.cuda.is_available():
            print(f"GPU Device: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    def test_matrix_multiplication(self, size):
        """Test matrix multiplication performance"""
        # Generate random matrices
        matrix1 = torch.randn(size, size)
        matrix2 = torch.randn(size, size)
        
        # CPU test
        start_time = time.time()
        matrix1_cpu = matrix1.to(self.device_cpu)
        matrix2_cpu = matrix2.to(self.device_cpu)
        result_cpu = torch.matmul(matrix1_cpu, matrix2_cpu)
        cpu_time = time.time() - start_time
        
        # GPU test
        start_time = time.time()
        matrix1_gpu = matrix1.to(self.device_gpu)
        matrix2_gpu = matrix2.to(self.device_gpu)
        result_gpu = torch.matmul(matrix1_gpu, matrix2_gpu)
        torch.cuda.synchronize()  # Wait for GPU operation to complete
        gpu_time = time.time() - start_time
        
        return cpu_time, gpu_time

    def test_neural_network(self, input_size, hidden_size, num_epochs):
        """Test neural network training performance"""
        class SimpleNN(nn.Module):
            def __init__(self, input_size, hidden_size):
                super(SimpleNN, self).__init__()
                self.layer1 = nn.Linear(input_size, hidden_size)
                self.layer2 = nn.Linear(hidden_size, 1)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.relu(self.layer1(x))
                x = self.layer2(x)
                return x

        # Generate dummy data
        X = torch.randn(10000, input_size)
        y = torch.randn(10000, 1)

        def train_model(device):
            model = SimpleNN(input_size, hidden_size).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters())
            
            X_device = X.to(device)
            y_device = y.to(device)
            
            start_time = time.time()
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                outputs = model(X_device)
                loss = criterion(outputs, y_device)
                loss.backward()
                optimizer.step()
                
                if device == self.device_gpu:
                    torch.cuda.synchronize()
                    
            return time.time() - start_time

        cpu_time = train_model(self.device_cpu)
        gpu_time = train_model(self.device_gpu)
        
        return cpu_time, gpu_time

    def plot_results(self, sizes, cpu_times, gpu_times, title):
        """Plot performance comparison"""
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, cpu_times, 'b-', label='CPU')
        plt.plot(sizes, gpu_times, 'r-', label='GPU')
        plt.xlabel('Matrix Size' if 'Matrix' in title else 'Input Size')
        plt.ylabel('Time (seconds)')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    test = PerformanceTest()
    
    # Test 1: Matrix Multiplication with different sizes
    matrix_sizes = [1000, 2000, 3000, 4000, 5000]
    cpu_times_matrix = []
    gpu_times_matrix = []
    
    print("\nTesting Matrix Multiplication:")
    for size in matrix_sizes:
        print(f"\nMatrix size: {size}x{size}")
        cpu_time, gpu_time = test.test_matrix_multiplication(size)
        cpu_times_matrix.append(cpu_time)
        gpu_times_matrix.append(gpu_time)
        print(f"CPU Time: {cpu_time:.4f} seconds")
        print(f"GPU Time: {gpu_time:.4f} seconds")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    
    # Plot matrix multiplication results
    test.plot_results(matrix_sizes, cpu_times_matrix, gpu_times_matrix, 
                     'Matrix Multiplication Performance Comparison')
    
    # Test 2: Neural Network Training
    input_sizes = [100, 200, 300, 400, 500]
    cpu_times_nn = []
    gpu_times_nn = []
    hidden_size = 256
    num_epochs = 100
    
    print("\nTesting Neural Network Training:")
    for input_size in input_sizes:
        print(f"\nInput size: {input_size}")
        cpu_time, gpu_time = test.test_neural_network(input_size, hidden_size, num_epochs)
        cpu_times_nn.append(cpu_time)
        gpu_times_nn.append(gpu_time)
        print(f"CPU Time: {cpu_time:.4f} seconds")
        print(f"GPU Time: {gpu_time:.4f} seconds")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    
    # Plot neural network results
    test.plot_results(input_sizes, cpu_times_nn, gpu_times_nn, 
                     'Neural Network Training Performance Comparison')

if __name__ == "__main__":
    main()