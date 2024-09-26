import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
from hcr import HCR
from kan import KANModel
from scipy.optimize import minimize

class HCRLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, m: int):
        super(HCRLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.m = m
        self.hcr = HCR(m)
        self.coefficients = nn.Parameter(torch.randn(output_dim, (m+1) ** input_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = self.legendre_basis(x)
        return self.hcr_density(basis)
    
    def legendre_basis(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        basis = np.array([self.hcr.legendre_basis(self.m, x_np[:, i]) for i in range(self.input_dim)])
        basis = np.transpose(basis, (1, 0, 2))
        return torch.tensor(basis, dtype=torch.float32, device=x.device)
    
    def hcr_density(self, basis: torch.Tensor) -> torch.Tensor:
        batch_size = basis.shape[0]
        density = torch.ones(batch_size, self.output_dim, device=basis.device)
        
        for idx in np.ndindex(*([self.m+1] * self.input_dim)):
            if sum(idx) > 0:
                flat_idx = np.ravel_multi_index(idx, (self.m+1,) * self.input_dim)
                coefficient_slice = self.coefficients[:, flat_idx]
                
                basis_slice = torch.prod(basis[:, range(self.input_dim), idx], dim=1)
                
                density += coefficient_slice.unsqueeze(0) * basis_slice.unsqueeze(1)
        
        return density

class HCRNN(nn.Module):
    def __init__(self, layer_dims: List[int], m: int):
        super(HCRNN, self).__init__()
        self.layers = nn.ModuleList([HCRLayer(layer_dims[i], layer_dims[i+1], m) for i in range(len(layer_dims) - 1)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class HCRKAN(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int, m: int):
        super(HCRKAN, self).__init__()
        self.kan = KANModel(input_dim, hidden_layers, output_dim)
        self.hcrnn = HCRNN([input_dim] + hidden_layers + [output_dim], m)
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_kan = torch.tensor(self.kan.kan_transform(x.detach().cpu().numpy())).float().to(x.device)
        x_hcr = self.hcrnn(x)
        return torch.cat([x_kan, x_hcr], dim=1)

def train_hcrkan(model: HCRKAN, X: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 0.001) -> List[float]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs[:, -model.output_dim:], y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return losses

def direct_estimation_training(model: HCRKAN, X: np.ndarray, y: np.ndarray, m: int):
    for layer in model.hcrnn.layers:
        coeffs = layer.hcr.estimate_coefficients_nd(np.column_stack((X, y)))
        coeffs_reshaped = coeffs.reshape(layer.output_dim, -1)
        if coeffs_reshaped.shape[1] != layer.coefficients.shape[1]:
            coeffs_reshaped = np.pad(coeffs_reshaped, ((0, 0), (0, layer.coefficients.shape[1] - coeffs_reshaped.shape[1])))
        layer.coefficients.data = torch.tensor(coeffs_reshaped, dtype=torch.float32)

def tensor_decomposition_training(model: HCRKAN, X: np.ndarray, y: np.ndarray, max_iter: int = 100):
    def objective(params):
        param_splits = np.split(params, [np.prod(layer.coefficients.shape) for layer in model.hcrnn.layers])
        for layer, layer_params in zip(model.hcrnn.layers, param_splits):
            layer.coefficients.data = torch.tensor(layer_params.reshape(layer.coefficients.shape), dtype=torch.float32)
        
        outputs = model(torch.tensor(X, dtype=torch.float32))
        return nn.MSELoss()(outputs[:, -model.output_dim:], torch.tensor(y, dtype=torch.float32).unsqueeze(1)).item()
    
    initial_params = np.concatenate([layer.coefficients.detach().cpu().numpy().ravel() for layer in model.hcrnn.layers])
    result = minimize(objective, initial_params, method='L-BFGS-B', options={'maxiter': max_iter})
    
    optimized_params = result.x
    param_splits = np.split(optimized_params, [np.prod(layer.coefficients.shape) for layer in model.hcrnn.layers])
    for layer, layer_params in zip(model.hcrnn.layers, param_splits):
        layer.coefficients.data = torch.tensor(layer_params.reshape(layer.coefficients.shape), dtype=torch.float32)

def information_bottleneck_training(model: HCRKAN, X: np.ndarray, y: np.ndarray, beta: float = 1.0, max_iter: int = 100):
    def objective(params):
        param_splits = np.split(params, [np.prod(layer.coefficients.shape) for layer in model.hcrnn.layers])
        for layer, layer_params in zip(model.hcrnn.layers, param_splits):
            layer.coefficients.data = torch.tensor(layer_params.reshape(layer.coefficients.shape), dtype=torch.float32)
        
        outputs = model(torch.tensor(X, dtype=torch.float32))
        mse = nn.MSELoss()(outputs[:, -model.output_dim:], torch.tensor(y, dtype=torch.float32).unsqueeze(1)).item()
        
        mi_x_t = 0
        mi_t_y = 0
        for layer in model.hcrnn.layers:
            t = layer(torch.tensor(X, dtype=torch.float32))
            mi_x_t += torch.sum(torch.abs(torch.cov(torch.cat([torch.tensor(X, dtype=torch.float32), t], dim=1)))).item()
            mi_t_y += torch.sum(torch.abs(torch.cov(torch.cat([t, torch.tensor(y, dtype=torch.float32).unsqueeze(1)], dim=1)))).item()
        
        return mse - beta * (mi_t_y - mi_x_t)
    
    initial_params = np.concatenate([layer.coefficients.detach().cpu().numpy().ravel() for layer in model.hcrnn.layers])
    result = minimize(objective, initial_params, method='L-BFGS-B', options={'maxiter': max_iter})
    
    optimized_params = result.x
    param_splits = np.split(optimized_params, [np.prod(layer.coefficients.shape) for layer in model.hcrnn.layers])
    for layer, layer_params in zip(model.hcrnn.layers, param_splits):
        layer.coefficients.data = torch.tensor(layer_params.reshape(layer.coefficients.shape), dtype=torch.float32)

# Example usage
if __name__ == '__main__':
    # Generate sample data
    np.random.seed(42)
    X = np.random.rand(1000, 5)
    y = np.sin(X[:, 0] * np.pi) + 0.1 * np.random.randn(1000)
    
    # Create HCRKAN model
    model = HCRKAN(input_dim=5, hidden_layers=[10, 5], output_dim=1, m=3)
    
    # Train using different methods
    print("Training with standard backpropagation:")
    losses = train_hcrkan(model, X, y)
    
    print("\nTraining with direct estimation:")
    direct_estimation_training(model, X, y, m=3)
    
    print("\nTraining with tensor decomposition:")
    tensor_decomposition_training(model, X, y)
    
    print("\nTraining with information bottleneck:")
    information_bottleneck_training(model, X, y)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(torch.tensor(X).float()).numpy()
    
    print("\nSample predictions:")
    print(predictions[:5])