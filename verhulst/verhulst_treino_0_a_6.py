import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Swish activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Solução analítica de Verhulst / Analytical solution of the Verhulst equation
def verhulst_solution(t, y0=0.1, r=1.0, K=1.0):
    return K / (1 + ((K - y0) / y0) * torch.exp(-r * t))

# Rede neural profunda / Neural network model
class NeuralODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            Swish(),
            nn.Linear(128, 128),
            Swish(),
            nn.Linear(128, 128),
            Swish(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

# Perda supervisionada + penalização da condição inicial
# Loss: supervised in [0, limit] + initial condition
def loss_function(model, x_sup, y_sup, x_ic, y_ic):
    y_pred = model(x_sup)
    loss_sup = torch.mean((y_pred - y_sup)**2)
    y_ic_pred = model(x_ic)
    loss_ic = (y_ic_pred - y_ic)**2
    return loss_sup + loss_ic

# Métricas / Evaluation metrics
def evaluate_metrics(y_true, y_pred):
    y_true = [y.item() for y in y_true]
    y_pred = [y.item() for y in y_pred]
    n = len(y_true)
    mse = sum((yt - yp)**2 for yt, yp in zip(y_true, y_pred)) / n
    mae = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n
    mean_y = sum(y_true) / n
    ss_tot = sum((yt - mean_y)**2 for yt in y_true)
    ss_res = sum((yt - yp)**2 for yt, yp in zip(y_true, y_pred))
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    return {
        "RMSE": mse**0.5,
        "MAE": mae,
        "R2": r2
    }

# Parâmetros / Parameters
r, K, y0 = 1.0, 1.0, 0.1
limite = 6.0

# Model and training
# Dados de entrada e referência 
x_full = torch.linspace(0, 10, 3000).reshape(-1, 1)
y_full = verhulst_solution(x_full, y0, r, K)

# Supervisão apenas no intervalo [0, 6]
x_sup = x_full[x_full <= limite].reshape(-1, 1)  
y_sup = y_full[x_full <= limite].reshape(-1, 1)

# Condição inicial
x_ic = torch.tensor([[0.0]])
y_ic = torch.tensor([[y0]])

torch.manual_seed(2)
# Modelo e otimizador
model = NeuralODE()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Treinamento
for epoch in range(3000):
    optimizer.zero_grad()
    loss = loss_function(model, x_sup, y_sup, x_ic, y_ic)
    loss.backward()
    optimizer.step()

# Avaliação / Evaluation
with torch.no_grad():
    y_pred = model(x_full)

metrics = evaluate_metrics(y_full, y_pred)
print(f"RMSE: {metrics['RMSE']:.6f}, MAE: {metrics['MAE']:.6f}, R²: {metrics['R2']:.4f}")

# Gráfico
plt.figure(figsize=(8, 4))
plt.plot(x_full, y_full, label="Analytical Solution", color="black")
plt.plot(x_full, y_pred, '--', label="Neural Network — trained on [0, 6]")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Supervised training on [0, 6] + Initial Condition")
plt.legend()
plt.grid(True)

# Exibir as métricas na parte inferior
plt.text(0.05, 0.05,
         f"RMSE = {metrics['RMSE']:.4f} MAE = {metrics['MAE']:.4f} R² = {metrics['R2']:.4f}",
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom',
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

plt.tight_layout()
plt.show()