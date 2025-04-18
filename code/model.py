import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

class HyperGraphEncoder(nn.Module):
    """GNN-энкодер для онтологических состояний в виде гиперграфов"""
    def __init__(self, node_dim, edge_dim, latent_dim=128):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.latent_dim = latent_dim

        # Графовые слои с механизмом внимания
        self.conv1 = GATv2Conv(node_dim, 64, edge_dim=edge_dim, heads=4)
        self.conv2 = GATv2Conv(64*4, latent_dim, edge_dim=edge_dim, heads=1)
        
        # Нормализация
        self.norm1 = nn.LayerNorm(64*4)
        self.norm2 = nn.LayerNorm(latent_dim)

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Первый графовый слой
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = self.norm1(x)
        
        # Второй графовый слой
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.norm2(x)
        
        # Глобальное усреднение
        return torch.mean(x, dim=0)

class CognitiveFunctor(nn.Module):
    """Функтор проекции онтологического пространства в когнитивное"""
    def __init__(self, input_dim, hidden_dim=256, output_dim=64):
        super().__init__()
        
        # Нелинейное преобразование
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Квантование через векторное кодирование
        self.codebook = nn.Parameter(torch.randn(128, output_dim))
        
        # Регуляризация
        self.dropout = nn.Dropout(0.1)

    def quantize(self, z):
        """Векторное квантование (аналог когнитивного округления)"""
        distances = (torch.sum(z**2, dim=1, keepdim=True) 
                 + torch.sum(self.codebook**2, dim=1)
                 - 2 * torch.matmul(z, self.codebook.t())
        
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(encoding_indices, self.codebook)
        return quantized

    def forward(self, phi):
        # Проекция в латентное пространство
        h = F.elu(self.fc1(phi))
        h = self.dropout(h)
        z = self.fc2(h)
        
        # Когнитивное квантование
        z_q = self.quantize(z)
        
        # Straight-through estimator
        z_q = phi + (z_q - phi).detach()
        return z_q

class OntologicalProjection(nn.Module):
    """Полная модель проекции с потерей информации"""
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.encoder = HyperGraphEncoder(node_dim, edge_dim)
        self.functor = CognitiveFunctor(128)  # latent_dim GNN -> 64

    def forward(self, data):
        # Кодирование гиперграфа
        phi = self.encoder(data)
        
        # Применение когнитивного функтора
        x = self.functor(phi)
        return x

# Пример использования
if __name__ == "__main__":
    # Тестовые данные (случайный гиперграф)
    num_nodes = 10
    node_dim = 32
    edge_dim = 16
    
    x = torch.randn(num_nodes, node_dim)
    edge_index = torch.randint(0, num_nodes, (2, 30))
    edge_attr = torch.randn(30, edge_dim)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # Инициализация модели
    model = OntologicalProjection(node_dim, edge_dim)
    
    # Forward pass
    x_proj = model(data)
    print(f"Input phi dim: {num_nodes}x{node_dim}")
    print(f"Output x dim: {x_proj.shape}")  # Ожидается: torch.Size([64])