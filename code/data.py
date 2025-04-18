import torch
import pickle
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import StandardScaler

class HyperGraphDataset(Dataset):
    """Датасет для обработки гиперграфовых онтологических состояний"""
    def __init__(self, file_path, transform=None):
        super().__init__()
        self.file_path = file_path
        self.transform = transform
        
        # Загрузка и предобработка данных
        try:
            with open(file_path, 'rb') as f:
                self.data = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")
        
        # Нормализация признаков узлов
        self._normalize_features()

    def _normalize_features(self):
        """Применяет стандартную нормализацию к узлам"""
        all_features = [d.x.numpy() for d in self.data]
        scaler = StandardScaler()
        scaled = scaler.fit_transform(np.vstack(all_features))
        
        ptr = 0
        for i, d in enumerate(self.data):
            num_nodes = d.x.size(0)
            self.data[i].x = torch.FloatTensor(scaled[ptr:ptr+num_nodes])
            ptr += num_nodes

    def _hyperedge_to_clique(self, hyperedges):
        """Преобразует гиперребра в клики (для совместимости с GNN)"""
        edge_index = []
        for he in hyperedges:
            nodes = sorted(he)
            # Создаём полносвязный граф для гиперребра
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    edge_index.extend([(nodes[i], nodes[j]), (nodes[j], nodes[i])])
        return torch.LongTensor(edge_index).t().contiguous()

    def len(self):
        return len(self.data)

    def get(self, idx):
        data = self.data[idx]
        
        # Преобразование гиперграфа в обычный граф
        if not hasattr(data, 'edge_index'):
            data.edge_index = self._hyperedge_to_clique(data.hyperedges)
        
        # Добавление меток если отсутствуют
        if not hasattr(data, 'y'):
            data.y = torch.LongTensor([0])  # Заглушка
            
        return data

    def __getitem__(self, idx):
        data = self.get(idx)
        
        # Применение трансформаций
        if self.transform:
            data = self.transform(data)
            
        return data

def create_sample_dataset(output_path, num_graphs=100):
    """Генератор примеров данных для тестирования"""
    data_list = []
    
    for _ in range(num_graphs):
        # Случайный гиперграф
        num_nodes = np.random.randint(5, 15)
        num_hyperedges = np.random.randint(2, 5)
        
        # Признаки узлов
        x = torch.randn(num_nodes, 32)
        
        # Гиперребра (списки связанных узлов)
        hyperedges = [
            np.random.choice(num_nodes, size=np.random.randint(2,5), replace=False).tolist()
            for _ in range(num_hyperedges)
        ]
        
        # Метка класса (случайная)
        y = torch.LongTensor([np.random.randint(0, 3)])
        
        data = Data(
            x=x,
            hyperedges=hyperedges,
            y=y
        )
        data_list.append(data)
    
    # Сохранение в файл
    with open(output_path, 'wb') as f:
        pickle.dump(data_list, f)

if __name__ == "__main__":
    # Пример создания тестовых данных
    create_sample_dataset("sample_data.pkl")
    
    # Проверка загрузки
    dataset = HyperGraphDataset("sample_data.pkl")
    sample = dataset[0]
    print(f"Пример графа:")
    print(f"Узлы: {sample.x.shape}")
    print(f"Ребра: {sample.edge_index.shape}")
    print(f"Метка: {sample.y.item()}")