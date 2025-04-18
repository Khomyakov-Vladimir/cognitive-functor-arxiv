import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import OntologicalProjection
from data import HyperGraphDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Trainer:
    def __init__(self, config):
        self.device = torch.device(config['device'])
        self.model = OntologicalProjection(
            node_dim=config['node_dim'],
            edge_dim=config['edge_dim']
        ).to(self.device)
        
        # Инициализация оптимизатора и планировщика
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs']
        )
        
        # Загрузка данных
        self.train_loader = DataLoader(
            HyperGraphDataset(config['train_path']),
            batch_size=config['batch_size'],
            shuffle=True
        )
        self.val_loader = DataLoader(
            HyperGraphDataset(config['val_path']),
            batch_size=config['batch_size']
        )
        
        # Конфигурация
        self.config = config
        self.best_loss = float('inf')
        
    def compute_loss(self, batch, mode='train'):
        # Forward pass
        x_proj = self.model(batch)
        
        # 1. Reconstruction Loss (ELBO)
        recon_loss = F.mse_loss(
            self.model.functor.quantize(x_proj),
            x_proj.detach()
        )
        
        # 2. KL-Divergence
        mu, logvar = self.model.encoder.mu, self.model.encoder.logvar
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 3. Categorical Loss (коммутативность)
        U = self.get_random_transform()
        F_Ux = self.model(self.transform_batch(batch, U))
        UF_x = U @ self.model(batch)
        cat_loss = F.mse_loss(F_Ux, UF_x)
        
        # Общий лосс
        total_loss = (
            self.config['recon_weight'] * recon_loss +
            self.config['kl_weight'] * kl_loss +
            self.config['cat_weight'] * cat_loss
        )
        
        # Логирование
        if mode == 'train':
            self.logger.log({
                'loss': total_loss.item(),
                'recon_loss': recon_loss.item(),
                'kl_loss': kl_loss.item(),
                'cat_loss': cat_loss.item()
            })
        
        return total_loss
    
    def transform_batch(self, batch, U):
        """Применяет преобразование U ко всем нодам в батче"""
        batch.x = torch.matmul(batch.x, U.T)
        return batch
    
    def get_random_transform(self):
        """Генерирует случайное унитарное преобразование"""
        dim = self.config['node_dim']
        random = torch.randn(dim, dim)
        return torch.linalg.qr(random)[0].to(self.device)
    
    def train_epoch(self):
        self.model.train()
        progress = tqdm(self.train_loader, desc='Training')
        
        for batch in progress:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            loss = self.compute_loss(batch)
            loss.backward()
            
            # Градиентный clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['max_grad_norm']
            )
            
            self.optimizer.step()
            progress.set_postfix({'loss': loss.item()})
        
        self.scheduler.step()
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                loss = self.compute_loss(batch, mode='val')
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Сохранение лучшей модели
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            torch.save({
                'model_state': self.model.state_dict(),
                'config': self.config
            }, f"{self.config['save_dir']}/best_model.pt")
        
        return avg_loss
    
    def visualize_projection(self, epoch):
        """Визуализация t-SNE проекций"""
        self.model.eval()
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                x_proj = self.model(batch)
                embeddings.append(x_proj.cpu())
                labels.append(batch.y.cpu())
        
        embeddings = torch.cat(embeddings).numpy()
        labels = torch.cat(labels).numpy()
        
        # t-SNE преобразование
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2).fit_transform(embeddings)
        
        # Визуализация
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            tsne[:, 0], tsne[:, 1],
            c=labels, cmap='viridis',
            alpha=0.6
        )
        plt.colorbar(scatter)
        plt.title(f't-SNE Projection (Epoch {epoch})')
        plt.savefig(f"{self.config['save_dir']}/tsne_epoch_{epoch}.png")
        plt.close()

class Logger:
    """Логирование метрик в TensorBoard и консоль"""
    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)
    
    def log(self, metrics):
        for key, value in metrics.items():
            self.writer.add_scalar(key, value)
        self.writer.flush()

if __name__ == "__main__":
    # Конфигурация обучения
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'node_dim': 32,
        'edge_dim': 16,
        'latent_dim': 64,
        'lr': 3e-4,
        'weight_decay': 1e-5,
        'epochs': 100,
        'batch_size': 32,
        'recon_weight': 1.0,
        'kl_weight': 0.1,
        'cat_weight': 0.05,
        'max_grad_norm': 1.0,
        'train_path': 'data/train_dataset.pkl',
        'val_path': 'data/val_dataset.pkl',
        'save_dir': 'checkpoints',
        'log_dir': 'logs'
    }
    
    # Инициализация
    trainer = Trainer(config)
    logger = Logger(config['log_dir'])
    trainer.logger = logger
    
    # Цикл обучения
    for epoch in range(1, config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        
        trainer.train_epoch()
        val_loss = trainer.validate()
        
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Визуализация каждые 10 эпох
        if epoch % 10 == 0:
            trainer.visualize_projection(epoch)