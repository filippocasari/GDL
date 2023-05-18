import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg

#Data(x=[5000, 1], pos=[5000, 3], file_id='image_0021.bin', label=[1], y=[1], bbox=[1, 5], edge_index=[2, 134022], edge_attr=[134022, 3])

class Compose(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, data):
        for aug in self.transforms:
            data = aug(data)
        return data

class NodeDrop(nn.Module):
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p

    def forward(self, data):
        x = data.x
        y = data.y
       
        train_mask=torch.ones(x.size(0)).to(torch.cuda.current_device())
        idx = torch.empty(x.size(0)).uniform_(0, 1)
        train_mask[torch.where(idx < self.p)] = 0

        x= x * train_mask

        new_data=data.clone()
        new_data.x=x
        new_data.y=y
       
        return new_data

class EdgeDrop(nn.Module):
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p

    def forward(self, data):
        x = data.x
        y = data.y
        
        edge_idx = data.edge_index

        edge_idx = edge_idx.permute(1, 0)
        idx = torch.empty(edge_idx.size(0)).uniform_(0, 1)
        edge_idx = edge_idx[torch.where(idx >= self.p)].permute(1, 0)
        new_data=data.clone()

        new_data.x=x
        new_data.edge_index=edge_idx
        new_data.y=y
        return new_data

class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, data):
        x = data.x
        y = data.y
        
        x = F.normalize(x)
        new_data=data.clone()
        new_data.x=x
        new_data.y=y
        
        return new_data

class NodeMixUp(nn.Module):
    def __init__(self, lamb, classes):
        super().__init__()
        self.lamb = lamb
        self.classes = classes

    def forward(self, data):
        x = data.x
        y = data.y
        
    
        n, d = x.shape

        pair_idx = torch.randperm(n)
        x_b = x[pair_idx]
        y_b = y[pair_idx]
        y_a_oh = F.one_hot(y, self.classes)
        y_b_oh = F.one_hot(y_b, self.classes)

        x_mix = (self.lamb * x) + (1-self.lamb)*x_b
        y_mix = (self.lamb * y_a_oh) + (1 - self.lamb) * y_b_oh
        new_y = y_mix.argmax(1)

        # new_x = torch.vstack([x, x_mix])
        # new_y = torch.vstack([y_a_oh, y_mix])

        
        #new_data = tg.data.Data(x=x_mix, y=new_y, edge_index=edge_idx, train_mask=train_mask, test_mask=test_mask)
        new_data=data.clone()
        new_data.x=x_mix
        new_data.y=new_y
     
        return new_data

class NodeFeatureMasking(nn.Module):
    def __init__(self, p=0.15):
        super().__init__()
        self.p = p

    def forward(self, data):
        x = data.x
        y = data.y
       
        n, d = x.shape
        
        idx = torch.empty((d,), dtype=torch.float32).uniform_(0, 1) < self.p
        x = x.clone()
        x[:, idx] = 0

        #new_data = tg.data.Data(x=x, y=y, edge_index=edge_idx, train_mask=train_mask, test_mask=test_mask, edge_attr=edge_attr)
        new_data=data.clone()
        new_data.x=x
        new_data.y=y
     
        edge_attr=edge_attr
        return new_data

class EdgeFeatureMasking(nn.Module):
    def __init__(self, p=0.15):
        super().__init__()
        self.p = p

    def forward(self, data):
        x = data.x
        y = data.y
        edge_attr = data.edge_attr
        
     
        n, d = edge_attr.shape
        
        idx = torch.empty((d,), dtype=torch.float32).uniform_(0, 1) < self.p
        edge_attr = edge_attr.clone()
        edge_attr[:, idx] = 0

        #new_data = tg.data.Data(x=x, y=y, edge_index=edge_idx, train_mask=train_mask, test_mask=test_mask, edge_attr=edge_attr)
        new_data=data.clone()
        new_data.x=x
        new_data.y=y
        new_data.edge_attr=edge_attr
        return new_data 