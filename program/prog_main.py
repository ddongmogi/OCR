from utils import Averager
from tqdm import tqdm
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Program(object):
    def __init__(self, conf):
        self.lr = conf['learning_rate']
        self.epochs = conf['epochs']
        self.batch_size = conf['batch_size']
        
        if not os.path.exists(conf['save_path']):
            os.makedirs(conf['save_path'])
            
        self.save_path = conf['save_path']
        self.data_setting = {
            'row_max':conf['row_max']
        }
        if conf['paragraph_type']:
            self.data_setting['col_count'] = conf['col_count']
        self.paragraph_type = conf['paragraph_type']
        
    def train(self,model,dataloader):
        
        classes = model.classes
        model = torch.nn.DataParallel(model).to(device)
        model.train()
        
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
        loss_avg = Averager()
        
        filtered_parameters = []
        params_num = []
        for p in filter(lambda p: p.requires_grad, model.parameters()):
            filtered_parameters.append(p)
            params_num.append(np.prod(p.size()))
        print('Trainable params num : ', sum(params_num))
        
        optimizer = optim.Adam(filtered_parameters, lr=self.lr, betas=(0.9, 0.999))
        best_loss = 100
        
        with tqdm(range(self.epochs),unit="epoch") as tepoch:
            for epoch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                
                if self.paragraph_type:
                    batch_sampler = dataloader.batch_generator(self.batch_size,\
                                                self.data_setting['row_max'],self.data_setting['col_count'])
                else:
                    batch_sampler = dataloader.batch_generator(self.batch_size,\
                                                self.data_setting['row_max'])
                img = batch_sampler[0]
                text = batch_sampler[1][0]
                length = batch_sampler[1][1]
                
                preds  = model(img,text[:, :-1],max(length).cpu().numpy())
                
                target = text[:, 1:]
                cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

                model.zero_grad()
                cost.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5)  # gradient clipping with 5 (Default)
                optimizer.step()

                loss_avg.add(cost)
                pred_max = torch.argmax(F.softmax(preds,dim=2).view(self.batch_size,-1,classes),2)
                
                
                tepoch.set_postfix(pred=dataloader.converter.decode(pred_max,length),loss=loss_avg.val().item())
               
                del batch_sampler
                
                if loss_avg.val().item() < best_loss and best_loss < 2:
                    best_loss = loss_avg.val().item()
                    torch.save(model.state_dict(), os.path.join(self.save_path,f'model_{self.lr}_{self.epochs}.pth'))


        
        
    def test(self):
        print('test')