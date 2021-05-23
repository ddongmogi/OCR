
import torch.nn as nn
from models.rec import modules as md

class Model(nn.Module):
    def __init__(self,conf,num_target):
        super(Model,self).__init__()
        conf = conf['Model']
        self.hidden_size = conf['hidden_size']
        self.output_size = conf['output_size']
        self.input_size = conf['input_size']
        self.classes = num_target
        
        self.use_tp = conf['use_tp']
        if self.use_tp:
            self.tp = md.transformation.TPS_SpatialTransformerNetwork(F=20,I_size=(32,100),I_r_size=(32,100),I_channel_num=3)

        self.fe = md.feature_extraction.ResNet_FeatureExtractor(self.input_size,self.output_size)
        self.ap = nn.AdaptiveAvgPool2d((None, 1))

        self.sm = nn.Sequential(
        md.sequence_modeling.BidirectionalLSTM(self.output_size, self.hidden_size, self.hidden_size),
        md.sequence_modeling.BidirectionalLSTM(self.hidden_size, self.hidden_size, self.hidden_size))

        self.pd = md.prediction.Attention(self.hidden_size, self.hidden_size, self.classes)


    def forward(self,img,text,max_batch=25,is_train=True):
        if self.use_tp:
            img = self.tp(img)
            
        x = self.fe(img)
        x = self.ap(x.permute(0, 3, 1, 2))
        x = self.sm(x.squeeze(3))
        
        x = self.pd(x.contiguous(),text,is_train,max_batch-1)
        return x
        
