from utils import Averager

class Program(object):
    def __init__(self, conf):
        self.lr = conf['learning_rate']
        self.epochs = conf['epochs']
        self.batch_size = conf['batch_size']
        
        
    def train(self,model,dataloader):
        print('train')
        
    def test(self):
        print('test')     