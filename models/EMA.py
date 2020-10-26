#https://www.programmersought.com/article/28492072406/
from copy import deepcopy

class EMA:
    def __init__(self, model, decay = 0.999, device='cpu'):
        self.model  = model
        self.ema = deepcopy(model)
        self.decay  = decay

    def re_init(self, model):
        self.model = model
        self.ema = deepcopy(model)

    def update(self):
        for (model_name, model_param),(ema_name, ema_param) in zip(self.model.named_parameters(), self.ema.named_parameters()):
            if model_name != ema_name:
                print("Incorect order")
            if model_param.requires_grad:
                new_average = (1.0 - self.decay) * model_param.data + self.decay * ema_param.data
                ema_param.data = new_average.clone()

    def get(self):
        return self.ema
    
    def __call__(self, i_tensor):
        return self.ema(i_tensor)



    