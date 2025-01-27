from torch import load, no_grad
import numpy as np

class segmentation():
    def __init__(
            self, 
            model_path, 
            use_cuda=False
            ):
        self.model_path = model_path
        self.use_cuda = use_cuda
        self.load_model()


    def load_model(self):
        self.model = load(self.model_path)
        
    def segment(self, image):
        if self.use_cuda:
            self.model.cuda()
            image = image.cuda()
        self.model.eval()
        
        with no_grad():
            pr_mask = np.argmax(self.model.forward(image).squeeze().cpu().numpy(), axis=0)
        
        return pr_mask

