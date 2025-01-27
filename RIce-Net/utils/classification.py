import numpy as np

from torchvision.models import resnet50
from torch import nn, load, no_grad


class classification():
    def __init__(
            self, 
            model_path, 
            classes,
            use_cuda=False
            ):
        self.model_path = model_path
        self.num_classes = len(classes)
        self.classes = classes
        self.use_cuda = use_cuda
        self.load_model()


    def load_model(self):
        model = resnet50()
        fc_inputs = model.fc.in_features
        model.fc = nn.Linear(fc_inputs, self.num_classes)
        model.load_state_dict(load(self.model_path).state_dict())
        self.model = model


    def classify(self, image):
        if self.use_cuda:
            self.model.cuda()
            image = image.cuda()

        self.model.eval()

        with no_grad():
            pr_label = self.model.forward(image).cpu().numpy()

            # save class probabilities
            probs_batch = np.array([[np.exp(l)/sum(np.exp(batch)) for l in batch] for batch in pr_label])

            # save class predictions
            preds_batch = np.argmax(pr_label, axis=1)
        
        return preds_batch, pr_label, probs_batch

