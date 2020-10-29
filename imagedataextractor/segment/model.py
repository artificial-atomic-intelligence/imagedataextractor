import torch
import numpy as np
from PIL import Image

from .cluster import Cluster
from .nnmodules import BranchedERFNet
from .uncertainty import expected_entropy, predictive_entropy, uncertainty_filtering

class ParticleSegmenter:

    def __init__(self, bayesian=True, n_samples=30, device='cpu'):
        self.bayesian = bayesian
        self.n_samples = n_samples
        self.seg_model = BranchedERFNet(num_classes=[4, 1]).to(device).eval()
        self.model_path = '/home/by256/Documents/Projects/imagedataextractor/imagedataextractor/models/seg-model.pt'
        self.seg_model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.cluster = Cluster(n_sigma=2, device=device)
        self.device = device

    def preprocess(self, image):
        image = Image.fromarray(image)
        image = image.resize((512, 512), resample=Image.BICUBIC)
        image = np.array(image)
        image = image / 255.0
        return image

    def postprocess_pred(self, image, h, w):
        image = Image.fromarray(image)
        image = image.resize((w, h), resample=Image.NEAREST)
        return np.array(image)

    def postprocess_uncertainty(self, image, h, w):
        """
        Resize uncertainty map. This is strictly for visualisation purposes.
        The output of this function will not be used for anything other 
        than visualisation.
        """
        image = Image.fromarray(image)
        image = image.resize((w, h), resample=Image.BICUBIC)
        return np.array(image)

    def enable_eval_dropout(self):
        for module in self.seg_model.modules():
            if 'Dropout' in type(module).__name__:
                module.train()

    def monte_carlo_predict(self, image):
        h, w = image.shape[-2:]
        cluster = Cluster(n_sigma=2, h=h, w=w, device=self.device)
        self.enable_eval_dropout()

        # get monte carlo model samples
        mc_outputs = []
        mc_seed_maps = []
        for i in range(self.n_samples):
            output = self.seg_model(image).detach()
            seed_map = torch.sigmoid(output[0, -1]).unsqueeze(0)
            mc_outputs.append(output)
            mc_seed_maps.append(seed_map)

        mc_outputs = torch.cat(mc_outputs, dim=0)
        mc_seed_maps = torch.cat(mc_seed_maps, dim=0)

        # MC prediction (cluster mean on MC samples)
        mc_prediction, _ = cluster.cluster(mc_outputs.mean(dim=0))

        # Uncertainty
        total = predictive_entropy(mc_seed_maps)
        aleatoric = expected_entropy(mc_seed_maps)
        epistemic = total - aleatoric  # $MI(y, \theta | x)$

        return mc_prediction, epistemic

    def segment(self, image):
        o_h, o_w = image.shape[:2]
        image = self.preprocess(image)
        image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.bayesian:
            # monte carlo predict
            pred, uncertainty = self.monte_carlo_predict(image)
            pred = uncertainty_filtering(pred, uncertainty)
            pred = pred.cpu().numpy()
            uncertainty = uncertainty.cpu().numpy()
            # post-process nuncertainty for visualisation
            uncertainty = self.postprocess_uncertainty(uncertainty, o_h, o_w)
        else:
            model_out = self.seg_model(image)[0].detach()
            pred = self.cluster.cluster(model_out)[0].cpu().numpy()
            uncertainty = None
        pred = self.postprocess_pred(pred, o_h, o_w)
        return pred, uncertainty
