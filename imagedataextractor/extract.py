# -*- coding: utf-8 -*-
"""
Main extraction modules for imagedataextractor.

.. codeauthor:: Batuhan Yildirim <by256@cam.ac.uk>
"""

import os
import cv2
import imghdr
import logging
import numpy as np
from PIL import Image
from skimage import img_as_ubyte
import torch

from .data import EMData
from .analysis import ShapeDetector
from .analysis.particlesize import aspect_ratio
from .scalebar import ScalebarDetector
from .segment import ParticleSegmenter
from .utils import shuffle_segmap
from ads.analysis.segment.empatches import EMPatches
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

log = logging.getLogger(__name__)


def extract(input_path: str | None = None, image: np.ndarray | None = None, seg_bayesian=True, seg_n_samples=30, seg_tu=0.0125, seg_device='cpu', scalebar:bool=True, patching:bool=False):
    """
    Extract from single image, single doc or directory of images/docs.
    
    Parameters
    ----------
    seg_bayesian: bool
        Option to use Bayesian inference in segmentation model. Trades off speed
        for accuracy (recommended) (default is True).
    seg_n_samples: int
        Number of monte carlo samples used for Bayesian inference in segmentation model
        (default is 30).
    seg_tu: float
        Uncertainty threshold beyond which to filter particles (default is 0.0125).
    seg_device: str {'cpu', 'cuda', None}
        Selected device to run segmentation model inference on. If None, will select 
        'cuda' if a GPU is available, otherwise will default to 'cpu' (default is 'cpu').
    """

    if input_path is not None and os.path.isfile(input_path):
        if imghdr.what(input_path) is not None:
            log.info('Input is an image of type {}.'.format(imghdr.what(input_path)))
            fn = os.path.splitext(input_path)[0].split('/')[-1]
            # data = []
            image = Image.open(input_path)
            if image.mode == 'I;16':
                image_8bit = image.point(lambda i: i * (1./256)).convert('L')
                rgb_image = Image.merge("RGB", (image_8bit, image_8bit, image_8bit))
                image = np.array(rgb_image)
            else:
                image = np.array(image)

    elif input_path is None and image is not None:
        if image.dtype != np.uint8:
            image = img_as_ubyte(image)
        fn = 'image'

    if image.ndim != 3:
        image = np.stack((image,)*3, axis=-1)
    
    if patching:
        best_patch_size = detect_best_patch_size(image)
        log.info('Best patch size detected: {}'.format(best_patch_size))
        em_data = _patched_extract_image(image=image, patch_size=best_patch_size, seg_bayesian=seg_bayesian, seg_n_samples=seg_n_samples, seg_tu=seg_tu, seg_device=seg_device)
    else:
        em_data = _extract_image(image=image, seg_bayesian=seg_bayesian, seg_n_samples=seg_n_samples, seg_tu=seg_tu, seg_device=seg_device, scalebar=scalebar)

    em_data.fn = fn
    # data.append(em_data)
    
    return em_data

def _extract_image(image, seg_bayesian=True, seg_n_samples=30, seg_tu=0.0125, seg_device='cpu', scalebar:bool=True):
    """
    Extract from a single image (not a panel).

    Parameters
    ----------
    image: np.ndarray
        Image to perform extraction on.
    seg_bayesian: bool
        Option to use Bayesian inference in segmentation model. Trades off speed
        for accuracy (recommended) (default is True).
    seg_n_samples: int
        Number of monte carlo samples used for Bayesian inference in segmentation model
        (default is 30).
    seg_tu: float
        Uncertainty threshold beyond which to filter particles (default is 0.0125).
    device: str {'cpu', 'cuda', None}
        Selected device to run segmentation model inference on. If None, will select 
        'cuda' if a GPU is available, otherwise will default to 'cpu' (default is 'cpu').
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    em_data = EMData()

    if scalebar:
        # detect scalebar
        sb_detector = ScalebarDetector()
        scalebar = sb_detector.detect(image)
        em_data.scalebar = scalebar
        text, units, conversion, scalebar_contour = scalebar.data
        if conversion is not None:
            log.info('Scalebar detection successful.')
        else:
            log.info('Scalebar detection failed. Measurements will be given in units of pixels.')
    else:
        text, units, conversion, scalebar_contour = None, None, None, None
    
    # initialise detectors
    segmenter = ParticleSegmenter(bayesian=seg_bayesian, n_samples=seg_n_samples, tu=seg_tu, device=seg_device)
    shape_detector = ShapeDetector()

    # initialise EM data object
    if not seg_bayesian:
        del em_data.data['uncertainty']

    em_data.image = image

    # segment particles
    segmentation, uncertainty, original = segmenter.segment(image)
    segmentation = shuffle_segmap(segmentation)
    em_data.segmentation = segmentation
    em_data.uncertainty = uncertainty
    if len(np.unique(segmentation)) > 1:
        log.info('Particle segmentation successful.')
    else:
        log.info('Particle segmentation was completed but no particles were found.')

    # extract particle measures
    for inst in np.unique(segmentation):
        if inst == 0.0:  # 0 is always background
            continue
        em_data.data['idx'].append(inst)
        em_data.data['original_units'].append(units)

        inst_mask = (segmentation == inst).astype(np.uint8)
        # area
        area = np.sum(inst_mask)  # pixels
        em_data.data['area (pixels^2)'].append(area)
        if conversion is not None:
            area = area * conversion**2  # meters^2
            em_data.data['area_units'].append('meters^2')
            em_data.data['diameter_units'].append('meters')
        else:
            em_data.data['area_units'].append('pixels^2')
            em_data.data['diameter_units'].append('pixels')
        em_data.data['area'].append(area)

        # center
        coords = np.argwhere(inst_mask == 1.0)
        center = coords.mean(axis=0, dtype=int)  # (y, x)
        em_data.data['center'].append(center)

        # edge
        edge_cond = (0 in coords) | (image.shape[0]-1 in coords[:, 0]) | (image.shape[1]-1 in coords[:, 1])
        em_data.data['edge'].append(edge_cond)

        # contours
        contours, _ = cv2.findContours(inst_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        em_data.data['contours'].append(contours)

        # aspect ratio
        ar = aspect_ratio(contours[0])
        em_data.data['aspect_ratio'].append(ar)

        # shape estimate
        shape_estimate, shape_distances = shape_detector.match_shapes(inst_mask)
        if shape_estimate == 'circle':
            em_data.data['shape_estimate'].append(shape_estimate)
            # diameter
            diameter = 2*np.sqrt(area/np.pi)
            em_data.data['diameter'].append(diameter)
        else:
            em_data.data['shape_estimate'].append(None)
            em_data.data['diameter'].append(None)

        # particle instance uncertainty
        if seg_bayesian:
            inst_uncertainty = np.mean(uncertainty[inst_mask.astype(bool)])
            em_data.data['uncertainty'].append(inst_uncertainty)
        
    if len(em_data) > 0:
        log.info('Extraction successful - Found {} particles.'.format(len(em_data)))
    else:
        log.info('Extraction failed - no particles were found.')

    return em_data


def detect_best_patch_size(image, patch_sizes=[512, 1024, 1536], num_trials: int = 5):
    
    aggregate_uncertainty = []
    for patch_size in patch_sizes:

        if image.shape[0] < patch_size or image.shape[1] < patch_size:
            patch_size = min(image.shape[0]-1, image.shape[1]-1)

        uncertainty = []
        for _ in range(num_trials):
            
            x, y = np.random.randint(0, image.shape[0]-patch_size), np.random.randint(0, image.shape[1]-patch_size)
            patch = image[x:x+patch_size, y:y+patch_size]
            data = _extract_image(patch, seg_bayesian=True, seg_n_samples=30, seg_tu=0.1, seg_device=device, scalebar=False)
            uncertainty.append(np.mean(data.uncertainty))

        aggregate_uncertainty.append(np.mean(uncertainty))
        print(aggregate_uncertainty)

    best_patch_size = patch_sizes[np.argmin(aggregate_uncertainty)]

    return best_patch_size

def _patched_extract_image(image, patch_size: int = 512, seg_bayesian=True, seg_n_samples=30, seg_tu=0.0125, seg_device='cpu'):
    """
    Extract from a single image (not a panel).

    Parameters
    ----------
    image: np.ndarray
        Image to perform extraction on.
    seg_bayesian: bool
        Option to use Bayesian inference in segmentation model. Trades off speed
        for accuracy (recommended) (default is True).
    seg_n_samples: int
        Number of monte carlo samples used for Bayesian inference in segmentation model
        (default is 30).
    seg_tu: float
        Uncertainty threshold beyond which to filter particles (default is 0.0125).
    device: str {'cpu', 'cuda', None}
        Selected device to run segmentation model inference on. If None, will select 
        'cuda' if a GPU is available, otherwise will default to 'cpu' (default is 'cpu').
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    em_data = EMData()

    text, units, conversion, scalebar_contour = None, None, None, None
    
    # initialise detectors
    segmenter = ParticleSegmenter(bayesian=seg_bayesian, n_samples=seg_n_samples, tu=seg_tu, device=seg_device)
    shape_detector = ShapeDetector()

    # initialise EM data object
    if not seg_bayesian:
        del em_data.data['uncertainty']

    em_data.image = image

    # segment particles
    patches, indices = EMPatches().extract_patches(img=image, patchsize=patch_size, overlap=0.1)
    patch_uncertainty = []
    patch_results = []
    for i, patch in enumerate(patches):
        segmentation, uncertainty, original = segmenter.segment(patch)
        segmentation = shuffle_segmap(segmentation)

        patch_results.append(segmentation)
        patch_uncertainty.append(uncertainty)

    zero_max = 0 
    for i, r in enumerate(patch_results):
        r[r> 0] = r[r> 0] + zero_max
        zero_max = r.max()
        patch_results[i] = r

    merged_segmentation = EMPatches().merge_patches(patch_results, indices, mode='mask')
    merged_uncertainty = EMPatches().merge_patches(patch_uncertainty, indices, mode='avg')
    em_data.segmentation = merged_segmentation
    em_data.uncertainty = merged_uncertainty

    # extract particle measures
    for inst in np.unique(merged_segmentation):
        if inst == 0.0:  # 0 is always background
            continue
        em_data.data['idx'].append(inst)
        em_data.data['original_units'].append(units)

        inst_mask = (merged_segmentation == inst).astype(np.uint8)
        # area
        area = np.sum(inst_mask)  # pixels
        em_data.data['area (pixels^2)'].append(area)
        if conversion is not None:
            area = area * conversion**2  # meters^2
            em_data.data['area_units'].append('meters^2')
            em_data.data['diameter_units'].append('meters')
        else:
            em_data.data['area_units'].append('pixels^2')
            em_data.data['diameter_units'].append('pixels')
        em_data.data['area'].append(area)

        # center
        coords = np.argwhere(inst_mask == 1.0)
        center = coords.mean(axis=0, dtype=int)  # (y, x)
        em_data.data['center'].append(center)

        # edge
        edge_cond = (0 in coords) | (image.shape[0]-1 in coords[:, 0]) | (image.shape[1]-1 in coords[:, 1])
        em_data.data['edge'].append(edge_cond)

        # contours
        contours, _ = cv2.findContours(inst_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        em_data.data['contours'].append(contours)

        # aspect ratio
        ar = aspect_ratio(contours[0])
        em_data.data['aspect_ratio'].append(ar)

        # shape estimate
        shape_estimate, shape_distances = shape_detector.match_shapes(inst_mask)
        if shape_estimate == 'circle':
            em_data.data['shape_estimate'].append(shape_estimate)
            # diameter
            diameter = 2*np.sqrt(area/np.pi)
            em_data.data['diameter'].append(diameter)
        else:
            em_data.data['shape_estimate'].append(None)
            em_data.data['diameter'].append(None)

        # particle instance uncertainty
        if seg_bayesian:
            inst_uncertainty = np.mean(merged_uncertainty[inst_mask.astype(bool)])
            em_data.data['uncertainty'].append(inst_uncertainty)
        
    if len(em_data) > 0:
        log.info('Extraction successful - Found {} particles.'.format(len(em_data)))
    else:
        log.info('Extraction failed - no particles were found.')

    return em_data