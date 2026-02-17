import importlib
import warnings

import torch
import numpy as np
import cv2

from .hilacam import hilacam_clip, hilacam_clipvip
from .ritsm import ritsm_clip, ritsm_clipvip
from .mhsa import mhsa_clip, mhsa_clipvip


def _load_pytorch_grad_cam_methods():
    method_map = {
        "gradcam": "GradCAM",
        "scorecam": "ScoreCAM",
        "gradcam++": "GradCAMPlusPlus",
        "ablationcam": "AblationCAM",
        "xgradcam": "XGradCAM",
        "eigencam": "EigenCAM",
        "eigengradcam": "EigenGradCAM",
        "layercam": "LayerCAM",
        "groupcam": "GroupCAM",
        "sscam1": "SSCAM1",
        "sscam2": "SSCAM2",
        "rawcam": "RawCAM",
        "gradientcam": "GradientCAM",
        "vitgradcam": "ViTGradCAM",
        "testhila": "HilaCAM",
        "gscorecam": "GScoreCAM",
        "rise": "RiseCAM",
        "gscorecambeta": "GScoreCAMBeta",
        "testcam": "TestCAM",
    }
    try:
        module = importlib.import_module("pytorch_grad_cam")
    except ImportError:
        return {}

    loaded = {}
    for cam_name, class_name in method_map.items():
        cam_class = getattr(module, class_name, None)
        if cam_class is not None:
            loaded[cam_name] = cam_class
    return loaded


_PYTORCH_GRAD_CAM_METHODS = _load_pytorch_grad_cam_methods()


# ============================================================================
# Text Handling Utilities for Multimodal CAM
# ============================================================================

def validate_text_input(text, allow_none=True):
    """Validate text input for multimodal CAM.
    
    Args:
        text: Text input (string, list of strings, or None)
        allow_none: Whether to allow None as valid input
        
    Returns:
        tuple: (is_valid, processed_text, error_message)
    """
    if text is None:
        if allow_none:
            return True, None, None
        return False, None, "Text input cannot be None"
    
    # Handle list of texts (batch processing)
    if isinstance(text, (list, tuple)):
        if len(text) == 0:
            return False, None, "Text list cannot be empty"
        # Validate each text in the list
        for i, t in enumerate(text):
            if not isinstance(t, str):
                return False, None, f"Text at index {i} is not a string: {type(t)}"
            if len(t.strip()) == 0:
                return False, None, f"Text at index {i} is empty"
        return True, text, None
    
    # Handle single string
    if isinstance(text, str):
        if len(text.strip()) == 0:
            return False, None, "Text input is empty"
        return True, text.strip(), None
    
    return False, None, f"Invalid text type: {type(text)}. Expected str, list, or None"


def tokenize_text(text, tokenizer, device='cuda'):
    """Tokenize text input for multimodal CAM.
    
    Args:
        text: Text input (string or list of strings)
        tokenizer: Tokenizer to use
        device: Device to move tokens to
        
    Returns:
        torch.Tensor: Tokenized text
    """
    if text is None:
        return None
    
    # Handle single string
    if isinstance(text, str):
        tokens = tokenizer(text).to(device)
        return tokens
    
    # Handle list of strings (batch)
    if isinstance(text, (list, tuple)):
        tokens = tokenizer(text).to(device)
        return tokens
    
    raise ValueError(f"Invalid text type for tokenization: {type(text)}")


def process_text_for_cam(text, tokenizer, device='cuda'):
    """Process text input for CAM with validation and tokenization.
    
    This is a convenience function that combines validation and tokenization.
    
    Args:
        text: Text input (string, list of strings, or None)
        tokenizer: Tokenizer to use
        device: Device to move tokens to
        
    Returns:
        tuple: (is_valid, tokens, error_message)
    """
    is_valid, processed_text, error = validate_text_input(text)
    if not is_valid:
        return False, None, error
    
    if processed_text is None:
        return True, None, None
    
    try:
        tokens = tokenize_text(processed_text, tokenizer, device)
        return True, tokens, None
    except Exception as e:
        return False, None, f"Tokenization failed: {str(e)}"


# ============================================================================
# CAM Wrapper Class
# ============================================================================

class CAMWrapper(object):
    _ORDERED_CAMS = [
        "gradcam",
        "scorecam",
        "gradcam++",
        "ablationcam",
        "xgradcam",
        "eigencam",
        "eigengradcam",
        "layercam",
        "hilacam",
        "hilacam_clipvip",
        "groupcam",
        "sscam1",
        "sscam2",
        "rawcam",
        "testhila",
        "gradientcam",
        "gscorecam",
        "vitgradcam",
        "rise",
        "gscorecambeta",
        "testcam",
        "ritsm",
        "ritsm_clipvip",
        "mhsa",
        "mhsa_clipvip",
        "AEROMIXER",
    ]
    CAM_DICT = {
        **_PYTORCH_GRAD_CAM_METHODS,
        "hilacam": hilacam_clip,
        "hilacam_clipvip": hilacam_clipvip,
        "ritsm": ritsm_clip,
        "ritsm_clipvip": ritsm_clipvip,
        "mhsa": mhsa_clip,
        "mhsa_clipvip": mhsa_clipvip,
    }
    CAM_LIST = []
    for cam_name in _ORDERED_CAMS:
        if cam_name in CAM_DICT or cam_name == "AEROMIXER":
            CAM_LIST.append(cam_name)

    def __init__(self, model, target_layers, tokenizer, cam_version, clip_version="ViT-B/32", preprocess=None, target_category=None, is_clip=True,
                 mute=False, cam_trans=None, is_transformer=False, attn_grad=True, **kwargs):
        """

        Args:
            model (model): The model to use for CAM
            target_layers (model layer): List of layers to extract features from
            tokenizer: Tokenizer for text processing
            cam_version (str): CAM method version
            clip_version (str, optional): CLIP model version. Defaults to "ViT-B/32".
            preprocess: Image preprocessing function
            target_category (int or tensor, optional): Target category for CAM
            is_clip (bool, optional): Whether model is CLIP-based. Defaults to True.
            mute (bool, optional): Suppress output. Defaults to False.
            cam_trans (function, optional): CAM transformation function
            is_transformer (bool, optional): Whether model is transformer-based. Defaults to False.
            attn_grad (bool, optional): Use attention gradients. Defaults to True.

        Raises:
            ValueError: If CAM version is not found
        """
        self.mute = mute
        self.model = model
        self.clip_version = clip_version
        self.version = self._resolve_cam_version(cam_version)
        self.target_layers = target_layers
        self.target_category = target_category
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.cam_trans = cam_trans
        self.is_transformer = is_transformer
        self.attn_grad = attn_grad
        self.is_clip = is_clip
        self.channels = None
        self.__dict__.update(kwargs)

        if self.version not in self.CAM_DICT:
            available = ", ".join(self.CAM_LIST)
            raise ValueError(f"CAM version '{self.version}' not found. Available CAMs: {available}")
        
        # Validate tokenizer for multimodal methods
        self._validate_tokenizer()
        
        # define cam
        self._load_cam()

    def _validate_tokenizer(self):
        """Validate tokenizer for multimodal CAM methods."""
        multimodal_methods = ['hilacam', 'hilacam_clipvip', 'ritsm', 'ritsm_clipvip', 'mhsa', 'mhsa_clipvip']
        
        if self.version in multimodal_methods and self.tokenizer is None:
            warnings.warn(
                f"CAM method '{self.version}' requires a tokenizer for multimodal input, "
                "but None was provided. Text input will be ignored.",
                UserWarning
            )

    def _resolve_cam_version(self, cam_version):
        version = str(cam_version).strip().lower()
        if version == "aeromixer":
            if "clip-vip" in str(self.clip_version).lower():
                return "ritsm_clipvip"
            return "ritsm"
        return version
    
    def _select_channels(self, text):
        if self.channel_dict is not None and text in self.channel_dict.keys():
            return self.channel_dict[text][:self.topk]
        else:
            return None

    # load cam
    def _load_cam(self):
        if self.version in ['hilacam', 'hilacam_clipvip', 'ritsm', 'ritsm_clipvip', 'mhsa', 'mhsa_clipvip']:
            self.cam = self.CAM_DICT[self.version]
        elif self.version == 'testhila':
            target_layer = self.model.visual.attnpool
            self.cam = self.CAM_DICT[self.version](model=self.model, target_layers=self.target_layers, use_cuda=True, clip=self.is_clip , reshape_transform=self.cam_trans, hila=True)

        elif self.version in ['scorecam', 'gscorecam']:
            batch_size = self.batch_size if hasattr(self, "batch_size") else 128
            self.cam = self.CAM_DICT[self.version](model=self.model, target_layers=self.target_layers, 
                                        use_cuda=True, is_clip=self.is_clip , reshape_transform=self.cam_trans, drop=self.drop, 
                                        mute=self.mute, channels=self.channels, topk=self.topk, batch_size=batch_size, is_transformer=self.is_transformer)
        elif self.version == 'groupcam':
            self.cam = self.CAM_DICT[self.version](self.model, self.target_layers[0], cluster_method='k_means', is_clip=self.is_clip)
        elif self.version == 'layercam':
            self.cam = self.CAM_DICT[self.version](model=self.model, target_layers=self.target_layers, 
                                        use_cuda=True, is_clip=self.is_clip , reshape_transform=self.cam_trans)
        elif self.version.startswith('sscam'):
            self.cam = self.CAM_DICT[self.version](model=self.model, is_clip=self.is_clip)
        elif self.version == 'rise':
            img_size = self.dataset_size if hasattr(self, 'dataset_size') else (384, 384)
            mask_path = f'data/rise_mask_{img_size[0]}x{img_size[1]}.npy'
            self.cam = self.CAM_DICT[self.version](model=self.model, image_size=img_size, mask_path=mask_path, batch_size=64)
        else:
            self.cam = self.CAM_DICT[self.version](model=self.model, target_layers=self.target_layers, 
                                        use_cuda=True, is_clip=self.is_clip , reshape_transform=self.cam_trans, is_transformer=self.is_transformer)

    def getCAM(self, input_img, input_text, cam_size, target_category, return_logits=False):
        """Generate CAM heatmap for image and optional text input.
        
        Args:
            input_img: Input image tensor
            input_text: Input text (tokenized or string)
            cam_size: Size of output heatmap
            target_category: Target category index
            return_logits: Whether to return logits
            
        Returns:
            numpy.ndarray: CAM heatmap
        """
        cam_input = (input_img, input_text) if self.is_clip else input_img
        self.cam.img_size = cam_size
        if self.version in ['hilacam', 'hilacam_clipvip', 'ritsm', 'ritsm_clipvip']:
            grayscale_cam = self.cam(input_img, input_text, self.model, 'cuda', cam_size=cam_size, index=target_category, return_logits=return_logits, attn_grad=self.attn_grad)
        elif self.version in ['mhsa', 'mhsa_clipvip']:
            grayscale_cam = self.cam(input_img, self.model, cam_size=cam_size, threshold=0.6)
        elif self.version == 'groupcam':
            grayscale_cam = self.cam(cam_input, class_idx=target_category)
            grayscale_cam = np.nan_to_num(grayscale_cam, nan=0.0)
        elif self.version.startswith('sscam'):
            grayscale_cam = self.cam(input_img, input_text, class_idx=target_category, 
                                     param_n=35, mean=0, sigma=2, mute=self.mute)
        elif self.version == 'layercam':
            grayscale_cam = self.cam(input_tensor=cam_input, targets=target_category)
            grayscale_cam = grayscale_cam[0, :]
        elif self.version == 'rise':
            grayscale_cam = self.cam(inputs=cam_input, targets=target_category, image_size=cam_size)
        else:
            
            grayscale_cam = self.cam(input_tensor=cam_input, targets=target_category)
            grayscale_cam = grayscale_cam[0, :]
        return grayscale_cam
    
    def __call__(self, inputs, label, heatmap_size):
        """Generate CAM heatmap.
        
        Args:
            inputs: Image tensor or tuple of (image, text)
            label: Target category
            heatmap_size: Size of output heatmap
            
        Returns:
            numpy.ndarray: CAM heatmap
        """

        if isinstance(inputs, tuple):
            img, text = inputs[0], inputs[1]
        else:
            img = inputs
            text = None

        if self.preprocess is not None:
            img = self.preprocess(img)
        
        # Tokenize text with validation
        text_token = None
        if text is not None:
            is_valid, text_token, error = process_text_for_cam(
                text, self.tokenizer, device='cuda'
            )
            if not is_valid:
                warnings.warn(f"Text processing failed: {error}. Proceeding without text.", UserWarning)
                text_token = None
        
        if len(img.shape) < 4:
            img = img.unsqueeze(0)
        if not img.is_cuda:
            img = img.cuda()
        if hasattr(self, "channel_dict"):
            # self.cam.channels = self.channel_dict[text]
            self.cam.channels = self._select_channels(text)

        return self.getCAM(img, text_token, heatmap_size, label)
    
    def getLogits(self, img, text):
        """Get model logits for image-text pairs.
        
        Args:
            img: Image tensor
            text: Text input (string or list of strings)
            
        Returns:
            tuple: (image_per_text_logits, text_per_image_logits)
        """
        with torch.no_grad():
            if self.preprocess is not None:
                img = self.preprocess(img)
            
            # Tokenize text with validation
            text_token = None
            if text is not None:
                is_valid, text_token, error = process_text_for_cam(
                    text, self.tokenizer, device='cuda'
                )
                if not is_valid:
                    warnings.warn(f"Text processing failed: {error}. Using empty text.", UserWarning)
                    text_token = self.tokenizer("").cuda()
            
            img_per_text, text_per_img = self.model(img.unsqueeze(0).cuda(), text_token)
        return img_per_text, text_per_img


def get_heatmap_from_mask(
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # heatmap = np.float32(heatmap) / 255

    # if np.max(img) > 1:
    #     raise Exception("The input image should np.float32 in the range [0, 1]")

    # cam = heatmap + img
    # cam = cam / np.max(cam)
    return heatmap
