from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.DATA = CN()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = ""

# Input mode - now only supports "image" (video support removed)
_C.DATA.INPUT_TYPE = "image"

# Image mode flag - set to True for image-only multimodal (no temporal processing)
_C.DATA.IMAGE_MODE = True

# Relative frame/image directory under PATH_TO_DATA_DIR.
_C.DATA.FRAME_DIR = ""

# Annotation backend for image datasets: auto|txt|yolo|coco|voc|custom_json
_C.DATA.ANNOTATION_FORMAT = "auto"

# The number of frames of the input clip.
# For image-only mode, set to 1
_C.DATA.NUM_FRAMES = 1

# The video sampling rate of the input clip.
# For image-only mode, set to 1
_C.DATA.SAMPLING_RATE = 1

# List of input frame channel dimensions.
_C.DATA.INPUT_CHANNEL_NUM = [3]

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

_C.DATA.TRAIN_MIN_SCALES = [256, 320]  # list
_C.DATA.TRAIN_MAX_SCALE = 1333  # int
_C.DATA.TEST_MIN_SCALES = [256]  # list
_C.DATA.TEST_MAX_SCALE = 1333  # int

_C.DATA.FIX_SIZE = []

# If True, revert the default input channel (RBG <-> BGR).
_C.DATA.REVERSE_INPUT_CHANNEL = False

# Use ['images'] for image pipeline only (video support removed).
_C.DATA.DATASETS = ["images"]

_C.DATA.OPEN_VOCABULARY = False
_C.DATA.MULTIMODAL = False
# Optional generic vocab files for open-vocabulary text prompts.
_C.DATA.VOCAB_FILE = ""
_C.DATA.VOCAB_OPEN_FILE = ""

_C.DATA.REFINE_VOCAB = False

# -----------------------------------------------------------------------------
# Multimodal Text Configuration
# -----------------------------------------------------------------------------
_C.DATA.TEXT = CN()

# Text input type: "single" (one text per image) or "multiple" (multiple texts per image)
_C.DATA.TEXT.INPUT_TYPE = "single"

# Maximum text length (in tokens)
_C.DATA.TEXT.MAX_LENGTH = 77

# Text prompt template for open vocabulary detection
# Use {class_name} as placeholder for class name
_C.DATA.TEXT.PROMPT_TEMPLATE = "a photo of {}"

# Legacy field name retained for compatibility; active runtime uses LiteText.
_C.DATA.TEXT.USE_CLIP_ENCODER = True

# Text augmentation options
_C.DATA.TEXT.AUGMENT = CN()
_C.DATA.TEXT.AUGMENT.ENABLE = False
_C.DATA.TEXT.AUGMENT.NUM_VARIATIONS = 5

# -----------------------------------------------------------------------------
# Image Dataset preprocess options
# -----------------------------------------------------------------------------
_C.IMAGES = CN()
_C.IMAGES.BGR = False
_C.IMAGES.TRAIN_USE_COLOR_AUGMENTATION = False
_C.IMAGES.TRAIN_COLOR_BRIGHTNESS = 0.25
_C.IMAGES.TRAIN_COLOR_CONTRAST = 0.25
_C.IMAGES.TRAIN_COLOR_SATURATION = 0.25
_C.IMAGES.TRAIN_PCA_JITTER_STD = 0.1
_C.IMAGES.TRAIN_PCA_JITTER_ONLY = True
_C.IMAGES.TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229]
_C.IMAGES.TRAIN_PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]
_C.IMAGES.TEST_FORCE_FLIP = False
_C.IMAGES.VOCAB_FILE = ""
_C.IMAGES.VOCAB_OPEN_FILE = ""

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of dataset loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 32
# Long-tail mitigation for detection datasets.
_C.DATALOADER.BALANCED_SAMPLING = False
_C.DATALOADER.BALANCED_SAMPLING_POWER = 0.75
_C.DATALOADER.BALANCED_SAMPLING_MIN_COUNT = 1.0
_C.DATALOADER.BALANCED_SAMPLING_EMPTY_WEIGHT = 0.25
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = False


_C.MODEL = CN()

_C.MODEL.WEIGHT = ""
_C.MODEL.DET = "AeroLiteDetector"
_C.MODEL.MULTI_LABEL_ACTION = False
_C.MODEL.PRE_EXTRACT_FEAT = False
_C.MODEL.USE_ROI_FEAT = False

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
# Supported public backbones: AeroLite-Det-T, AeroLite-Det-S, AeroLite-Det-B
_C.MODEL.BACKBONE.CONV_BODY = "AeroLite-Det-S"
_C.MODEL.BACKBONE.PATHWAYS = 1
_C.MODEL.BACKBONE.FROZEN_BN = False
_C.MODEL.BACKBONE.FREEZE_PRETRAINED_VISUAL = True
_C.MODEL.BACKBONE.FREEZE_PRETRAINED_TEXT = True
_C.MODEL.BACKBONE.RESIDUAL_LATERAL = False

# For alphaction backbones
_C.MODEL.BACKBONE.BN_MOMENTUM = 0.1
_C.MODEL.BACKBONE.BN_EPSILON = 1e-05

# Kaiming:
# We may use 0 to initialize the residual branch of a residual block,
# so the inital state of the block is exactly identiy. This helps optimizaiton.
_C.MODEL.BACKBONE.BN_INIT_GAMMA = 0.0

_C.MODEL.BACKBONE.I3D = CN()
_C.MODEL.BACKBONE.I3D.CONV3_NONLOCAL = True
_C.MODEL.BACKBONE.I3D.CONV4_NONLOCAL = True
_C.MODEL.BACKBONE.I3D.CONV3_GROUP_NL = False

# ---------------------------------------------------------------------------- #
# Slowfast options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE.SLOWFAST = CN()
_C.MODEL.BACKBONE.SLOWFAST.BETA = 1.0 / 8
_C.MODEL.BACKBONE.SLOWFAST.LATERAL = "tconv"
_C.MODEL.BACKBONE.SLOWFAST.SLOW = CN()
_C.MODEL.BACKBONE.SLOWFAST.SLOW.ACTIVE = True
_C.MODEL.BACKBONE.SLOWFAST.SLOW.CONV3_NONLOCAL = True
_C.MODEL.BACKBONE.SLOWFAST.SLOW.CONV4_NONLOCAL = True
_C.MODEL.BACKBONE.SLOWFAST.SLOW.CONV3_GROUP_NL = False
_C.MODEL.BACKBONE.SLOWFAST.FAST = CN()
_C.MODEL.BACKBONE.SLOWFAST.FAST.ACTIVE = True
_C.MODEL.BACKBONE.SLOWFAST.FAST.CONV3_NONLOCAL = False
_C.MODEL.BACKBONE.SLOWFAST.FAST.CONV4_NONLOCAL = False
_C.MODEL.BACKBONE.SLOWFAST.FAST.CONV3_GROUP_NL = False

# ---------------------------------------------------------------------------- #
# Nonlocal options
# ---------------------------------------------------------------------------- #
_C.MODEL.NONLOCAL = CN()
_C.MODEL.NONLOCAL.CONV_INIT_STD = 0.01
_C.MODEL.NONLOCAL.USE_ZERO_INIT_CONV = False
_C.MODEL.NONLOCAL.NO_BIAS = False
_C.MODEL.NONLOCAL.USE_MAXPOOL = True
_C.MODEL.NONLOCAL.USE_SOFTMAX = True
_C.MODEL.NONLOCAL.USE_SCALE = True

_C.MODEL.NONLOCAL.USE_BN = True
_C.MODEL.NONLOCAL.FROZEN_BN = False

_C.MODEL.NONLOCAL.BN_MOMENTUM = 0.1
_C.MODEL.NONLOCAL.BN_EPSILON = 1e-05
_C.MODEL.NONLOCAL.BN_INIT_GAMMA = 0.0


# For PySlowFast backbones
_C.RESNET = CN()

# Transformation function.
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_BN = False

# Number of weight layers.
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# Size of stride on different res stages.
_C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]

# Size of dilation on different res stages.
_C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]

# Whether use modulated DCN on Res2, Res3, Res4, Res5 or not
_C.RESNET.DEFORM_ON_PER_STAGE = [False, False, False, False]

_C.NONLOCAL = CN()
# Index of each stage and block to add nonlocal layers.
_C.NONLOCAL.LOCATION = [[[]], [[]], [[]], [[]]]

# Number of group for nonlocal for each stage.
_C.NONLOCAL.GROUP = [[1], [1], [1], [1]]

# Instatiation to use for non-local layer.
_C.NONLOCAL.INSTANTIATION = "dot_product"


# Size of pooling layers used in Non-Local.
_C.NONLOCAL.POOL = [
    # Res2
    [[1, 2, 2], [1, 2, 2]],
    # Res3
    [[1, 2, 2], [1, 2, 2]],
    # Res4
    [[1, 2, 2], [1, 2, 2]],
    # Res5
    [[1, 2, 2], [1, 2, 2]],
]

_C.SLOWFAST = CN()

# Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# the Slow and Fast pathways.
_C.SLOWFAST.BETA_INV = 8

# Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# Fast pathways.
_C.SLOWFAST.ALPHA = 8

# Ratio of channel dimensions between the Slow and Fast pathways.
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# Kernel dimension used for fusing information from Fast pathway to Slow
# pathway.
_C.SLOWFAST.FUSION_KERNEL_SZ = 5


_C.ViT = CN()
_C.ViT.TUBELET_SIZE = 2
_C.ViT.PATCH_SIZE = 16
_C.ViT.IN_CHANS = 3
_C.ViT.EMBED_DIM = 768
_C.ViT.PRETRAIN_IMG_SIZE = 224
_C.ViT.USE_LEARNABLE_POS_EMB = False
_C.ViT.DROP_RATE = 0.0
_C.ViT.ATTN_DROP_RATE = 0.0
_C.ViT.DROP_PATH_RATE = 0.2  #
_C.ViT.DEPTH = 12
_C.ViT.NUM_HEADS = 12
_C.ViT.MLP_RATIO = 4
_C.ViT.QKV_BIAS = True
_C.ViT.QK_SCALE = None
_C.ViT.INIT_VALUES = 0.0
_C.ViT.USE_CHECKPOINT = True
_C.ViT.LAYER_DECAY = 0.75
_C.ViT.WEIGHT_DECAY = 0.05
_C.ViT.NO_WEIGHT_DECAY = ["pos_embed"]


# STMixer
_C.MODEL.STM = CN()
_C.MODEL.STM.NUM_QUERIES = 100
_C.MODEL.STM.QUERY_INIT_MODE = "learnable_anchors"  # learnable_anchors | full_image
_C.MODEL.STM.QUERY_INIT_BASE_SCALE = 0.20
_C.MODEL.STM.QUERY_INIT_MIN_SCALE = 0.02
_C.MODEL.STM.QUERY_INIT_MAX_SCALE = 0.60
_C.MODEL.STM.QUERY_INIT_CENTER_OFFSET = 0.25
_C.MODEL.STM.QUERY_INIT_LOG_WH_CLAMP = 2.0
_C.MODEL.STM.QUERY_INIT_SMALL_OBJECT_BIAS = True
_C.MODEL.STM.QUERY_INIT_SMALL_OBJECT_SCALE = 0.70
_C.MODEL.STM.ATTN_TELEMETRY = False
_C.MODEL.STM.ATTN_TELEMETRY_STAGEWISE = True
_C.MODEL.STM.ATTN_TELEMETRY_COMPARE_NOMASK = False
_C.MODEL.STM.IOF_TAU_MODE = "learned"  # learned | zero | fixed | clamp
_C.MODEL.STM.IOF_TAU_FIXED = 0.0
_C.MODEL.STM.IOF_TAU_CLAMP_MIN = 0.0
_C.MODEL.STM.IOF_TAU_CLAMP_MAX = 4.0
_C.MODEL.STM.HIDDEN_DIM = 256
_C.MODEL.STM.NUM_STAGES = 6
_C.MODEL.STM.ACTION_CLASSES = 80
_C.MODEL.STM.OBJECT_CLASSES = 1
_C.MODEL.STM.NUM_HEADS = 8
_C.MODEL.STM.DROPOUT = 0.0
_C.MODEL.STM.DIM_FEEDFORWARD = 2048
_C.MODEL.STM.NUM_FCS = 2
_C.MODEL.STM.ACTIVATION = "ReLU"
_C.MODEL.STM.SPATIAL_POINTS = 32
_C.MODEL.STM.TEMPORAL_POINTS = 4
_C.MODEL.STM.OUT_MULTIPLIER = 4
_C.MODEL.STM.N_GROUPS = 4
_C.MODEL.STM.NUM_CLS = 1
_C.MODEL.STM.NUM_ACT = 1
_C.MODEL.STM.NUM_REG = 1
_C.MODEL.STM.NUM_SEV = 1
_C.MODEL.STM.OBJECT_WEIGHT = 2.0
_C.MODEL.STM.ACTION_WEIGHT = 24.0
_C.MODEL.STM.GIOU_WEIGHT = 2.0
_C.MODEL.STM.L1_WEIGHT = 2.0
_C.MODEL.STM.SEVERITY_WEIGHT = 1.0
_C.MODEL.STM.BACKGROUND_WEIGHT = 0.1
_C.MODEL.STM.FOCAL_WEIGHT = 0.0
_C.MODEL.STM.INTERMEDIATE_SUPERVISION = True
_C.MODEL.STM.PREDICT_SEVERITY = False
_C.MODEL.STM.PERSON_THRESHOLD = 0.6
_C.MODEL.STM.SCORE_THRESHOLD = 0.0
_C.MODEL.STM.USE_CLS_FEAT = False
_C.MODEL.STM.COND_CLS = False
_C.MODEL.STM.COND_MODALITY = "visual"
_C.MODEL.STM.FUSE_CLS = False
_C.MODEL.STM.FUSE_METHOD = ""
_C.MODEL.STM.FUSE_FACTOR = -1.0
_C.MODEL.STM.CHN_WEIGHT = 1.0
_C.MODEL.STM.TEXT_SCORE_FUSION = False
_C.MODEL.STM.TEXT_SCORE_FUSION_ALPHA = 0.35
_C.MODEL.STM.TEXT_LOGIT_SCALE = 12.0
_C.MODEL.STM.TEXT_QUERY_COND = False
_C.MODEL.STM.TEXT_QUERY_COND_SCALE = 0.20
_C.MODEL.STM.SCALE_TEXT_ROUTING = False
_C.MODEL.STM.SCALE_TEXT_ROUTING_HIDDEN = 128
_C.MODEL.STM.SCALE_TEXT_ROUTING_GAIN = 0.75
_C.MODEL.STM.SCALE_TEXT_ROUTING_TEMP = 1.0
_C.MODEL.STM.PROTOTYPE_MEMORY = False
_C.MODEL.STM.PROTOTYPE_MEMORY_MOMENTUM = 0.90
_C.MODEL.STM.PROTOTYPE_MEMORY_BLEND = 0.35
_C.MODEL.STM.PROTOTYPE_MEMORY_CONTEXT_BLEND = 0.25
_C.MODEL.STM.CLASS_PARTITION_QUERIES = False
_C.MODEL.STM.CLASS_PARTITION_QUERY_RATIO = 0.20
_C.MODEL.STM.CLASS_PARTITION_TOPK = 3
_C.MODEL.STM.CLASS_PARTITION_SCALE = 0.30
_C.MODEL.STM.PROMPT_ADAPTIVE_QUERIES = False
_C.MODEL.STM.PROMPT_ADAPTIVE_QUERY_RATIO = 0.30
_C.MODEL.STM.PROMPT_ADAPTIVE_QUERY_SCALE = 0.35
_C.MODEL.STM.PROMPT_ADAPTIVE_MIN_FRACTION = 0.50
_C.MODEL.STM.TILE_GLOBAL_CONTEXT = False
_C.MODEL.STM.TILE_GLOBAL_CONTEXT_HIDDEN = 128
_C.MODEL.STM.TILE_GLOBAL_CONTEXT_BLEND = 0.20
_C.MODEL.STM.TILE_GLOBAL_CONTEXT_QUERY_SCALE = 0.20
_C.MODEL.STM.TILE_GLOBAL_CONTEXT_LEVEL_GAIN = 0.35
_C.MODEL.STM.CROSS_TILE_CONSISTENCY = False
_C.MODEL.STM.CROSS_TILE_CONSISTENCY_WEIGHT = 0.20
_C.MODEL.STM.CROSS_TILE_CONSISTENCY_TOPK = 12
_C.MODEL.STM.CROSS_TILE_CONSISTENCY_MIN_SCORE = 0.10
_C.MODEL.STM.CROSS_TILE_CONSISTENCY_IOU = 0.20
_C.MODEL.STM.CROSS_TILE_CONSISTENCY_BOX_WEIGHT = 0.25
_C.MODEL.STM.CROSS_TILE_CONSISTENCY_LOGIT_WEIGHT = 1.00
_C.MODEL.STM.CROSS_TILE_CONSISTENCY_EMBED_WEIGHT = 0.25

_C.MODEL.STM.PRETRAIN_ACTION = False
_C.MODEL.STM.DeST = False
_C.MODEL.STM.FS_GAMMA = 0.0
# Optional class-weighted CE for long-tail robustness.
_C.MODEL.STM.CLASS_WEIGHTING = "none"  # none | inverse_freq | sqrt_inverse_freq
_C.MODEL.STM.CLASS_WEIGHT_POWER = 1.0
_C.MODEL.STM.CLASS_WEIGHT_MIN = 0.25
_C.MODEL.STM.CLASS_WEIGHT_MAX = 4.0
_C.MODEL.STM.CLASS_WEIGHT_EMA = 0.9

_C.MODEL.TEXT_ENCODER = "LITE_TEXT"
_C.MODEL.USE_PRIOR_MAP = False
_C.MODEL.PRIOR_BOXES_INIT = ""

_C.MODEL.FOCAL_LOSS = CN()
_C.MODEL.FOCAL_LOSS.ALPHA = 0.25
_C.MODEL.FOCAL_LOSS.GAMMA = 2.0


_C.MODEL.LITE_TEXT = CN()
_C.MODEL.LITE_TEXT.EMBED_DIM = 256
_C.MODEL.LITE_TEXT.MAX_TOKENS = 12
_C.MODEL.LITE_TEXT.MAX_VARIANTS = 4
_C.MODEL.LITE_TEXT.CONTEXT_TOKENS = 4
_C.MODEL.LITE_TEXT.VOCAB_SIZE = 4096
_C.MODEL.LITE_TEXT.NUM_LAYERS = 2
_C.MODEL.LITE_TEXT.NUM_HEADS = 4
_C.MODEL.LITE_TEXT.FFN_DIM = 512
_C.MODEL.LITE_TEXT.DROPOUT = 0.0

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.ITER_PER_EPOCH = -1
_C.SOLVER.MAX_EPOCH = 12

_C.SOLVER.BASE_LR = 0.0002
_C.SOLVER.BETAS = (0.9, 0.999)
_C.SOLVER.WEIGHT_DECAY = 0.0001
# Use for bn
_C.SOLVER.WEIGHT_DECAY_BN = 0.0
_C.SOLVER.SCHEDULER = "warmup_multi_step"
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (5, 8)

_C.SOLVER.WARMUP_ON = True
_C.SOLVER.WARMUP_FACTOR = 0.1
_C.SOLVER.WARMUP_EPOCH = 2
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.CHECKPOINT_TOPK = 1
_C.SOLVER.EVAL_PERIOD = 1
_C.SOLVER.EVAL_AFTER = 2

# Number of training samples per batch for image datasets.
_C.SOLVER.IMAGES_PER_BATCH = 16
_C.SOLVER.OPTIMIZING_METHOD = "adamw"

# ---------------------------------------------------------------------------- #
# Specific test options
# ------------------------------------------------------FCLIP---------------------- #
_C.TEST = CN()
# Number of test samples per batch for image datasets.
_C.TEST.IMAGES_PER_BATCH = 16
_C.TEST.EVAL_OPEN = False
_C.TEST.METRIC = "frame_ap"
_C.TEST.SMALL_OPEN_WORLD = False
_C.TEST.INDEPENDENT_EVAL = False
_C.TEST.IOU_THRESH = 0.5
_C.TEST.REPORT_AP5095 = False
_C.TEST.AP5095_MIN = 0.5
_C.TEST.AP5095_MAX = 0.95
_C.TEST.AP5095_STEP = 0.05
_C.TEST.PRIOR_BOX_TEST = False
_C.TEST.TILE_STITCH_EVAL = False
_C.TEST.TILE_STITCH_NMS_IOU = 0.5
_C.TEST.TILE_STITCH_GT_DEDUP_IOU = 0.9

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
