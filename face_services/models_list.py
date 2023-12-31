from face_services.processors.utilities import resolve_relative_path

FACE_ANALYZER_MODELS =\
{
	'recognition':
	{ 
		'face_recognition_arcface_inswapper':
		{
			'url': 'https://huggingface.co/bluefoxcreation/insightface-retinaface-arcface-model/resolve/main/w600k_r50.onnx',
			'path': resolve_relative_path('../models/face_analyzer/w600k_r50.onnx'),
		},
		'face_recognition_arcface_simswap':
		{
			'url': 'https://github.com/harisreedhar/Face-Swappers-ONNX/releases/download/simswap/simswap_arcface_backbone.onnx',
			'path': resolve_relative_path('../models/face_analyzer/simswap_arcface_backbone.onnx'),
		},
    },
	'gender_age_estimation':
	{
		'gender_age_model':
		{
			'url': 'https://huggingface.co/facefusion/buffalo_l/resolve/main/genderage.onnx',
			'path': resolve_relative_path('../models/face_analyzer/genderage.onnx'),
		},
	},
	'detection':
	{
		'face_detection_yunet':
		{
			'url': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
			'path': resolve_relative_path('../models/face_analyzer/face_detection_yunet_2023mar.onnx'),
		},
	},
}

FACE_SWAPPER_MODELS =\
{
	'inswapper_128':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128_fp16.onnx',
		'path': resolve_relative_path('../models/face_swapper/inswapper_128_fp16.onnx'),
        'size': (128, 128),
        'template': 'arcface_v2',
	},
	'inswapper_128_fp16':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx',
		'path': resolve_relative_path('../models/face_swapper/inswapper_128.onnx'),
        'size': (128, 128),
        'template': 'arcface_v2',
	},
}

FACE_ENHANCER_MODELS =\
{
	'gfpgan_1.4':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/GFPGANv1.4.onnx',
		'path': resolve_relative_path('../models/face_enhancer/GFPGANv1.4.onnx')
	},
	'gfpgan_1.2':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/GFPGANv1.2.onnx',
		'path': resolve_relative_path('../models/face_enhancer/GFPGANv1.2.onnx')
	},
	'gfpgan_1.3':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/GFPGANv1.3.onnx',
		'path': resolve_relative_path('../models/face_enhancer/GFPGANv1.3.onnx')
	},
	'gpen_bfr_512':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/GPEN-BFR-512.onnx',
		'path': resolve_relative_path('../models/face_enhancer/GPEN-BFR-512.onnx')
	},
	'codeformer':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/codeformer.onnx',
		'path': resolve_relative_path('../models/face_enhancer/codeformer.onnx')
	}
}

VISUAL_DUBBER_MODELS =\
{
	'wav2lip':
	{
		'url': '',
		'path': resolve_relative_path('../models/visual_dubber/wav2lip.pth'),
	},
	'wav2lip_gan':
	{
		'url': '',
		'path': resolve_relative_path('../models/visual_dubber/wav2lip_gan.pth'),
	},
}