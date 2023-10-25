from face_services.processors.utilities import resolve_relative_path

FACE_SWAPPER_MODELS =\
{
	'inswapper_128':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128_fp16.onnx',
		'path': resolve_relative_path('../.assets/face_swapper/inswapper_128_fp16.onnx')
	},
	'inswapper_128_fp16':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx',
		'path': resolve_relative_path('../.assets/face_swapper/inswapper_128.onnx')
	}
}


FACE_ENHANCER_MODELS =\
{
	'gfpgan_1.4':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/GFPGANv1.4.onnx',
		'path': resolve_relative_path('../.assets/face_enhancer/GFPGANv1.4.onnx')
	},
	'gfpgan_1.2':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/GFPGANv1.2.onnx',
		'path': resolve_relative_path('../.assets/face_enhancer/GFPGANv1.2.onnx')
	},
	'gfpgan_1.3':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/GFPGANv1.3.onnx',
		'path': resolve_relative_path('../.assets/face_enhancer/GFPGANv1.3.onnx')
	},
	'gpen_bfr_512':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/GPEN-BFR-512.onnx',
		'path': resolve_relative_path('../.assets/face_enhancer/GPEN-BFR-512.onnx')
	},
	'codeformer':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/codeformer.onnx',
		'path': resolve_relative_path('../.assets/face_enhancer/codeformer.onnx')
	}
}
