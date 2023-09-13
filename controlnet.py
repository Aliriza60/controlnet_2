

# python controlnet_2_cpu.py --controlnet_type canny mlsd --image_paths input_image_vermeer.png person.jpeg --base_model_path runwayml/stable-diffusion-v1-5 --controlnet_model_paths control_v11p_sd15_canny control_v11p_sd15_mlsd --prompt "disco dancer with colorful lights" --net_scale 1.0 0.8 

# python controlnet_2_cpu.py --controlnet_type canny mlsd --image_paths input_image_vermeer.png person.jpeg --base_model_path v1-5-pruned.safetensors --controlnet_model_paths control_v11p_sd15_canny control_v11p_sd15_mlsd --prompt "disco dancer with colorful lights" --net_scale 1.0 0.8 --single_base_model_file

# python controlnet_2_cpu.py --controlnet_type canny --image_paths input_image_vermeer.png --base_model_path v1-5-pruned.safetensors --controlnet_model_paths control_v11p_sd15_canny --prompt "disco dancer with colorful lights" --single_base_model_file


import argparse
import os
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from torch import autocast
import numpy as np
from diffusers.utils import load_image
from controlnet_aux import MLSDdetector
from controlnet_aux import OpenposeDetector
# pip install controlnet_aux
from controlnet_aux import HEDdetector
from controlnet_aux import PidiNetDetector, HEDdetector
from transformers import pipeline
from controlnet_aux import NormalBaeDetector
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from controlnet_aux import ContentShuffleDetector
from controlnet_aux import LineartDetector
from controlnet_aux import LineartAnimeDetector

from diffusers import LMSDiscreteScheduler
from diffusers import DDIMScheduler
from diffusers import DPMSolverMultistepScheduler
from diffusers import EulerDiscreteScheduler
from diffusers import PNDMScheduler
from diffusers import DDPMScheduler
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import UniPCMultistepScheduler

def list_of_strings(arg):
	 return arg.split(',')

parser = argparse.ArgumentParser()


parser.add_argument(
	"--workerTaskId",
	type=int,
	default=0,
	help="Worker Task Id"
)

parser.add_argument(
	"--base_model_path",
	type=str,
	default="",
	help="Model Path",
)

parser.add_argument('-cmp',
	"--controlnet_model_paths",
	nargs='+',
	type=str,
	default=[],
	help='<Required> Set flag',
	required=True)


parser.add_argument('-i',
	'--image_paths',
	nargs='+',
	type=str,
	default=[],
	help='<Required> Set flag',
	required=True)

parser.add_argument(
	"--output",
	type=str,
	default="",
	help="Output Folder",
)

parser.add_argument(
	"--prompt",
	type=str,
	default="",
	help="Positive Prompt",
)

parser.add_argument(
	"--negative_prompt",
	type=str,
	default="",
	help="Negative Prompt",
)

parser.add_argument(
	"--width",
	type=int,
	default=512,
	help="width"
)

parser.add_argument(
	"--height",
	type=int,
	default=512,
	help="height"
)

parser.add_argument(
	"--samples",
	type=int,
	default=1,
	help="Samples"
)
parser.add_argument(
	"--inference_steps",
	type=int,
	default=50,
	help="Inference_Steps"
)
parser.add_argument(
	"--guidance_scale",
	type=float,
	default=7.5,
	help="Guidance Scale"
)

parser.add_argument(
	"--scheduler",
	type=str,
	default="UniPCMultistepScheduler",
	help="",
)

parser.add_argument(
	"--seed",
	type=int,
	default=3466454,
	help="Seed"
)

parser.add_argument(
	"--single_base_model_file",
	default=False,
	action = "store_true",
	help="check single file or not"
)


parser.add_argument(
	"--pre_process",
	default=False,
	action = "store_true",
	help="check pre process or not"
)

parser.add_argument(
	"--multi_controlnet_or_not",
	default=True,
	action = "store_true",
	help="check single controlnet or not"
)

parser.add_argument("-c",
	"--controlnet_type",
	nargs='+',
	type=str,
	default=[],
	help='<Required> Set flag',
	required=True)

parser.add_argument("-n",
	"--net_scale",
	nargs='+',
	type=str,
	default=[],
	help='<Required> Set flag',
	required=False)

ap = parser.parse_args()

# Show Info Part
info_text = f"Script Version : 1.1 \
				Arguments : \
				--workerTaskId: {ap.workerTaskId} \
				--base_model_path: {ap.base_model_path} \
				--image_paths: {ap.image_paths} \
				--controlnet_model_paths: {ap.controlnet_model_paths} \
				--output:{ap.output} \
				--prompt: {ap.prompt} \
				--negative_prompt: {ap.negative_prompt} \
				--width: {ap.width} \
				--height: {ap.height} \
				--samples: {ap.samples} \
				--inference_steps: {ap.inference_steps} \
				--guidance_scale: {ap.guidance_scale} \
				--scheduler: {ap.scheduler} \
				--seed: {ap.seed} \
				--single_base_model_file: {ap.single_base_model_file} \
				--controlnet_type: {ap.controlnet_type} \
				--pre_process: {ap.pre_process} \
                                --multi_controlnet_or_not: {ap.multi_controlnet_or_not} \
				--net_scale: {ap.net_scale}"

print(info_text)


x_1 = ap.image_paths[0]

image_paths = x_1.split(";")

x_2 = ap.controlnet_model_paths[0]

controlnet_model_paths = x_2.split(";")

x_3 = ap.controlnet_type[0]

controlnet_type = x_3.split(";")

if multi_controlnet_or_not:
	x_4 = ap.net_scale[0]
	x_5 = x_4.split(";")
	net_scale = []
	for ns in (x_5):
		net_scale.append(np.float64(ns))

def canny(image_path):

	image = load_image(image_path)

	image = np.array(image)

	low_threshold = 100

	high_threshold = 200

	image = cv2.Canny(image, low_threshold, high_threshold)

	image = image[:, :, None]

	image = np.concatenate([image, image, image], axis=2)

	control_image = Image.fromarray(image)

	return control_image


def mlsd(image_path):

	mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')

	image = load_image(image_path)

	control_image = mlsd(image)

	return control_image

def open_pose(image_path):

	openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

	image = load_image(image_path)

	control_image = openpose(image)

	return control_image


def scribble(image_path):

	hed = HEDdetector.from_pretrained('lllyasviel/Annotators')

	image = load_image(image_path)

	control_image = hed(image, scribble=True)

	return control_image


def depth(image_path):

	depth_estimator = pipeline('depth-estimation')

	image = load_image(image_path)

	image = depth_estimator(image)['depth']

	image = np.array(image)

	image = image[:, :, None]

	image = np.concatenate([image, image, image], axis=2)

	control_image = Image.fromarray(image)

	return control_image

def normalbae(image_path):

	image = load_image(image_path)

	processor = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")

	control_image = processor(image)

	return control_image


def seg(image_path):

	ada_palette = np.asarray([
	  [0, 0, 0],
	  [120, 120, 120],
	  [180, 120, 120],
	  [6, 230, 230],
	  [80, 50, 50],
	  [4, 200, 3],
	  [120, 120, 80],
	  [140, 140, 140],
	  [204, 5, 255],
	  [230, 230, 230],
	  [4, 250, 7],
	  [224, 5, 255],
	  [235, 255, 7],
	  [150, 5, 61],
	  [120, 120, 70],
	  [8, 255, 51],
	  [255, 6, 82],
	  [143, 255, 140],
	  [204, 255, 4],
	  [255, 51, 7],
	  [204, 70, 3],
	  [0, 102, 200],
	  [61, 230, 250],
	  [255, 6, 51],
	  [11, 102, 255],
	  [255, 7, 71],
	  [255, 9, 224],
	  [9, 7, 230],
	  [220, 220, 220],
	  [255, 9, 92],
	  [112, 9, 255],
	  [8, 255, 214],
	  [7, 255, 224],
	  [255, 184, 6],
	  [10, 255, 71],
	  [255, 41, 10],
	  [7, 255, 255],
	  [224, 255, 8],
	  [102, 8, 255],
	  [255, 61, 6],
	  [255, 194, 7],
	  [255, 122, 8],
	  [0, 255, 20],
	  [255, 8, 41],
	  [255, 5, 153],
	  [6, 51, 255],
	  [235, 12, 255],
	  [160, 150, 20],
	  [0, 163, 255],
	  [140, 140, 140],
	  [250, 10, 15],
	  [20, 255, 0],
	  [31, 255, 0],
	  [255, 31, 0],
	  [255, 224, 0],
	  [153, 255, 0],
	  [0, 0, 255],
	  [255, 71, 0],
	  [0, 235, 255],
	  [0, 173, 255],
	  [31, 0, 255],
	  [11, 200, 200],
	  [255, 82, 0],
	  [0, 255, 245],
	  [0, 61, 255],
	  [0, 255, 112],
	  [0, 255, 133],
	  [255, 0, 0],
	  [255, 163, 0],
	  [255, 102, 0],
	  [194, 255, 0],
	  [0, 143, 255],
	  [51, 255, 0],
	  [0, 82, 255],
	  [0, 255, 41],
	  [0, 255, 173],
	  [10, 0, 255],
	  [173, 255, 0],
	  [0, 255, 153],
	  [255, 92, 0],
	  [255, 0, 255],
	  [255, 0, 245],
	  [255, 0, 102],
	  [255, 173, 0],
	  [255, 0, 20],
	  [255, 184, 184],
	  [0, 31, 255],
	  [0, 255, 61],
	  [0, 71, 255],
	  [255, 0, 204],
	  [0, 255, 194],
	  [0, 255, 82],
	  [0, 10, 255],
	  [0, 112, 255],
	  [51, 0, 255],
	  [0, 194, 255],
	  [0, 122, 255],
	  [0, 255, 163],
	  [255, 153, 0],
	  [0, 255, 10],
	  [255, 112, 0],
	  [143, 255, 0],
	  [82, 0, 255],
	  [163, 255, 0],
	  [255, 235, 0],
	  [8, 184, 170],
	  [133, 0, 255],
	  [0, 255, 92],
	  [184, 0, 255],
	  [255, 0, 31],
	  [0, 184, 255],
	  [0, 214, 255],
	  [255, 0, 112],
	  [92, 255, 0],
	  [0, 224, 255],
	  [112, 224, 255],
	  [70, 184, 160],
	  [163, 0, 255],
	  [153, 0, 255],
	  [71, 255, 0],
	  [255, 0, 163],
	  [255, 204, 0],
	  [255, 0, 143],
	  [0, 255, 235],
	  [133, 255, 0],
	  [255, 0, 235],
	  [245, 0, 255],
	  [255, 0, 122],
	  [255, 245, 0],
	  [10, 190, 212],
	  [214, 255, 0],
	  [0, 204, 255],
	  [20, 0, 255],
	  [255, 255, 0],
	  [0, 153, 255],
	  [0, 41, 255],
	  [0, 255, 204],
	  [41, 0, 255],
	  [41, 255, 0],
	  [173, 0, 255],
	  [0, 245, 255],
	  [71, 0, 255],
	  [122, 0, 255],
	  [0, 255, 184],
	  [0, 92, 255],
	  [184, 255, 0],
	  [0, 133, 255],
	  [255, 214, 0],
	  [25, 194, 194],
	  [102, 255, 0],
	  [92, 0, 255],
  ])

	image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")

	image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")


	image = load_image(image_path)

	pixel_values = image_processor(image, return_tensors="pt").pixel_values

	with torch.no_grad():
		outputs = image_segmentor(pixel_values)

	seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
	color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3

	for label, color in enumerate(ada_palette):
		color_seg[seg == label, :] = color

	color_seg = color_seg.astype(np.uint8)
	control_image = Image.fromarray(color_seg)

	return control_image


def softedge(image_path):

	image = load_image(image_path)

	processor = HEDdetector.from_pretrained('lllyasviel/Annotators')

	processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')

	control_image = processor(image, safe=True)

	return control_image


def shuffle(image_path):

	image = load_image(image_path)

	processor = ContentShuffleDetector()

	control_image = processor(image)

	return control_image


def lineart(image_path):

	image = load_image(image_path)

	image = image.resize((512, 512))

	processor = LineartDetector.from_pretrained("lllyasviel/Annotators")

	control_image = processor(image)

	return control_image

def lineart_anime(image_path):

	image = load_image(image_path)

	image = image.resize((512, 512))

	processor = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")

	control_image = processor(image)

def pix_to_pix(image_path):

	control_image = load_image(image_path)

	return control_image



total_images = []

i = 0

for con_type in (controlnet_type):

	if con_type == "canny":

		if ap.pre_process == True:

			image = canny(image_paths[i])

		else:

			image = load_image(image_paths[i])

		total_images.append(image)

	elif con_type == "mlsd":

		if ap.pre_process == True:

			image = mlsd(image_paths[i])

		else:

			image = load_image(image_paths[i])

		total_images.append(image)

	elif con_type == "open_pose":

		if ap.pre_process == True:

			image = open_pose(image_paths[i])
		else:

			image = load_image(image_paths[i])

		total_images.append(image)

	elif con_type == "depth":

		if ap.pre_process == True:

			image = depth(image_paths[i])
		else:

			image = load_image(image_paths[i])

		total_images.append(image)

	elif con_type == "shuffle":

		if ap.pre_process == True:

			image = shuffle(image_paths[i])
		else:

			image = load_image(image_paths[i])

		total_images.append(image)

	elif con_type == "softedge":

		if ap.pre_process == True:

			image = softedge(image_paths[i])
		else:

			image = load_image(image_paths[i])

		total_images.append(image)

	elif con_type == "scribble":

		if ap.pre_process == True:

			image = scribble(image_paths[i])
		else:

			image = load_image(image_paths[i])

		total_images.append(image)

	elif con_type == "lineart_anime":

		if ap.pre_process == True:

			image = lineart_anime(image_paths[i])
		else:

			image = load_image(image_paths[i])

		total_images.append(image)

	elif con_type == "lineart":

		if ap.pre_process == True:

			image = lineart(image_paths[i])
		else:

			image = load_image(image_paths[i])

		total_images.append(image)

	elif con_type == "segmentation":

		if ap.pre_process == True:

			image = seg(image_paths[i])
		else:

			image = load_image(image_paths[i])

		total_images.append(image)

	elif con_type == "normalbae":

		if ap.pre_process == True:

			image = normalbae(image_paths[i])
		else:

			image = load_image(image_paths[i])

		total_images.append(image)

	elif con_type == "pix_to_pix":

		if ap.pre_process == True:

			image = pix_to_pix(image_paths[i])
		else:

			image = load_image(image_paths[i])

		total_images.append(image)

	i += 1


controlnet = []


modeIndex = 0

for con_type_1 in (controlnet_type):

	if con_type_1 == "canny":

		controlnet.append(ControlNetModel.from_pretrained(controlnet_model_paths[modeIndex],torch_dtype=torch.float16))

	elif con_type_1 == "open_pose":

		controlnet.append(ControlNetModel.from_pretrained(controlnet_model_paths[modeIndex],torch_dtype=torch.float16))

	elif con_type_1 == "mlsd":

		controlnet.append(ControlNetModel.from_pretrained(controlnet_model_paths[modeIndex],torch_dtype=torch.float16))

	elif con_type_1 == "depth":

		controlnet.append(ControlNetModel.from_pretrained(controlnet_model_paths[modeIndex],torch_dtype=torch.float16))

	elif con_type_1 == "shuffle":

		controlnet.append(ControlNetModel.from_pretrained(controlnet_model_paths[modeIndex],torch_dtype=torch.float16))

	elif con_type_1 == "softedge":

		controlnet.append(ControlNetModel.from_pretrained(controlnet_model_paths[modeIndex],torch_dtype=torch.float16))

	elif con_type_1 == "scribble":

		controlnet.append(ControlNetModel.from_pretrained(controlnet_model_paths[modeIndex],torch_dtype=torch.float16))

	elif con_type_1 == "lineart_anime":

		controlnet.append(ControlNetModel.from_pretrained(controlnet_model_paths[modeIndex],torch_dtype=torch.float16))

	elif con_type_1 == "lineart":

		controlnet.append(ControlNetModel.from_pretrained(controlnet_model_paths[modeIndex],torch_dtype=torch.float16))

	elif con_type_1 == "segmentation":

		controlnet.append(ControlNetModel.from_pretrained(controlnet_model_paths[modeIndex],torch_dtype=torch.float16))

	elif con_type_1 == "normalbae":

		controlnet.append(ControlNetModel.from_pretrained(controlnet_model_paths[modeIndex],torch_dtype=torch.float16))

	elif con_type_1 == "pix_to_pix":

		controlnet.append(ControlNetModel.from_pretrained(controlnet_model_paths[modeIndex],torch_dtype=torch.float16))

	modeIndex += 1


if ap.single_base_model_file == False:

	pipeline = StableDiffusionControlNetPipeline.from_pretrained(ap.base_model_path, controlnet=controlnet,safety_checker=None,use_safetensors=True,torch_dtype=torch.float16)

else:

	pipeline = StableDiffusionControlNetPipeline.from_single_file(ap.base_model_path, controlnet=controlnet[0],safety_checker=None,use_safetensors=True,torch_dtype=torch.float16)


if ap.scheduler == "DDIMScheduler":
	pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
elif ap.scheduler == "LMSDiscreteScheduler":
	pipeline.scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
elif ap.scheduler == "UniPCMultistepScheduler":
	pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
elif ap.scheduler == "DPMSolverMultistepScheduler":
	pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
elif ap.scheduler == "EulerDiscreteScheduler":
	pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
elif ap.scheduler == "PNDMScheduler":
	pipeline.scheduler = PNDMScheduler.from_config(pipeline.scheduler.config)
elif ap.scheduler == "DDPMScheduler":
	pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
elif ap.scheduler == "EuerlAncestralDiscreteScheduler":
	pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
else:
	pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)


pipeline.enable_model_cpu_offload()

pipeline.enable_xformers_memory_efficient_attention()

generator = torch.Generator().manual_seed(ap.seed)

with autocast("cuda"), torch.inference_mode():

	if len(total_images) == 1:

		images = pipeline(
			prompt=ap.prompt,
			image=total_images,
			width = ap.width,
			height = ap.height,
			generator = generator,
			negative_prompt=ap.negative_prompt,
			num_images_per_prompt=ap.samples,
			num_inference_steps=ap.inference_steps,
			guidance_scale = ap.guidance_scale,).images

	else:

		images = pipeline(
			prompt=ap.prompt,
			image=total_images,
			width = ap.width,
			height = ap.height,
			generator = generator,
			negative_prompt=ap.negative_prompt,
			num_images_per_prompt=ap.samples,
			num_inference_steps=ap.inference_steps,
			guidance_scale = ap.guidance_scale,
			controlnet_conditioning_scale = net_scale,).images

os.makedirs(ap.output, exist_ok=True)
for i, image in enumerate(images):
	image.save(f""+ap.output+"/"+str(i)+".png")
