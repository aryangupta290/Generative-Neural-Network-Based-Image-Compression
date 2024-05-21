from celebA import celebA
import torchvision
from utils import *
from compressor import *
import torch
import inquirer
import datetime
import os
from analysis import *
questions = [
    inquirer.Text('image',
                  message="Enter the image file name",
                  ),
]
answers = inquirer.prompt(questions)

answers['image'] = "image_1.png"
# Define a folder wrt time and date to save the images
path = "../57/" + str(datetime.datetime.now())
print(path)
answers['save_path'] = path

# Create the folder if it doesn't exist
os.makedirs(path, exist_ok=True)
# Check if image file exists
try:
    open(answers['image'])
except FileNotFoundError:
    print("File not found")
    exit()

gpu = True if torch.cuda.is_available() else False
model = torch.hub.load(
    "facebookresearch/pytorch_GAN_zoo:hub", "PGAN", model_name="celebAHQ-512", pretrained=True, useGPU=gpu
).netG
# noise = torch.randn(1, 512)
# noise_image = model(noise)
for param in model.parameters():
    param.requires_grad = False

image = Image.open(answers['image'])
image = image.convert("RGB")
convert = torchvision.transforms.ToTensor()
image = convert(image)
latent_vector = image.unsqueeze(0)
save_image(latent_vector, answers['save_path'], "original")
# o_img = noise_image.clone().detach().cpu()
# save_image(o_img, answers['save_path'], "original")
compressed_image_vector = compressor(
    model, answers['save_path'], "GAN", latent_vector)
torch.save(compressed_image_vector, f"{answers['save_path']}/SSIM.pt")
# compressed_noise_vector = compressor(
#     model, answers['save_path'], "GAN", noise)
# torch.save(compressed_noise_vector, f"{answers['save_path']}/SSIM.pt")

analysis(path+"/original/image_.png", compressed_folder=path+"/compressed/",
         output_folder=path + '/analysis/', GAN_folder=path + '/GAN/')
