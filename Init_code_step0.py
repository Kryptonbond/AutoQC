## Step 0: Import Datasets

import numpy as np
from glob import glob

import os
os.chdir('/home/deeplearning/Desktop/DL/dog_app/')

# load filenames for human and dog images
##human_files = np.array(glob("./lfw/*/*"))
dog_files = np.array(glob("./dogImages/*/*/*"))    ### <<------- Change the main dataset here keep the asterix as it is.!

# print number of images in each dataset
##print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))

##Step 2: Detect Dogs
import torch
import torchvision.models as models

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()

### (IMPLEMENTATION) Making Predictions with a Pre-trained Model
from PIL import Image
import torchvision.transforms as transforms

def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    image = Image.open(img_path).convert('RGB')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    im_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
          
    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = im_transform(image)[:3,:,:].unsqueeze(0)
    
    if use_cuda:
        image = image.cuda()
    im_VGG = VGG16(image)
    Max_val = torch.max(im_VGG,1)
    Val_index = Max_val[1].item() 
#     torch.max(ret,1)[1].item()
    
    return Val_index # predicted class index


