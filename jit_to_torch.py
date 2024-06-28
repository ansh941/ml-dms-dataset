import os
import logging
import argparse

import numpy as np
import cv2
from PIL import Image

import torch
import torchvision.transforms as TTR

from models.dms import DMS

logging.basicConfig(level=logging.INFO)

def jit_to_torch(jit_path='checkpoints/DMS46_v1.pt', torch_path='checkpoints/DMS46_v1.pth', test_image_path='4.png'):
    # Load the model
    logging.info('Loading the jit model...')
    jit_model = torch.jit.load(jit_path)
    jit_named_params = dict(jit_model.named_parameters())
        
    # Weight injection
    logging.info('Weight injection...')
    torch_model = DMS()
    
    for ((torch_name, torch_module), (jit_name, jit_module)) in zip(torch_model.named_modules(), jit_model.named_modules()):
        torch_module.load_state_dict(jit_module.state_dict())
    
    # Test
    torch_model.eval()
    dummy_input = torch.randn(1, 3, 512, 512)
    
    # test image setting
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    img = Image.open(test_image_path).resize((512,512), Image.Resampling.LANCZOS)
    img = np.array(img)[...,:3]
    image = torch.from_numpy(img.transpose((2, 0, 1))).float()
    image = TTR.Normalize(mean, std)(image).unsqueeze(0)
    
    with torch.no_grad():
        jit_encoder_output = jit_model.encoder.forward(dummy_input)
        torch_encoder_output = torch_model.encoder(dummy_input)
        logging.info(f'jit_encoder_out and torch_encoder_out are similar : {torch.allclose(jit_encoder_output, torch_encoder_output)}')
        
        jit_decoder_output = jit_model(image)[0].data.cpu()[0, 0].numpy()
        torch_decoder_output = torch_model(image, (512, 512))
        output = torch.argmax(torch_decoder_output, dim=1).cpu()[0].numpy()
        
        cv2.imwrite('output_jit.png', jit_decoder_output)
        cv2.imwrite('output_torch.png', output)
    
    torch.save(torch_model.state_dict(), torch_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert jit model to torch model')
    parser.add_argument('--jit_path', type=str, default='checkpoints/DMS46_v1.pt', help='jit model path')
    parser.add_argument('--torch_path', type=str, default='checkpoints/DMS46_v1.pth', help='torch model path for saving')
    parser.add_argument('--test_image_path', type=str, default='4.png', help='test image path')
    
    args = parser.parse_args()
    
    jit_path = args.jit_path
    torch_path = args.torch_path
    test_image_path = args.test_image_path
    
    if not os.path.exists(torch_path):
        os.makedirs(os.path.dirname(jit_path), exist_ok=True)
        jit_to_torch(jit_path=jit_path, torch_path=torch_path, test_image_path=test_image_path)