
import torch
from PIL import Image
from models import *

from model_training import *

# Loads test_images. Then the model. Then writes to folder
def predict_with_best_model(path_to_test_images_folder, folder_to_write_in,path_to_best_model):
    timg = load_test_images(path_to_test_images_folder)
    unet_pretrained_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                           in_channels=3, out_channels=1, init_features=32, pretrained=True)
    model = Unet_with_aux_loss_tanh(unet_pretrained_model)
    model.load_state_dict(torch.load(path_to_best_model))
    model.eval()
    mean = 0.3082
    std = 0.5541
    preprocess_input = transforms.Compose([transforms.Normalize(mean=mean, std=std)])
    for i in range(0,50):
        input = preprocess_input(timg[i].unsqueeze(0))
        a,_ = model(input.to(device))
        a = a.to('cpu').detach()
        #a = avg_pool(a)
        #a = preprocess_label_basic_unet(a,threshold = 0.3)
        a,_ = torch_to_numpy_format(a,torch.zeros(1,608,608))

        a = a * 255
        if i < 9 :
            Image.fromarray(a[0,:,:,0].astype(np.uint8)).save(folder_to_write_in + "prediction_00" + str(i+1) + ".png")
        else:
            Image.fromarray(a[0,:,:,0].astype(np.uint8)).save(folder_to_write_in + "prediction_0" + str(i+1) + ".png")
