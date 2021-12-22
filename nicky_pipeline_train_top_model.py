##michael import nnecessary py files

from nicky_helpers import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BEST_MODEL_DATASET = TRAIN_FOLDER + 'BESTMODELDATA/'


unet_pretrained_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)
model = Unet_with_aux_loss_tanh(unet_pretrained_model)
#model.load_state_dict(torch.load('/content/drive/MyDrive/ml/results/best_model_alpha05'))
model.to(device)

mean,std = compute_mean_std(train_set)
print(mean,std)
preprocess_input = transforms.Compose([transforms.Normalize(mean=mean, std=std)])


pos_weight = compute_pos_weight_matrix(train_set,aux_loss = True)

pos_weight= pos_weight.to(device)
criterion2 = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


model.train()
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)
alpha = 0.5
betha = 0.5
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',patience = 4,threshold= 0.001,threshold_mode = 'abs',verbose = True,factor=0.5 )
train_loss,val_loss , _ , _ = train_model(model,train_set,test_set,optimizer,criterion,scheduler,
                                          '/content/drive/MyDrive/ml/results/best_model_alpha50',preprocess_input,preprocess_label_basic_unet,alpha = alpha,betha = betha,mini_batch_size = 10,nb_epochs = 50, criterion2 = criterion2,
                                          use_scheduler = True, print_progress= True,aux_loss = True)
