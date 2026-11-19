import torch
from Missile_or_Not_v1 import LocalizationNet

model = LocalizationNet()
model.load_state_dict(torch.load("st_output/st_lstm_autoencoder.pt", weights_only=True))
model.eval()