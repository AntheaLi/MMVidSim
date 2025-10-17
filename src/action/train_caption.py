'''
Created on Fri Nov 17 10:57:21 2023

@author: Kendrick
'''
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from models.sensor_transformer import SensorTransformer
from models.sensor_text_decoder import SensorCaption
from transformers import BertTokenizer
from data.create_dataset import create_samples, split_data_subject
from data.dataset import ActionSenseCaption
from config import sensor_config
from tqdm import tqdm

def train_caption(**kwargs):
    batch_size = kwargs["batch_size"]
    device = kwargs["device"]
    epochs = kwargs["epochs"]
    
    trainset = ActionSenseCaption("ActionSense/Dataset/train")
    validset = ActionSenseCaption("ActionSense/Dataset/valid")
    trainloader = DataLoader(trainset, batch_size, shuffle=True, collate_fn=trainset.collect_fn, drop_last=True)
    validloader = DataLoader(validset, batch_size, collate_fn=validset.collect_fn)
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'eos_token':'[SEP]'})
    
    sensor_encoder = SensorTransformer(
        sensor_config.sensor_cfg,
        device)
    
    if not os.path.exists(kwargs["ckpt_path"]):
        os.makedirs(kwargs["ckpt_path"], exist_ok=True)
    if kwargs["sensor_encoder_ckpt"] is not None:
        sensor_encoder.load_state_dict(torch.load(kwargs["sensor_encoder_ckpt"]))
    
    model = SensorCaption(tokenizer,
                          sensor_encoder,
                          "bert-base-uncased",
                          device=device)
    if kwargs["continue_train"] == True:
        ckpt_path = kwargs["ckpt_path"]
        model_name = kwargs["model_name"]
        save_path = os.path.join(ckpt_path, model_name)
        model.load_state_dict(torch.load(save_path))

    model = model.to(device)
    optimizer = optim.AdamW(model.text_decoder.parameters(), kwargs["lr"])
    best_val_loss = 1e13
    best_val_acc = 0
    for i_epoch in range(epochs):
        train_loss = 0
        valid_loss = 0
        model.train()
        for i, (signal, text) in enumerate(tqdm(trainloader)):
            joint = signal["xsens"].to(device).float()
            emg = signal["emg"].to(device).float()
            tactile = signal["tactile"].to(device).float()

            text = tokenizer(text, padding=True, return_tensors="pt").to(device)
            input_ids = text.input_ids.clone()
            input_ids[:, 0] = tokenizer.bos_token_id
            output = model(joint=joint,
                           emg=emg,
                           tactile=tactile,
                           text=text.input_ids,
                           text_attention_mask=text.attention_mask)
            loss = output.loss
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_avg_loss = round(train_loss/(i+1), 4)
        model.eval()
        with torch.no_grad():
            for i, (signal, text) in enumerate(tqdm(validloader)):
                joint = signal["xsens"].to(device).float()
                emg = signal["emg"].to(device).float()
                tactile = signal["tactile"].to(device).float()
                
                text = tokenizer(text, padding=True, return_tensors="pt").to(device)
                output = model(
                            joint=joint,
                            emg=emg,
                            tactile=tactile,
                            text=text.input_ids,
                            text_attention_mask=text.attention_mask)
                
                loss = output.loss
                valid_loss += loss.item()
                
            valid_avg_loss = round(valid_loss/(i+1), 4)
        print(f"Epoch:{i_epoch +1 } || Train Loss:{train_avg_loss} || Valid Loss:{valid_avg_loss}")
    
        if valid_avg_loss <= best_val_loss:
            ckpt_path = kwargs["ckpt_path"]
            model_name = kwargs["model_name"]
            save_path = os.path.join(ckpt_path, model_name)
            torch.save(model.state_dict(),  save_path)

if __name__ == "__main__":
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    data = create_samples()
    subjects = ["S00", "S02", "S03", "S04", "S05"]
    
    for leave_out in subjects:
        print(f"Leave subject {leave_out} out...")
        args = {
        "batch_size": 88,
        "lr": 5e-5,
        "device": device,
        "epochs": 30,
        "sensor_encoder_ckpt": "pretrain_sensor_encoder_weight.pt",
        "ckpt_path": "./checkpoints/sensor2text",
        "model_name": f"sensor2text_finetune_{leave_out}.pt",
        "continue_train": False
        }

        split_data_subject(data, leave_out)
        print("Start Training...")
        train_caption(**args)