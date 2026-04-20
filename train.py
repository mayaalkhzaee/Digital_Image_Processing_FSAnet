import torch
from torch.utils.data import DataLoader
from dataset import DubaiDataset 
from model import get_baseline_model


def collate_fn(batch): #Because each image can have a different number of buildings
    return tuple(zip(*batch))

def train_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training on: {device}") 

    dataset = DubaiDataset(img_dir='data/train/images', mask_dir='data/train/masks')
    
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)

    model = get_baseline_model(num_classes=2) 
    model.to(device) 

    params = [p for p in model.parameters() if p.requires_grad] #Only optimize the parameters that require gradients
    optimizer = torch.optim.Adam(params, lr=0.0001)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train() 
        epoch_loss = 0 #Track the total loss for the epoch
        
        for images, targets in data_loader:
            images = list(image.to(device) for image in images) 
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets] 

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values()) 
            epoch_loss += losses.item()

            optimizer.zero_grad() 
            losses.backward() 
            optimizer.step() #Update the model parameters based on the computed gradients

        print(f"Epoch: {epoch+1}/{num_epochs}, Average Loss: {epoch_loss/len(data_loader):.4f}")

    torch.save(model.state_dict(), 'baseline_maskrcnn.pth')
    
if __name__ == '__main__':
    train_model()