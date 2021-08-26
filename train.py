from data_generation.data_sets import ImageDataset
from models.mobilenet import MobileNet
from models.resnet import ResNet
from models.densenet import DenseNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
from torchvision import transforms
from  torch.utils.data import Dataset, DataLoader

def train(model, criterion, optimizer, train_loader, batch_size, device):
    model.train()

    training_loss = 0
    training_acc = 0

    total_batch = len(train_loader)

    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        pred = output.data.max(dim = 1, keepdim = True)[1]

        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        training_loss += loss / total_batch
        training_acc += pred.eq(y.data.view_as(pred)).cpu().sum() / batch_size / total_batch
        
    return training_loss, training_acc


def evaluate(model, criterion, val_loader, device, batch_size):
    model.eval()

    val_loss = 0
    val_acc = 0  

    total_batch = len(val_loader)

    with torch.no_grad(): 
        for x,y in val_loader:  
            x, y = x.to(device), y.to(device)  
            output = model(x) 
            pred = output.max(1, keepdim=True)[1]
            
            loss = criterion(output, y)
            
            val_loss += loss / total_batch
            val_acc += pred.eq(y.data.view_as(pred)).cpu().sum() / batch_size / total_batch 
            
    return val_loss, val_acc


def train_epochs(model, epochs, criterion, optimizer, scheduler, train_loader, val_loader, batch_size, device, checkpoint_path):
    
    model.to(device)
    criterion.to(device)
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        print('-------------- epoch {} ----------------'.format(epoch))

        start = time.time()
        train_loss, train_acc = train(model, criterion, optimizer, train_loader, batch_size, device)
        val_loss, val_acc = evaluate(model, criterion, val_loader, device, batch_size)
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            model_path = "{}/{}_{:.4f}_{:.4f}.pt".format(checkpoint_path, epoch, val_loss, val_acc)
            torch.save(model.state_dict(), model_path)
        
        time_elapsed = time.time() - start
        
        print('train Loss: {:.4f}, Accuracy: {:.4f}'.format(train_loss, train_acc))
        print('val Loss: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_acc))
        print('Completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128
    EPOCHS = 30

    model = MobileNet()
    model.to(DEVICE)
    

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.5, patience = 5)
     
    transform_train = transforms.Compose([
        transforms.RandomAffine(20),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.CenterCrop(280),
        transforms.ToTensor(),
        transforms.Normalize([0.5601, 0.5241, 0.5014], [0.2331, 0.2430, 0.2456])]) 
    
    transform_val = transforms.Compose([
        transforms.CenterCrop(280),
        transforms.ToTensor(),
        transforms.Normalize([0.5601, 0.5241, 0.5014], [0.2331, 0.2430, 0.2456])]) 

    train_dataset = ImageDataset("./data/train/train1", transforms = transform_train)
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, drop_last = True, num_workers = 4)
    
    val_dataset = ImageDataset("./data/train/val1", transforms = transform_val)
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, drop_last = True, num_workers = 8)

    train_epochs(model, EPOCHS, criterion, optimizer, scheduler, train_loader, val_loader, BATCH_SIZE, DEVICE, "./results/models/mobile_centercrop280")