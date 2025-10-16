# advanced_main.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from vgg7_ann import VGG7
from torch.cuda.amp import autocast, GradScaler
import time

def train_epoch(net, trainloader, criterion, optimizer, scaler, device, use_mixup=True, mixup_alpha=1.0):
    """Advanced training loop with mixup and mixed precision"""
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Mixup augmentation
        if use_mixup and torch.rand(1).item() > 0.5:
            lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
            rand_index = torch.randperm(inputs.size(0)).to(device)
            target_a = targets
            target_b = targets[rand_index]
            inputs = lam * inputs + (1 - lam) * inputs[rand_index]
            
            optimizer.zero_grad()
            with autocast():
                outputs = net(inputs)
                loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)
        else:
            optimizer.zero_grad()
            with autocast():
                outputs = net(inputs)
                loss = criterion(outputs, targets)
        
        # Mixed precision training
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return train_loss/(batch_idx+1), 100.*correct/total

def test(net, testloader, criterion, device):
    """Evaluation loop"""
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return test_loss/(batch_idx+1), 100.*correct/total

def main():
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # ===== HYPERPARAMETERS =====
    epochs = 200  # More epochs with better optimization
    batch_size = 128
    initial_lr = 0.1  # Higher initial learning rate
    
    # Advanced augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # Add rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Color augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),  # Cutout/Random Erasing
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load CIFAR-10 Dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)
    
    # Initialize Model
    net = VGG7(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
    
    # ===== OPTIMIZER CHOICE =====
    # Option 1: AdamW (modern, robust)
    optimizer = optim.AdamW(net.parameters(), lr=initial_lr, weight_decay=5e-4)
    
    # Option 2: SGD with Nesterov (uncomment to use)
    # optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9, 
    #                       weight_decay=5e-4, nesterov=True)
    
    # ===== LEARNING RATE SCHEDULER =====
    # OneCycleLR - highly effective
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=initial_lr,
        epochs=epochs,
        steps_per_epoch=len(trainloader),
        pct_start=0.3,  # 30% warmup
        anneal_strategy='cos'
    )
    
    # Alternative: CosineAnnealingWarmRestarts
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=10, T_mult=2, eta_min=1e-6
    # )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Early stopping variables
    best_acc = 0
    patience = 20
    patience_counter = 0
    
    print("\n===== Training Configuration =====")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Initial LR: {initial_lr}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Scheduler: {scheduler.__class__.__name__}")
    print(f"Using Mixup: True")
    print(f"Using Label Smoothing: True")
    print(f"Using Mixed Precision: True")
    print("===================================\n")
    
    # Training Loop
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            net, trainloader, criterion, optimizer, scaler, device, 
            use_mixup=True, mixup_alpha=1.0
        )
        
        # Test
        test_loss, test_acc = test(net, testloader, criterion, device)
        
        # Update learning rate (for non-OneCycleLR schedulers)
        if not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch [{epoch+1}/{epochs}] | '
              f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}% | '
              f'LR: {current_lr:.6f} | Time: {epoch_time:.2f}s')
        
        # Save best model
        if test_acc > best_acc:
            print(f'>>> Saving best model (Acc: {test_acc:.2f}%)')
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, 'vgg7_cifar10_best.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break
    
    total_time = time.time() - start_time
    print(f'\n===== Training Complete =====')
    print(f'Total Time: {total_time/60:.2f} minutes')
    print(f'Best Test Accuracy: {best_acc:.2f}%')
    
    # Load best model and save final version
    checkpoint = torch.load('vgg7_cifar10_best.pth')
    net.load_state_dict(checkpoint['model_state_dict'])
    torch.save(net.state_dict(), 'vgg7_cifar10_advanced.pth')
    print('Best model saved to vgg7_cifar10_advanced.pth')

if __name__ == '__main__':
    main()