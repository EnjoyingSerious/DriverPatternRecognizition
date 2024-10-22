from Autoencoder import *
import os

def train_model(model, dataloader, device, num_epochs=20, lr=5e-2, save_path='./exp/best_model.pth'):
    criterion = nn.MSELoss()  # 重建误差损失
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')   # 初始最佳损失为无穷大
    if not os.path.exists('./exp'):
        os.makedirs('./exp')   # 确保保存检查点的文件夹存在
        
    for epoch in range(num_epochs):
        
        for batch in dataloader:
            encoder_inputs = batch[0].to(device)
            
            outputs = model(encoder_inputs).to(device)
            
            loss = criterion(outputs, encoder_inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        # 检查是否是最优损失
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model with loss :{best_loss:.4f} at epoch {epoch+1}")