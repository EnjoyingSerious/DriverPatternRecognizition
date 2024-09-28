from Autoencoder import *


def train_model(model, dataloader, device, num_epochs=20, lr=5e-2):
    criterion = nn.MSELoss()  # 重建误差损失
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    noise = None
    for epoch in range(num_epochs):
        
        for batch in dataloader:
            encoder_inputs = batch[0].to(device)
            
            outputs = model(encoder_inputs).to(device)
            
            loss = criterion(outputs, encoder_inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')