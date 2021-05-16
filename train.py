import torch
from torch.utils.tensorboard import SummaryWriter

import autoencoder
import generate_search_index
import get_dataloader
from torch import nn


def train_model(train_loader, optimizer, loss, net, device, print_every = 1000, verbose=True, log="runs", save_every=5):
    writer = SummaryWriter(log_dir = log)
    net = net.to(device)
    step = 0
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(2):
        total = 0
        total_loss = 0
        for i, sampled_data in enumerate(train_loader):
            input = sampled_data['docstring_emb'].to(device)

            optimizer.zero_grad()
            output = net(input)
            with torch.cuda.amp.autocast():
                loss_value = loss(output, sampled_data['code_emb'].to(device))
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss_value.item()
            total += len(input)

            if step % print_every == 0:
                writer.add_scalar("Loss/train", total_loss/total, step)
                if verbose:
                    print("Epoch: %s --Step: %s Loss: %s" %(epoch, step, total_loss/total))
            step = step +1
        torch.save(net.state_dict(), f'generated_resources/autoencoder_{epoch}.pt')

if __name__ == '__main__':
    train_dataset = get_dataloader.get_dataset('train')
    train_dataloader = get_dataloader.get_dataloaders('train', 32, True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = autoencoder.AutoEncoder(768, 256).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    train_model(train_dataloader, optimizer, criterion, model, device)
    generate_search_index.generate_search_index('train')

