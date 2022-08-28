
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import tqdm
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import torch.optim as optim
from cvae_model import ConditionalVAE


torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


CHANNELS_IMG = 1
IMG_SIZE = 32
BATCH_SIZE = 128
EPOCHS = 40


transform = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
    ]
)


trainset = torchvision.datasets.MNIST(root='../data', 
                                      train=True,
                                      transform=transform,
                                      download= True)
testset = torchvision.datasets.MNIST(root='../data', 
                                      train=False,
                                      transform=transform,
                                      download= True)


train_dataloarder = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

model = ConditionalVAE(in_channels=CHANNELS_IMG,latent_dim=2,num_classes=len(trainset.classes),image_size=IMG_SIZE,device=device)

model.to(device)

optimizer = optim.Adam(model.parameters())


def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x.flatten(1), x.flatten(1), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


writer_real = SummaryWriter(f"logs/CVAE_MNIST/real")
writer_fake = SummaryWriter(f"logs/CVAE_MNIST/fake")
step = 0
train_losses = []
test_losses = []
for epoch in tqdm.tqdm(range(EPOCHS)):
    train_epoch_loss = 0
    test_epoch_loss = 0
    for batch_idx,(img,lab) in enumerate(train_dataloarder):
        model.train()
        img = img.to(device)
        lab = lab.to(device)
        optimizer.zero_grad()
        recon_img, mu, log_var = model(img,lab)
        #print(mu)
        #print(log_var)
    
        loss = loss_function(recon_img, img, mu, log_var)
        
        #print(loss.item())
        train_epoch_loss += loss.item() 
        loss.backward()
        optimizer.step()
    
        model.eval()
        if batch_idx % 100 == 0 and batch_idx > 0:
                # print(
                #    f"Epoch [{epoch}/{EPOCHS}] Batch {batch_idx}/{len(train_dataloarder)} \
                #      Loss : {train_epoch_loss:.4f}"
                # )

                with torch.no_grad():
                    gen_img = model.sample(BATCH_SIZE,lab)
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(img[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(gen_img[:32], normalize=True)

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1
    train_losses.append(train_epoch_loss)
    model.eval()
    with torch.no_grad():
        for img,lab in test_loader:
            img = img.to(device)
            lab = lab.to(device)
            recon_img, mu, log_var = model(img,lab)
            loss = loss_function(recon_img, img, mu, log_var)
            test_epoch_loss += loss.item()
    test_losses.append(test_epoch_loss)


losses_df = pd.DataFrame({"train_loss":train_losses,"test_loss":test_losses})
losses_df.to_csv("models_save/losses.csv")

torch.save(model.state_dict(), f"models_save/{EPOCHS}_epochs_cvae_model.pt")





