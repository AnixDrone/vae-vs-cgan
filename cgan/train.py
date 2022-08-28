import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import gradient_penalty, save_checkpoint, load_checkpoint
from model import Discriminator, Generator, initialize_weights
import pandas as pd


device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMG_SIZE = 64
NUM_CLASSES = 10
GEN_EMBEDDING = 100
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 20
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

transforms = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
    ]
)

dataset = datasets.MNIST(root="../mnist_datasets/data", transform=transforms, download=True)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN,NUM_CLASSES,IMG_SIZE,GEN_EMBEDDING).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC,NUM_CLASSES,IMG_SIZE).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/CGAN/real")
writer_fake = SummaryWriter(f"logs/CGAN/fake")
step = 0

gen.train()
critic.train()

generator_losses = []
critic_losses = []

for epoch in tqdm.tqdm(range(NUM_EPOCHS)):
    for batch_idx, (real, labels) in enumerate(loader):
        real = real.to(device)
        labels = labels.to(device)
        cur_batch_size = real.shape[0]
        
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise,labels)
            critic_real = critic(real,labels).reshape(-1)
            critic_fake = critic(fake,labels).reshape(-1)
            gp = gradient_penalty(critic,labels, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        gen_fake = critic(fake,labels).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0 and batch_idx > 0:
            

            with torch.no_grad():
                fake = gen(noise,labels)
                
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
            # print(
            #     f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
            #       Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            # )
            generator_losses.append(loss_gen.item())
            critic_losses.append(loss_critic.item())

            step += 1


losses_df = pd.DataFrame({"generator_loss":generator_losses, "critic_loss":critic_losses})
losses_df.to_csv("models_saved/losses.csv")

torch.save(gen.state_dict(), f"models_saved/{NUM_EPOCHS}_epochs_generator_model.pt")
torch.save(critic.state_dict(), f"models_saved/{NUM_EPOCHS}_epochs_critic_model.pt")
