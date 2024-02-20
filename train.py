import torch
from torch import nn
import torchaudio
from torch.utils.data import DataLoader
from dataset import UrbanSoundDataset
from model import Network

BATCH_SIZE = 256
EPOCHS = 25
LEARNING_RATE = 0.01

ANNOTATIONS_FILE = 'dataset/UrbanSound8K.csv'
AUDIO_DIR = 'dataset'
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimizer, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        # loss function
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimizer, epochs, device):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, data_loader, loss_fn, optimizer, device)
        print("-----------------------------------------------")
    print("Training completed!!")


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    # instantiate dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram,
                            SAMPLE_RATE, NUM_SAMPLES, device)

    train_dataloader = create_data_loader(usd, BATCH_SIZE)

    cnn = Network().to(device)
    print(cnn)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    # train
    train(cnn, train_dataloader, loss_fn, optimizer, EPOCHS, device)

    # save model
    torch.save(cnn.state_dict(), "saved_model/sound_classifier.pth")
    print("Trained model saved at sound_classifier.pth")
