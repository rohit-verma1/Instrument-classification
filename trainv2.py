import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from datav2 import TinySOLDataset
from modelv2 import CNN

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "C:\\Users\\Lenovo\\Desktop\\summer intern\\task1\\TinySOL_metadata.csv"
AUDIO_DIR = "C:\\Users\\Lenovo\\Desktop\\summer intern\\task1"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050





def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=default_collate, shuffle=True)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for batch in data_loader:
        input, target = batch['mel'], batch['gt']
        input, target = input.to(device), target.to(device)
        target = target.long()
        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")

custom_label_encoding = {
        "BTb": 0,
        "Hn": 1,
        "Tbn": 2,
        "TpC": 3,
        "Acc": 4,
        "Vc": 5,
        "Cb": 6,
        "Va": 7,
        "Vn": 8,
        "ASax": 9,
        "Bn": 10,
        "ClBb": 11,
        "Fl": 12,
        "Ob": 13 
        }

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=400,
        hop_length=160,
        n_mels=32
    )

    dataset = TinySOLDataset(ANNOTATIONS_FILE,
                             AUDIO_DIR,
                             mel_spectrogram,
                             SAMPLE_RATE,
                             NUM_SAMPLES,
                             custom_label_encoding
                             )

    train_dataloader = create_data_loader(dataset, BATCH_SIZE)

    # construct model and assign it to device
    cnn = CNN().to(device)
    print(cnn)

    # initialise loss function + optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimizer, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "feedforwardnet.pth")
    print("Trained feed forward net saved at feedforwardnet.pth")
