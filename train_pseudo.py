import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
from sklearn import preprocessing

class TinySOLDataset_v2(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 custom_label_encoding):
        self.annotations = pd.read_csv(annotations_file)
        self.custom_label_encoding = custom_label_encoding
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        filename = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        
        label = self._get_audio_sample_label(index)
        pseudo =self._get_audio_sample_pseudo_label(index)
        signal, sr = torchaudio.load(filename)

        if signal is None:
            # Handle the case where signal is None
            #  For example, you can return a zero tensor for input and target
            return torch.zeros((1,)), torch.zeros((1,))

        # Calculate the duration of the audio in seconds
        duration_seconds = signal.shape[1] / sr

        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        mel_spec = self.transformation(signal)
        #print(signal.shape)
        #print(mel_spec.shape)
        

        sample = {
            'file': filename,
            'audio': signal,
            'mel': mel_spec,
            'gt': label,
            'pseudo': pseudo,
            'duration_seconds': duration_seconds
        }
        return sample


    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_label(self, index):
        instrument_name = self.annotations.iloc[index, 3]
        label = self.custom_label_encoding.get(instrument_name, -1)  # Default to -1 if not found

        #one-hot encoding
        #num_classes = len(self.custom_label_encoding)
        #one_hot_label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=num_classes)

        return label
    def _get_audio_sample_pseudo_label(self, index):
        instrument_name = self.annotations.iloc[index, 14]
        pseudo = self.custom_label_encoding.get(instrument_name, -1)  # Default to -1 if not found

        #one-hot encoding
        #num_classes = len(self.custom_label_encoding)
        #one_hot_label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=num_classes)

        return pseudo

import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from modelv2 import CNN

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "C:\\Users\\Lenovo\\Desktop\\summer intern\\task1\\updated_metadata.csv"
AUDIO_DIR = "C:\\Users\\Lenovo\\Desktop\\summer intern\\task1"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050





def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=default_collate, shuffle=True)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for batch in data_loader:
        input, target = batch['mel'], batch['pseudo']
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

    dataset = TinySOLDataset_v2(ANNOTATIONS_FILE,
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
    torch.save(cnn.state_dict(), "pseudo_model.pth")
    print("Trained feed forward net saved at pseduo_model.pth")

























