import torch
import torchaudio
import pandas as pd

import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from train_pseudo import TinySOLDataset_v2
from modelv2 import CNN

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "C:\\Users\\Lenovo\\Desktop\\summer intern\\task1\\updated_metadata.csv"
AUDIO_DIR = "C:\\Users\\Lenovo\\Desktop\\summer intern\\task1"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

from modelv2 import CNN
from train_pseudo import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES,custom_label_encoding
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score ,classification_report

reversed_label_encoding = {
    0: "BTb",
    1: "Hn",
    2: "Tbn",
    3: "TpC",
    4: "Acc",
    5: "Vc",
    6: "Cb",
    7: "Va",
    8: "Vn",
    9: "ASax",
    10: "Bn",
    11: "ClBb",
    12: "Fl",
    13: "Ob"
}
def predict(model, input, target, reversed_label_encoding):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        #print("Raw Predictions:", predictions)
        #print("Predicted Index:", predictions.argmax(dim=1))
        predicted_index = predictions.argmax(dim=1).item()
        #print(predicted_index.size)
        predicted = reversed_label_encoding[predicted_index]
        expected = reversed_label_encoding[target]

        pseduo_labels = predicted

    return predicted, expected ,predicted_index

if __name__ == "__main__":
    # load back the model
    cnn = CNN()
    state_dict = torch.load("pseudo_model.pth")
    cnn.load_state_dict(state_dict)

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=400,
        hop_length=160,
        n_mels=32
    )

    testing = TinySOLDataset_v2(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            custom_label_encoding)
   
    # Initialize lists to store predicted and expected labels
    all_predicted_labels = []
    all_expected_labels = []

    random_indices_list = torch.randint(0, len(testing), (2913,)).tolist()
    for index in random_indices_list:
        # Get data point at the current index
        data_point = testing[index]
        
        # Extract input and target
        input, target = data_point['mel'], data_point['pseudo']
        input.unsqueeze_(0)
        
        # Make an inference
        predicted, expected ,prediction_psudeo_train = predict(cnn, input, target, reversed_label_encoding)
        all_predicted_labels.append(predicted)
        all_expected_labels.append(expected)
        
    accuracy = accuracy_score(all_expected_labels, all_predicted_labels)
    print(f"Accuracy with pseudo labels: {accuracy}")

    
    

    #Print classification report
    print("Classification Report:")
    print(classification_report(all_expected_labels, all_predicted_labels))







