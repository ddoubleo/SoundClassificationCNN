import torch
import torchaudio
from dataset import UrbanSoundDataset
from model import Network
from train import AUDIO_DIR, ANNOTATIONS_FILE, NUM_SAMPLES, SAMPLE_RATE

class_map = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]
guessed = {
    "air_conditioner": 0,
    "car_horn": 0,
    "children_playing": 0,
    "dog_bark": 0,
    "drilling": 0,
    "engine_idling": 0,
    "gun_shot": 0,
    "jackhammer": 0,
    "siren": 0,
    "street_music": 0
}

outputs = {
    "air_conditioner": 0,
    "car_horn": 0,
    "children_playing": 0,
    "dog_bark": 0,
    "drilling": 0,
    "engine_idling": 0,
    "gun_shot": 0,
    "jackhammer": 0,
    "siren": 0,
    "street_music": 0
}


def predict(model, input, target, class_map):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_map[predicted_index]
        expected = class_map[target]
    return predicted, expected


if __name__ == "__main__":
    # load back the model
    cnn = Network()
    cnn = cnn.to('cuda')
    state_dict = torch.load("saved_model/sound_classifier.pth")
    cnn.load_state_dict(state_dict)
    # load urban sound dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram,
                            SAMPLE_RATE, NUM_SAMPLES, "cuda")
    cnt = 0
    for i in range(0, 8733):
        input, target = usd[i][0], usd[i][1]
        input.cuda()
        input.unsqueeze_(0)
        predicted, expected = predict(cnn, input, target,
                                      class_map)
        outputs[predicted] = outputs[predicted] + 1
        if predicted == expected:
            guessed[predicted] = guessed[predicted] + 1
            cnt = cnt + 1
    print(cnt / 8733)
    for item in guessed:
        print(item, ':', guessed[item])
    for item in outputs:
        print(item, ':', outputs[item])

    # get a sample from the urban sounds dataset for inference
    # input, target = usd[0][0], usd[0][1]  # [num_cha, fr, t]
    # input.unsqueeze_(0)
    # # make an inference
    # predicted, expected = predict(cnn, input, target,
    #                               class_map)
    # print(f"Predicted: '{predicted}', expected: '{expected}'")
