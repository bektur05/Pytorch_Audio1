from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms
import soundfile as sf
import io
import uvicorn


class CheckAudio(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 12, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels = torch.load("label.pth")
index_to_label = {i: l for i, l in enumerate(labels)}
num_classes = len(labels)


model = CheckAudio(num_classes)
model.load_state_dict(torch.load("ModelAudio.pth", map_location=device))
model.to(device)
model.eval()


mel_transform = transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=64
)
max_len = 100


def preprocess_audio(waveform, sample_rate):

    if sample_rate != 16000:
        resampler = transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(torch.tensor(waveform.T))

    spec = mel_transform(waveform).squeeze(0)

    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]
    elif spec.shape[1] < max_len:
        pad_amount = max_len - spec.shape[1]
        spec = F.pad(spec, (0, pad_amount))


    spec = spec.unsqueeze(0).unsqueeze(0)
    return spec.to(device)



app = FastAPI(title="Audio Classifier API", version="1.0")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Файл бош")

        waveform, sample_rate = sf.read(io.BytesIO(data), dtype="float32")

        if len(waveform.shape) > 1:  
            waveform = waveform.mean(axis=1)

        waveform = torch.tensor(waveform).unsqueeze(0)  # (1, N)

        spec = preprocess_audio(waveform, sample_rate)

        with torch.no_grad():
            output = model(spec)
            pred_index = torch.argmax(output, dim=1).item()
            pred_label = index_to_label[pred_index]

        return {"class": pred_label, "index": pred_index}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8077)



