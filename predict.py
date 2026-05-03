import torch
import torchvision.io as io
import torchvision.transforms as T
from model import CNN_Emoji as TheModel
from config import resize_x, resize_y
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# preprocessing (same as dataset.py)
transform = T.Compose([
    T.Resize((resize_y, resize_x)),
    T.ConvertImageDtype(torch.float32)
])

def the_predictor(list_of_img_paths):
    # load model
    model = TheModel().to(device)
    model.load_state_dict(torch.load("checkpoints/final_weights.pth", map_location=device))
    model.eval()

    # convert to input suitable for the model
    batch = []
    for p in list_of_img_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")

        img = io.read_image(p)      # [3,H,W]
        img = transform(img)        # resize + float32
        batch.append(img)

    batch = torch.stack(batch).to(device)   # [B,3,H,W]

    # predict the outcome
    with torch.no_grad():
        logits = model(batch)
        preds = torch.argmax(logits, dim=1)

    # return list of labels (class indices)
    return preds.cpu().tolist()
