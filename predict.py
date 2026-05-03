import torch
import torchvision.io as io
import torch.nn.functional as F
from model import TheModel
from config import resize_x, resize_y
import torchvision.transforms as T

device = "cuda" if torch.cuda.is_available() else "cpu"

# preprocessing transform (same as dataset.py)
transform = T.Compose([
    T.Resize((resize_y, resize_x)),
    T.ConvertImageDtype(torch.float32)
])

def the_predictor(list_of_img_paths):
    """
    Takes a list of file paths (strings) from data/ directory
    Returns a list of predicted class indices
    """

    # Load model
    model = TheModel().to(device)
    model.load_state_dict(torch.load("checkpoints/final_weights.pth", map_location=device))
    model.eval()

    batch = []

    # Load each image
    for p in list_of_img_paths:
        img = io.read_image(p)          # shape: [3,H,W]
        img = transform(img)            # resize + float32
        batch.append(img)

    batch = torch.stack(batch).to(device)   # shape: [B,3,H,W]

    with torch.no_grad():
        logits = model(batch)               # shape: [B, num_classes]
        preds = torch.argmax(logits, dim=1) # shape: [B]

    return preds.cpu().tolist()
