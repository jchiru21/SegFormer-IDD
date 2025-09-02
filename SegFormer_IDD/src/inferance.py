import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

NUM_CLASSES = 19
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IGNORE_INDEX = 255
ORIGINAL_LABEL_IDS = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
label_remap = {orig:i for i, orig in enumerate(ORIGINAL_LABEL_IDS)}

# Load model
model_path = "segformer_checkpoints/best_model.pt"
model_name = "nvidia/segformer-b2-finetuned-ade-512-512"
model = SegformerForSemanticSegmentation.from_pretrained(
    model_name,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Load image
image_path = "test_image.png"  # replace
img = Image.open(image_path).convert("RGB").resize((512,512))

# Feature extractor
feature_extractor = SegformerImageProcessor(do_resize=True, size=(512,512), do_normalize=True)
inputs = feature_extractor(images=img, return_tensors="pt")
pixel_values = inputs["pixel_values"].to(DEVICE)

# Predict
with torch.no_grad():
    logits = model(pixel_values).logits
    preds = logits.argmax(dim=1)[0].cpu().numpy()

# Color map for visualization
colors = np.random.randint(0, 255, size=(NUM_CLASSES,3))
seg_color = colors[preds]

# Overlay
alpha = 0.5
overlay = (np.array(img) * (1-alpha) + seg_color * alpha).astype(np.uint8)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original")
plt.subplot(1,2,2)
plt.imshow(overlay)
