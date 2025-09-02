from transformers import SegformerForSemanticSegmentation

model_name = "nvidia/segformer-b2-finetuned-ade-512-512"
model = SegformerForSemanticSegmentation.from_pretrained(model_name)

# Save locally
save_path = "./segformer_checkpoints/pretrained_segformer_b2"
model.save_pretrained(save_path)
