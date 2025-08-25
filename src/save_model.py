import torch
import logging

import customTorchLightning


best_ckpt_path = "../saved_models/checkpoints/epoch=29-step=660.ckpt"
logging.info(f"Loading best checkpoint from: {best_ckpt_path}")
best_model = customTorchLightning.CNNAudioClassifierLightning.load_from_checkpoint(
    best_ckpt_path,
    output_dim=7  # or dataset.num_classes
)
best_model.eval()

torch.save(best_model.state_dict(), f"best_model_rock_metal.pth")
logging.info("Best model saved as best_model.pt")

dummy_input = torch.randn(1, 1, 256, 10336, device="cuda")

torch.onnx.export(
    best_model,
    dummy_input,
    f"best_model_rock_metal.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    export_params=True,
)
logging.info("Best model exported as best_model.onnx")