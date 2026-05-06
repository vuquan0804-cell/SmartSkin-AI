from __future__ import annotations

import base64
import json
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

DEFAULT_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


@dataclass
class Prediction:
    label: str
    probability: float


class SkinModel:
    def __init__(self, model_path: Path, labels_path: Path) -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = self._load_labels()
        self.model = self._build_model(num_classes=len(self.labels))
        self.is_ready = False

        if self.model_path.exists():
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()
            self.is_ready = True

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _load_labels(self) -> List[str]:
        if self.labels_path.exists():
            with self.labels_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data.get("labels", DEFAULT_LABELS)
        return DEFAULT_LABELS

    def _build_model(self, num_classes: int) -> torch.nn.Module:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
        return model

    def predict(self, image: Image.Image, top_k: int = 3) -> List[dict]:
        if not self.is_ready:
            raise RuntimeError("Model weights not loaded")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        top_k = min(top_k, len(self.labels))
        values, indices = torch.topk(probs, top_k)
        results = []
        for value, index in zip(values.tolist(), indices.tolist()):
            results.append({"label": self.labels[index], "probability": value})
        return results

    def gradcam(self, image: Image.Image, class_index: int | None = None) -> dict:
        if not self.is_ready:
            raise RuntimeError("Model weights not loaded")

        target_layer = self.model.features[-1]
        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        def forward_hook(_, __, output):
            activations.append(output)

        def backward_hook(_, __, grad_output):
            gradients.append(grad_output[0])

        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        if class_index is None:
            class_index = int(torch.argmax(logits, dim=1).item())

        score = logits[0, class_index]
        self.model.zero_grad(set_to_none=True)
        score.backward()

        forward_handle.remove()
        backward_handle.remove()

        activation = activations[0]
        gradient = gradients[0]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activation, dim=1)
        cam = F.relu(cam).squeeze(0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        heatmap = cam.detach().cpu().numpy()
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize(image.size, Image.BILINEAR)

        heatmap_arr = np.array(heatmap_img).astype(np.float32) / 255.0
        base_arr = np.array(image).astype(np.float32) / 255.0
        overlay = base_arr.copy()
        overlay[..., 0] = np.clip(overlay[..., 0] + heatmap_arr * 0.6, 0, 1)
        overlay = (overlay * 255).astype(np.uint8)

        output = Image.fromarray(overlay)
        buffer = BytesIO()
        output.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

        return {
            "label": self.labels[class_index],
            "class_index": class_index,
            "overlay_base64": encoded,
        }
