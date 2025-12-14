import bentoml
import torch
from PIL import Image as PILImage
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

@bentoml.service(resources={"cpu": "2"}, traffic={"timeout": 30})
class MobileNetV3SmallService:
    def __init__(self) -> None:
        self.device_id = "cpu"
        self.device = torch.device(self.device_id)
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.categories = weights.meta["categories"]
        self.model = models.mobilenet_v3_small(weights=weights).to(self.device)
        self.model.eval()

    @bentoml.api(route="/predict")
    def predict(self, image: PILImage.Image, /) -> dict:
        if image.mode != "RGB":
            image = image.convert("RGB")
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)[0]
            probs = torch.softmax(logits, dim=0)
        topk = torch.topk(probs, k=5)
        top5 = [
            {"label": self.categories[int(i)], "score": float(s)}
            for s, i in zip(topk.values.tolist(), topk.indices.tolist())
        ]
        return {"top5": top5}
