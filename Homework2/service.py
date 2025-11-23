import torch
import bentoml

MODEL_TAG = "fashion_mnist_lit:latest"

@bentoml.service
class FashionMNISTService:
    def __init__(self) -> None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        self.model = bentoml.pytorch.load_model(MODEL_TAG, device_id=device_str)
        self.model.eval()

    @bentoml.api
    def predict(self, images: torch.Tensor) -> list[int]:
        with torch.no_grad():
            images = images.to(self.device)
            logits = self.model(images)
            preds = logits.argmax(dim=1)
        return preds.cpu().tolist()
