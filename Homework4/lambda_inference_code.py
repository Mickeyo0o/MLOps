import json
import boto3
import torch
import torchvision.transforms as T
from PIL import Image
import os
import tempfile

from torchvision.models.mobilenetv3 import MobileNetV3
torch.serialization.add_safe_globals([MobileNetV3])

s3 = boto3.client("s3")

device = torch.device("cpu")

MODEL_PATH = "/var/task/model/saved_model.pt"
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.to(device)
model.eval()

transform = T.Compose(
    [
        T.Resize((96, 96)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


def run_inference(image_path: str) -> dict:
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)

    probs = torch.softmax(logits, dim=1)[0]
    pred = probs.argmax().item()

    return {
        "class_index": pred,
        "class_name": CLASSES[pred],
        "confidence": float(probs[pred]),
    }


def handler(event, context):
    record = event["Records"][0]
    bucket = record["s3"]["bucket"]["name"]
    key = record["s3"]["object"]["key"]

    print(f"Processing s3://{bucket}/{key}")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_image_path = os.path.join(tmpdir, "input_image")

        s3.download_file(
            Bucket=bucket,
            Key=key,
            Filename=local_image_path,
        )
        result = run_inference(local_image_path)

    output_key = f"results/{os.path.basename(key)}.json"

    s3.put_object(
        Bucket=bucket,
        Key=output_key,
        Body=json.dumps(result),
        ContentType="application/json",
    )

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "input": f"s3://{bucket}/{key}",
                "output": f"s3://{bucket}/{output_key}",
                "result": result,
            }
        ),
    }