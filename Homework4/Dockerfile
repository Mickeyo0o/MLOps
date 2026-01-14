FROM public.ecr.aws/lambda/python:3.10

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN yum install -y libjpeg-turbo libpng && yum clean all
RUN pip install --no-cache-dir torch torchvision pillow boto3

COPY face_emotion_classifier/saved_model.pt ${LAMBDA_TASK_ROOT}/model/saved_model.pt

# Copy your code
COPY lambda_inference_code.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_inference_code.handler"]