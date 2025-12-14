# **Homework: Deploy Your Deep Learning Model on a Cloud VM Using a Model Serving Framework**

**Deadline:** **15.01.2025**

In this assignment, you will take **any deep learning model of your choice** (recommended: a tiny ResNet variant pre-trained or fine-tuned on ImageNet) and deploy it to a **cloud virtual machine**.  
If you are starting I recommend using **AWS EC2**, as demonstrated in the class, but you may of course also use **Google Cloud (GCE)** or **Microsoft Azure (VMs)**—the steps are very similar across all providers.

Your deployment should expose an HTTP endpoint that I will call during evaluation.

---

## **Task Summary**

1. Choose and prepare **any DL model** (image classifier, NLP model, regression model, etc.).  
   - Recommended: a tiny ResNet classifier for simplicity and speed. 
   - Recommend some one-liners models from huggin face
   - IF you are a giga-chad use the model from HOMEWORK_1 or a model from any of your other projects 
2. Package the model using a serving tool such as:
   - **NVIDIA Triton**
   - **BentoML**
   - **TorchServe**
   - or a comparable alternative
3. Containerize your application using **Docker**. (Notice here that the serving tool will most likely help you to containerize itself, for example bentoml has a command for that)
4. Deploy your Docker container on a cloud VM (AWS/GCP/Azure).
5. Provide a **Python test script (using only `requests`)** that sends an inference request to your deployed service.

During grading, I will run this script (with your endpoint URL inserted) to validate your deployment.

---

## **Improved Step-by-Step Deployment Guide**

### **1. Choose & Prepare Your Model Locally**
- Pick any deep learning model (recommended: small ResNet variant).
- Train, fine-tune, or use a pre-trained model.
- Export it in the correct format for your serving framework.

---

### **2. Test Your Model Serving Locally**
- Integrate your model with Triton, BentoML, TorchServe, etc.
- Launch the server locally (optionally in Docker).
- Verify inference using a simple `curl` or Python request.

---

## **3. Prepare Your Deployment Container**
Select one of the following:

### **Option A — Build Locally & Push to Registry**
1. Build your Docker image locally.
2. Push it to a container registry:
   - **ECR** if using AWS  
   - **GCR** if using Google Cloud  
   - **ACR** if using Azure  
3. Pull the image on your VM.


*Warnning*: (This is the hard professional way, easier is to just copy the container directly to your VM via SCP command!!)

### **Option B (recommended) — Build Directly on the VM**
1. Upload your project files to the VM (git clone, SCP, etc.).
2. Build the Docker image directly on the machine.

---

## **4. Launch Your Cloud VM**
We *recommend AWS EC2* as shown during the lectures, but **GCP Compute Engine** or **Azure VMs** work equally well.

Typical setup:
- Ubuntu Linux image
- CPU instance (unless you need GPU acceleration)
- SSH access enabled
- Assign a public IP address

---

## **5. Install & Configure the Environment**
- Update the system.
- Install Docker.
- (Optional) Install Python if needed for on-VM builds.
- Authenticate with your cloud's container registry if pulling images.

---

## **6. Run Your Docker Container**
- Start the container with necessary ports exposed.
- Confirm the model server runs correctly.
- Test inference locally on the VM.

---

## **7. Configure Firewall & Networking**
Across AWS/GCP/Azure, you must:
- Open the port your service listens on (e.g., 8000/8080).
- Adjust security groups / firewall rules / network rules.
- (Recommended) Restrict access to your IP for safety.

---

## **8. Test External Access**
- Run your local `requests` script against your VM’s public endpoint.
- Confirm that the response matches expectations.

---

## **Acceptance Criteria**

Your GitHub repository should include:
- The service code
- If used, the Dockerfile
- A Python script using only `requests` package

Example:

```python
import requests

resp = requests.post(
    "http://<your-public-ip>:<port>/infer",
    json={"input": [1, 2, 3]}  # adapt to your model format
)
print(resp.json())
```
