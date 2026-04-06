# Vast.ai Server Execution Plan 🚀
## Project: DermNet Skin Disease Classifier (23 Classes)

### 1. The Strategy
Train a robust, high-accuracy Convolutional Neural Network (CNN) on the full 15,557-image **DermNet** dataset to classify 23 extremely similar skin diseases, and then deploy it to the internet with a GUI for a live university presentation.

### 2. The Architecture: Hybrid Vision Transformer (ViT) + ConvNeXt
Instead of utilizing standard aged 2021-era CNNs, we are deploying the absolute **2025/2026 State-of-the-Art (SOTA)** Hybrid architecture:
*   **Brain 1 (Global Context):** `ViT` (Vision Transformer from Hugging Face). Uses self-attention to see the "big picture" of the skin disease.
*   **Brain 2 (Local Texture):** `ConvNeXtBase` (Keras Native). Extracts macro skin textures to find localized anomalies.

The `src/hybrid_transformer_model.py` script perfectly merges these two brains. It leverages an advanced `Adamax` optimizer, heavy L2 Regularization, severe Dropout (50%), and automatically balances the massive 23-class imbalance using calculated class weights.

### 3. Server Hardware (Vast.ai)
Because an Ensemble network requires high VRAM to rapidly load augmented images in dense batches:
*   **Recommended GPU:** NVIDIA RTX 4070 Ti (or RTX 3090, 4090, 5060 Ti). 
*   **VRAM Required:** At least 12 GB.
*   **Estimated Training Time:** 3 to 5 hours.
*   **Cost:** ~$0.08 to $0.15 per hour.

### 4. Step-by-Step Execution on the Server
1. Rent the Ubuntu/Python server on vast.ai.
2. Transfer the `skin_disease_classifier` project folder onto the remote server.
3. Install python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the bleeding-edge Transformer training script mathematically mapped to all 23 classes:
   ```bash
   python3 src/hybrid_transformer_model.py
   ```
5. Wait ~4 hours. The script will automatically output a file named `final_hybrid_transformer.keras` containing the network's knowledge.

### 5. Final Presentation Deployment (Hugging Face Spaces)
For the final presentation, the AI needs to be "live" so the professors can test it without coding tools.
1. We will NOT use Kaggle or ZeroGPU (too many limitations and short 3-minute timeouts).
2. We will upload `app.py` (our Streamlit GUI) and the finished `final_ensemble_model.keras` brain to **Hugging Face Spaces**.
3. We will select their **Free Standard CPU Tier** (2 vCPU, 16GB RAM) because inference only takes milliseconds.
4. Hugging Face will host our interface 24/7 for free exactly as required by the syllabus, allowing anyone to upload a disease photo and get an instant prediction from our trained Ensemble model.
