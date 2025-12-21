# 🍎 Yummy AI: Intelligent Food & Fruit Analysis

**Yummy AI** is a comprehensive computer vision system designed to analyze food and fruit images. It combines multiple deep learning models to perform binary classification (Food vs. Fruit), fruit subcategory classification, few-shot food recognition, calorie estimation, and image segmentation.  
**Link**:https://ai-calories-calculator-zrqrwcwv8ggdogx5shjdjs.streamlit.app/
## 🚀 Key Features

*   **Binary Classification:** Distinguishes between "Food" and "Fruit" images.
*   **Fruit Recognition:** Identifies specific fruit types (e.g., Apple, Banana) using the OZnet classifier.
*   **Food Recognition:** Uses a One-Few Shot Learning approach (SwinProtoNet) to recognize food dishes with minimal training examples.
*   **Calorie Estimation:** Automatically calculates total calories based on classification consistency and estimated weight (from image metadata or assumptions).
*   **Segmentation:**
    *   **Binary Segmentation:** Isolates the fruit/food object from the background.
    *   **Multiclass Segmentation:** Segments different parts or types within the image.
*   **Siamese Matching:** Finds the most visually similar image from a dataset using similarity matching (Siamese Network).

## 🛠️ Technology Stack

*   **Language:** Python 3.8+
*   **Frameworks:** PyTorch, Streamlit
*   **Libraries:** OpenCV, PIL (Pillow), NumPy
*   **Models:** Custom CNNs, Swin Transformer, ProtoNet

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/omarelnokrashy/AI-Calories-calculator.git
    cd AI-Calories-calculator
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    Since `requirements.txt` is not provided, manually install the core libraries:
    ```bash
    pip install streamlit torch torchvision opencv-python pillow numpy
    ```

## 💻 Usage

### 1. Web Application (GUI)
Launch the interactive web interface to test single images or run similarity checks.
```bash
streamlit run project.py
```
*   **Single Image Analysis:** Upload an image to detect its class, subcategory, and calories, and view segmentation masks.
*   **Batch Integrated Test:** Point to a folder of images to process them all at once.
*   **Similarity Matching:** Find the best match for an "anchor" image within a dataset.

### 2. Batch Processing Script (CLI)
Run the automated test pipeline to process all images in the `Test Cases Structure` directory.
```bash
python main_script.py
```
Results will be saved in `test_results/Final_Integrated_Run`.

## 📂 Project Structure

```
├── project.py                 # Main Streamlit application
├── main_script.py             # CLI script for batch processing
├── Data/                      # Dataset handling
├── FoodFruitClassification/   # Binary classifier models
├── FruitClassification/       # OZnet fruit classifier models
├── One-Few shot model/        # Few-shot learning models
├── Fruit binary Segmentation/ # Binary selection models
├── Fruit Multiclass Segmentation/ # Multiclass segmentation models
├── TestModules/               # Logic wrappers for each model
└── test_results/              # Output directory for test runs
```

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.


