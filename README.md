Object Detection Application
An interactive object detection application built with Python, leveraging Streamlit for the frontend and PyTorch's Faster R-CNN for detecting objects in images. This application uses a pre-trained Faster R-CNN model trained on the COCO dataset to predict object categories and confidence scores, displaying them with bounding boxes.

Features
Upload and Detect: Upload images in .jpg, .jpeg, or .png formats to detect objects.
Bounding Boxes: Displays objects with yellow bounding boxes.
Confidence Scores: Each bounding box includes the object's confidence score in percentages.
Highlighting: Black text with a white background ensures legible object labels.
Streamlit Interface: Provides a user-friendly and interactive web app interface.
Real-time Predictions: Uses the Faster R-CNN model for fast and accurate predictions.
Screenshots
Main Interface:![Screenshot 2024-12-04 012500](https://github.com/user-attachments/assets/cc6fa34c-3038-498e-834d-3b13f254224f)


Prerequisites
Ensure you have the following installed on your system:

Python 3.8+
pip (Python package manager)
Installation
Clone this repository:

bash
Copy code
git clone https://github.com/Ajayvinayak/object-detection-app.git
cd object-detection-app
Create and activate a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Dependencies
The requirements.txt file contains all necessary libraries:

plaintext
Copy code
torch==2.x  # PyTorch library
torchvision==0.x  # TorchVision for Faster R-CNN
streamlit==1.x  # Streamlit for UI
numpy
Pillow
matplotlib
opencv-python
Running the Application
Start the Streamlit app:

bash
Copy code
streamlit run app.py
Open your browser and go to the local Streamlit URL (typically http://localhost:8501).

Upload an image and see the detected objects with bounding boxes and confidence scores.

Project Structure
plaintext
Copy code
object-detection-app/
├── app.py                # Main application code
├── requirements.txt      # Dependencies
├── README.md             # Documentation
├── example_screenshot.png # Screenshot of the app
How It Works
Upload Image: Users upload an image via the Streamlit interface.
Model Processing: The uploaded image is passed to a Faster R-CNN model (pre-trained on the COCO dataset).
Bounding Boxes:
The model predicts objects and their locations (bounding boxes).
Labels include the object name and confidence score, highlighted with a white background and black text.
Output: The processed image is displayed with bounding boxes and labels.
Example Output
Uploaded Image
Input an image with objects (e.g., people, cars, animals).

Output
Objects detected are displayed with yellow bounding boxes.
Each bounding box includes a label with:
Object name (e.g., "Person").
Confidence score (e.g., "95%").
Text styled with black font on a white background.
Customization
Changing Background Color
To customize the background color of the app:

Open app.py.
Edit the .main class in the set_styles function.
Known Issues
Large images may take longer to process.
The app relies on a pre-trained model and may not detect non-COCO objects.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
PyTorch for the Faster R-CNN implementation.
Streamlit for the web application framework.
COCO Dataset for the pre-trained model.
Feel free to use and improve this project! If you encounter any issues or have suggestions, please create an issue or pull request. 😊

