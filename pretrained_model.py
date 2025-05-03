import torch
import mediapipe as mp
import numpy as np
import cv2
import json
import requests
import os
from pathlib import Path

class ISLPretrainedModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_map = None
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def download_model(self):
        """Download the pre-trained model and label map from AI4Bharat's INCLUDE project"""
        model_dir = Path("pretrained_models")
        model_dir.mkdir(exist_ok=True)
        
        # URLs for the model and label map
        model_url = "https://github.com/AI4Bharat/INCLUDE/raw/master/models/transformer_include50.pth"
        label_map_url = "https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master/label_maps/include50_label_map.json"
        
        # Download model
        model_path = model_dir / "transformer_include50.pth"
        if not model_path.exists():
            print("Downloading pre-trained model...")
            response = requests.get(model_url)
            with open(model_path, 'wb') as f:
                f.write(response.content)
        
        # Download label map
        label_map_path = model_dir / "include50_label_map.json"
        if not label_map_path.exists():
            print("Downloading label map...")
            response = requests.get(label_map_url)
            with open(label_map_path, 'w') as f:
                f.write(response.text)
                
        return model_path, label_map_path
    
    def load_model(self):
        """Load the pre-trained model and label map"""
        model_path, label_map_path = self.download_model()
        
        # Load the model
        self.model = torch.jit.load(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load the label map
        with open(label_map_path) as f:
            self.label_map = json.load(f)
    
    def preprocess_frame(self, frame):
        """Process frame and extract hand landmarks using MediaPipe"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
            
        # Extract landmarks and convert to model input format
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            hand_points = []
            for landmark in hand_landmarks.landmark:
                hand_points.extend([landmark.x, landmark.y, landmark.z])
            landmarks.extend(hand_points)
            
        return torch.tensor(landmarks).float().unsqueeze(0).to(self.device)
    
    def predict(self, frame):
        """Predict the sign from the input frame"""
        if self.model is None:
            self.load_model()
            
        # Preprocess the frame
        inputs = self.preprocess_frame(frame)
        if inputs is None:
            return None
            
        # Get prediction
        with torch.no_grad():
            outputs = self.model(inputs)
            prediction = torch.argmax(outputs, dim=1).item()
            
        # Convert prediction to label
        return self.label_map[str(prediction)] 