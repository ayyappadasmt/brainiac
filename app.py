import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import torch.nn as nn  


st.set_page_config(layout="wide")  # Enables wide mode permanently

# Define class labels
class_labels = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

def load_model():
    model = models.resnet50(pretrained=False)  
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_labels))  
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom CSS for full-width styling and animations
st.markdown("""
    <style>
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes glow {
            0% { box-shadow: 0 0 10px gold; }
            50% { box-shadow: 0 0 20px gold; }
            100% { box-shadow: 0 0 10px gold; }
        }
        .navbar {
            background-color: #2c3e50;
            overflow: hidden;
            padding: 10px;
            text-align: center;
            width: 100%;
        }
        .navbar a {
            color: white;
            padding: 14px 20px;
            text-decoration: none;
            font-size: 18px;
            display: inline-block;
        }
        .navbar a:hover {
            background-color: #1a252f;
        }
        .header {
            text-align: center;
            color: #2c3e50;
            width: 100%;
        }
        .button {
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            font-size: 18px;
            margin-top: 10px;
            width: 100%;
            animation: fadeIn 1s ease-in-out;
        }
        .button:hover {
            background-color: #2980b9;
        }
        .about-section {
            background-color: #FFD700;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            width: 100%;
            color: black;
            text-align: center;
            animation: fadeIn 1s ease-in-out;
        }
        .team-container {
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
            margin-top: 20px;
        }
        .team-member {
            text-align: center;
            width: 200px;
            padding: 10px;
            border-radius: 10px;
            background-color: white;
            transition: transform 0.3s ease-in-out;
        }
        .team-member:hover {
            transform: scale(1.1);
        }
        .team-member img {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            border: 3px solid #2c3e50;
        }
        .team-member h3 {
            margin: 10px 0 5px 0;
            color: #2c3e50;
        }
        .team-member p {
            font-size: 14px;
            color: black;
        }
        .contact-section {
            background-color: #FFD700;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            width: 100%;
            color: black;
            text-align: center;
            animation: fadeIn 1s ease-in-out;
        }
    </style>
""", unsafe_allow_html=True)

# Navbar
st.markdown("""
    <div class="navbar">
        <a href="#home">Home</a>
        <a href="#about">About Us</a>
        <a href="#contact">Contact Us</a>
    </div>
""", unsafe_allow_html=True)

# Main UI
st.markdown("<h1 class='header'>Brainiac - MRI Tumor Classification</h1>", unsafe_allow_html=True)
st.write("Upload an MRI image to classify it as Meningioma, Pituitary Tumor, or No Tumor.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI Image...", type=["jpg", "png", "jpeg"])

# Custom CSS for file uploader
st.markdown("""
    <style>
        .stFileUploader {
            border: 2px solid gold !important;
            border-radius: 10px;
            padding: 10px;
        }
        div[data-testid="stFileUploadDropzone"] {
            border: 2px solid gold !important;
            border-radius: 10px;
        }
        button[kind="secondary"] {
            background-color: gold !important;
            color: black !important;
            font-weight: bold;
        }
        button[kind="secondary"]:hover {
            background-color: #FFD700 !important;
        }
    </style>
""", unsafe_allow_html=True)


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Scan with AI"):
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()
        st.write(f"### Predicted Tumor Type: **{class_labels[prediction]}**")

# About Us Section
import streamlit as st

# Injecting CSS to style the team section
# Injecting CSS to style the team section
st.markdown("""
    <style>
        .about-section {
            text-align: center;
            padding: 40px;
            background-color: #FFD700; /* Yellow background */
        }
        .about-section h2 {
            font-size: 28px;
            margin-bottom: 10px;
        }
        .about-section p {
            font-size: 16px;
            color: black;
        }
        .team-container {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 20px;
            padding: 20px;
            background-color: #1E1E1E; /* Dark background for contrast */
            border-radius: 10px;
        }
        .team-member {
            text-align: center;
            color: white;
            background: #2C2C2C;
            padding: 15px;
            border-radius: 10px;
            width: 200px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: space-between;
            height: 250px; /* Ensures equal height for all cards */
        }
        .team-member img {
            border-radius: 50%;
            width: 100px;
            height: 100px;
        }
        .team-member h3 {
            margin: 10px 0 5px;
        }
        .team-member p {
            font-size: 14px;
            color: #ddd;
            min-height: 20px; /* Ensures alignment by maintaining equal height */
            display: flex;
            align-items: center;
            justify-content: center;
        }
            
        #ha{
            margin-top:34px;
        }
    </style>
""", unsafe_allow_html=True)


# Adding the "About Us" section with team members
st.markdown("""
    <div class="about-section" id="about">
        <h2>About Us</h2>
        <p>We are a team of passionate developers working on AI solutions for healthcare.</p>
    </div>
    <div class="team-container">
        <div class="team-member">
            <img src="https://media.licdn.com/dms/image/v2/D5603AQEgzmJJN2_XVA/profile-displayphoto-shrink_800_800/B56ZTazewbHsAk-/0/1738837718302?e=1747267200&v=beta&t=yhXU_rDrLAWYSGf1U0zjhEm6kOz1iC364Ba3kZ_-7pQ" alt="Developer 1">
            <h3> Vighnesh H.</h3>
            <p id="ha">AI Specialist</p>
        </div>
        <div class="team-member">
            <img src="https://media.licdn.com/dms/image/v2/D4E03AQH7aOdy-2IAww/profile-displayphoto-shrink_800_800/B4EZTa2fdPHcAg-/0/1738838506155?e=1747267200&v=beta&t=RobvW3fBGpauf8hZBD5ix7Vc7LU5F2sKoyf_pt6aUi0    " alt="Developer 2">
            <h3>Ayyappadas MT</h3>
            <p>AI Specialist</p>
        </div>
        <div class="team-member">
            <img src="https://media.licdn.com/dms/image/v2/D5603AQG86yMQv1AjPA/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1709066248618?e=1747267200&v=beta&t=HtMUqNcp15XMw7kBvN5qajyKNgqz8PVy5iWxnFtS3dc" alt="Developer 3">
            <h3>Yadhu Vipin M.</h3>
            <p>Web Developer</p>
        </div>
            
    </div>
""", unsafe_allow_html=True)


# Contact Us Section
st.markdown("""
    <div class="contact-section" id="contact">
        <h2>Contact Us</h2>
        <p><b>Email:</b> <a href="mailto:support@brainiac.com" style="color: black; text-decoration: none;">support@brainiac.com</a></p>
        <p><b>Phone:</b> <a href="tel:+919447540712" style="color: black; text-decoration: none;">+919447540712</a></p>
    </div>
""", unsafe_allow_html=True)
