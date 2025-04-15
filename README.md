# Emotion-Based-Music-Player-using-a-combination-of-OpenCV-and-Raspberry-Pi

# Emotion-Based Music Player 🎵😊

This project is an innovative Emotion-Based Music Player developed using a **Raspberry Pi**. It captures the user's image, detects their emotion, and plays a song that matches the mood.

## 📌 Features

- Detects human emotion using facial expressions
- Plays music based on detected emotion
- Works offline with songs stored on a memory card
- Simple and efficient interface
- Built using Python and OpenCV

## 🧠 How It Works

1. The Raspberry Pi camera captures an image of the user.
2. The system detects the user's emotion using machine learning.
3. Based on the emotion (e.g., happy, sad, angry, neutral), a relevant song is selected.
4. The selected song is played through the speaker.

## 📂 Folder Structure

The project uses a folder named `static` which contains trained models and important assets.  
**This folder is not included in the repository due to size limits.**  
You can download it separately from the link below:

🔗 [Download the `static` folder from Google Drive](PASTE_YOUR_DRIVE_LINK_HERE)

After downloading, place the `tratic` folder inside the root directory of the project.

## 💻 Requirements

- Raspberry Pi (any model with camera support)
- Camera Module
- Speaker
- Python 3.x

### Python Libraries Required:
- opencv-python
- numpy
- pygame
- keras / tensorflow (if you're using deep learning model)
