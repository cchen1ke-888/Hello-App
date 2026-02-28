from transformers import pipeline
from PIL import Image
import streamlit as st

def main():
  st.header("Title: Age Classification using ViT")

  age_classifier = pipeline("image-classification",
                            model="MatanBT/age-vit-classifier")
  
  image_name = "middleagedMan.jpg"
  image_name = Image.open(image_name).convert("RGB")

  age_predictions = age_classifier(image_name)
  st.write(age_predictions)
  age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)

  st.write("Predicted Age Range:")
  st.write(f"Age range: {age_predictions[0]['label']}")

  st.write("Done")   # 'st.write' in Web Application= 'print' in Python

if __name__ == "__main__":
    main()
