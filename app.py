import gradio as gr
import os
from src.components.inference_pipeline import ModelInference
from PIL import Image


cwd = os.getcwd()
model_path = r'best_model\best_model.pth'


class_name = {0:'Cat', 1:'Dog', 2:'person'}
classifier = ModelInference(model_path)

def classify_image(image):
    image_path = "outputs/uploaded_image.jpg"
    image.save(image_path)
    
    label, output_path = classifier.predict(image_path)
    
    return label, Image.open(output_path)
    
    
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type='pil'),
    outputs=[gr.Textbox(label='Prediction'),gr.Image(label='labeled image')],
    title = "image classification gradio app",
    description = 'upload an image to clssify it as dog, cat or preson'
)

if __name__ == "__main__":
    demo.launch()