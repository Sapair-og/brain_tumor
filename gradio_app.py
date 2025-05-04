import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from datetime import datetime

# Import tensorflow first to avoid import issues
import tensorflow as tf
from tensorflow.keras.models import load_model  #type:ignore
from tensorflow.keras.preprocessing.image import img_to_array  #type:ignore

# Configure matplotlib to use Agg backend for non-GUI environments
import matplotlib
matplotlib.use('Agg')

# Set environment variable to avoid Gradio analytics issues
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Import gradio
try:
    import gradio as gr
except ImportError as e:
    print(f"Error importing Gradio: {e}")
    print("Try reinstalling Gradio with: pip uninstall -y gradio gradio-client && pip install gradio --no-cache-dir")
    exit(1)

# Define classes for brain tumor classification
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Path to model
MODEL_PATH = r"C:\Users\Yashvardhan Singh\Desktop\AI and Machine learning\PROJECTS\brain_tumor_project\checkpoints\brain_tumor_detection_final.keras"

# Load model
try:
    model = load_model(MODEL_PATH)
    model_loaded = True
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    model_loaded = False
    print(f"Error loading model: {e}")

# Class descriptions for information
class_descriptions = {
    "glioma": "Gliomas are tumors that occur in the brain and spinal cord. They begin in glial cells that surround and support nerve cells. Gliomas can be low-grade (slow-growing) or high-grade (fast-growing).",
    "meningioma": "Meningiomas arise from the meninges, the membranes that surround your brain and spinal cord. Most meningiomas are noncancerous (benign) and grow slowly. Many meningiomas can be monitored without treatment.",
    "notumor": "No tumor detected in the brain MRI scan. The scan appears normal with no visible signs of tumor growth or abnormal tissue structures.",
    "pituitary": "Pituitary tumors are abnormal growths that develop in the pituitary gland at the base of the brain. Most pituitary tumors are benign (not cancer) and don't spread to other parts of the body."
}

# Clinical information for each tumor type
clinical_information = {
    "glioma": {
        "common_symptoms": ["Headaches", "Seizures", "Difficulty thinking or speaking", "Behavioral changes", "Progressive weakness or paralysis"],
        "treatment_options": ["Surgery", "Radiation therapy", "Chemotherapy", "Targeted drug therapy"],
        "prognosis": "Varies widely depending on tumor grade, location, and genetic factors. Low-grade gliomas may be slow growing, while high-grade gliomas (like glioblastoma) are aggressive.",
        "follow_up": "Regular MRI scans every 2-3 months initially, then at increasing intervals if stable."
    },
    "meningioma": {
        "common_symptoms": ["Headaches", "Hearing loss", "Vision problems", "Memory problems", "Seizures"],
        "treatment_options": ["Observation for small, slow-growing tumors", "Surgery", "Radiation therapy for inoperable tumors"],
        "prognosis": "Generally favorable for benign meningiomas with complete surgical removal. Recurrence is possible, particularly for incompletely removed tumors.",
        "follow_up": "MRI scans at 3 months post-treatment, then annually for several years."
    },
    "notumor": {
        "common_symptoms": ["N/A - No tumor present"],
        "treatment_options": ["N/A - No tumor present"],
        "prognosis": "Excellent - no tumor detected",
        "follow_up": "Standard medical care as needed for original symptoms"
    },
    "pituitary": {
        "common_symptoms": ["Headaches", "Vision changes", "Fatigue", "Hormonal imbalances", "Unexplained weight changes"],
        "treatment_options": ["Medication to control hormone production", "Surgery", "Radiation therapy"],
        "prognosis": "Generally favorable with appropriate treatment. Function of the pituitary gland may be affected.",
        "follow_up": "Regular blood tests to monitor hormone levels, follow-up MRI scans."
    }
}

# Sample images for display
sample_images = {
    "glioma": r"brain_tumor_project\brain_tumor_data\glioma\Te-gl_0026.jpg",
    "meningioma": r"brain_tumor_project\brain_tumor_data\meningioma\Te-me_0027.jpg",
    "notumor": r"brain_tumor_project\brain_tumor_data\notumor\Te-no_0017.jpg",
    "pituitary": r"brain_tumor_project\brain_tumor_data\pituitary\Te-pi_0018.jpg"
}

def preprocess_image(img):
    """Preprocess the image for model prediction"""
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_brain_tumor(img, patient_name, patient_age, patient_gender, patient_id, clinical_notes):
    """
    Predict the type of brain tumor from an MRI image and generate a clinical report
    
    Args:
        img: The uploaded MRI image
        patient_name: Name of the patient
        patient_age: Age of the patient
        patient_gender: Gender of the patient
        patient_id: Patient ID number
        clinical_notes: Additional clinical notes
        
    Returns:
        Tuple containing various output components for the Gradio interface
    """
    if not model_loaded:
        error_html = """
        <div style="border: 2px solid red; padding: 20px; border-radius: 5px; background-color: #fff5f5;">
            <h3 style="color: red;">❌ ERROR: Model not loaded</h3>
            <p>Please ensure the model file exists at: {MODEL_PATH}</p>
        </div>
        """
        return img, error_html, "", "", {class_name: 0 for class_name in CLASSES}, None
    
    # Progress simulation for a more "medical analysis" feel
    time.sleep(1)
    
    try:
        # Preprocess image
        img_array = preprocess_image(img)
        
        # Make prediction
        prediction = model.predict(img_array)[0]
        predicted_class = CLASSES[np.argmax(prediction)]
        confidence = prediction[np.argmax(prediction)] * 100
        
        # Format confidence scores
        confidence_scores = {CLASSES[i]: float(prediction[i]) for i in range(len(CLASSES))}
        
        # Create annotated image using matplotlib
        plt.figure(figsize=(10, 8))
        plt.imshow(np.array(img), cmap='gray')
        
        # Add border color based on prediction
        if predicted_class == "notumor":
            border_color = 'green'
            border_width = 5
        else:
            border_color = 'red'
            border_width = 5
            
        # Add border
        plt.gca().spines['top'].set_color(border_color)
        plt.gca().spines['bottom'].set_color(border_color)
        plt.gca().spines['left'].set_color(border_color)
        plt.gca().spines['right'].set_color(border_color)
        plt.gca().spines['top'].set_linewidth(border_width)
        plt.gca().spines['bottom'].set_linewidth(border_width)
        plt.gca().spines['left'].set_linewidth(border_width)
        plt.gca().spines['right'].set_linewidth(border_width)
        
        # Add prediction text at top of image
        plt.title(f"Prediction: {predicted_class.upper()}", fontsize=18, color=border_color, weight='bold', pad=20)
        
        plt.axis('off')
        fig = plt.gcf()
        fig.tight_layout()
        
        # Convert figure to image
        fig.canvas.draw()
        annotated_img = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close()
        
        # Format patient information
        patient_name = patient_name if patient_name else "Unknown"
        patient_age = patient_age if patient_age else "Unknown"
        patient_gender = patient_gender if patient_gender else "Unknown"
        patient_id = patient_id if patient_id else "Unknown"
        clinical_notes = clinical_notes if clinical_notes else "No clinical notes provided."
        
        # Get current date and time
        current_date = datetime.now().strftime("%B %d, %Y")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Determine confidence level text
        if confidence >= 90:
            confidence_level = "Very High"
        elif confidence >= 75:
            confidence_level = "High"
        elif confidence >= 60:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"
        
        # Create HTML for diagnosis label
        if predicted_class == "notumor":
            diagnosis_color = "green"
            diagnosis_icon = "✅"
            diagnosis_text = "NO TUMOR DETECTED"
        else:
            diagnosis_color = "#e74c3c"  # A softer red
            diagnosis_icon = "⚠️"
            diagnosis_text = f"{predicted_class.upper()} DETECTED"
        
        diagnosis_label = f"""
        <div style="background-color: {diagnosis_color}; color: white; text-align: center; padding: 15px; 
                 border-radius: 5px; font-weight: bold; font-size: 18px; margin-bottom: 10px;">
            {diagnosis_icon} {diagnosis_text}
        </div>
        <div style="text-align: center; font-size: 16px; color: #555;">
            Confidence: {confidence:.1f}% ({confidence_level})
        </div>
        """
        
        # Create medical report HTML
        # Get specific clinical information for the predicted class
        clinical_info = clinical_information[predicted_class]
        
        medical_report_html = f"""
        <div style="border: 1px solid #2c3e50; border-radius: 10px; padding: 20px; background-color: white;">
            <div style="display: flex; justify-content: space-between; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; margin-bottom: 15px;">
                <div>
                    <h2 style="margin: 0; color: #2c3e50;">Brain MRI Analysis Report</h2>
                    <p style="margin: 5px 0 0 0; color: #7f8c8d;">AI-Assisted Diagnosis</p>
                </div>
                <div style="text-align: right;">
                    <p style="margin: 0; font-weight: bold;">Date: {current_date}</p>
                    <p style="margin: 5px 0 0 0;">Time: {current_time}</p>
                </div>
            </div>
            
            <div style="margin-bottom: 20px;">
                <h3 style="color: #2c3e50; margin-bottom: 10px;">Patient Information</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 5px; border: 1px solid #ddd; width: 150px;"><strong>Name:</strong></td>
                        <td style="padding: 5px; border: 1px solid #ddd;">{patient_name}</td>
                        <td style="padding: 5px; border: 1px solid #ddd; width: 150px;"><strong>Patient ID:</strong></td>
                        <td style="padding: 5px; border: 1px solid #ddd;">{patient_id}</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px; border: 1px solid #ddd;"><strong>Age:</strong></td>
                        <td style="padding: 5px; border: 1px solid #ddd;">{patient_age}</td>
                        <td style="padding: 5px; border: 1px solid #ddd;"><strong>Gender:</strong></td>
                        <td style="padding: 5px; border: 1px solid #ddd;">{patient_gender}</td>
                    </tr>
                </table>
            </div>
            
            <div style="margin-bottom: 20px;">
                <h3 style="color: #2c3e50; margin-bottom: 10px;">Clinical Notes</h3>
                <p style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">{clinical_notes}</p>
            </div>
            
            <div style="margin-bottom: 20px; padding: 15px; border: 1px solid {diagnosis_color}; border-radius: 5px; background-color: {diagnosis_color}10;">
                <h3 style="color: {diagnosis_color}; margin-bottom: 10px;">Diagnosis</h3>
                <p style="font-weight: bold; font-size: 18px; margin-bottom: 10px;">{diagnosis_text}</p>
                <p style="margin-bottom: 5px;">Tumor Type: {predicted_class if predicted_class != "notumor" else "N/A"}</p>
                <p style="margin-bottom: 5px;">Confidence Level: {confidence:.1f}% ({confidence_level})</p>
                <p style="margin-bottom: 0px;">Description: {class_descriptions[predicted_class]}</p>
            </div>
            
            <div style="margin-bottom: 20px;">
                <h3 style="color: #2c3e50; margin-bottom: 10px;">Clinical Information</h3>
                
                <h4 style="margin-bottom: 5px; color: #555;">Common Symptoms</h4>
                <ul style="margin-top: 5px; margin-bottom: 15px;">
                    {''.join(f"<li>{symptom}</li>" for symptom in clinical_info['common_symptoms'])}
                </ul>
                
                <h4 style="margin-bottom: 5px; color: #555;">Treatment Options</h4>
                <ul style="margin-top: 5px; margin-bottom: 15px;">
                    {''.join(f"<li>{treatment}</li>" for treatment in clinical_info['treatment_options'])}
                </ul>
                
                <h4 style="margin-bottom: 5px; color: #555;">Prognosis</h4>
                <p style="margin-top: 5px; margin-bottom: 15px; padding-left: 15px;">{clinical_info['prognosis']}</p>
                
                <h4 style="margin-bottom: 5px; color: #555;">Follow-up Recommendations</h4>
                <p style="margin-top: 5px; margin-bottom: 0px; padding-left: 15px;">{clinical_info['follow_up']}</p>
            </div>
            
            <div style="margin-bottom: 20px;">
                <h3 style="color: #2c3e50; margin-bottom: 10px;">AI Analysis Information</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 5px; border: 1px solid #ddd; width: 150px;"><strong>Model:</strong></td>
                        <td style="padding: 5px; border: 1px solid #ddd;">ResNet50V2 Neural Network</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px; border: 1px solid #ddd;"><strong>Classes:</strong></td>
                        <td style="padding: 5px; border: 1px solid #ddd;">Glioma, Meningioma, No Tumor, Pituitary</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px; border: 1px solid #ddd;"><strong>Confidence Scores:</strong></td>
                        <td style="padding: 5px; border: 1px solid #ddd;">
                            Glioma: {confidence_scores['glioma']*100:.1f}%<br>
                            Meningioma: {confidence_scores['meningioma']*100:.1f}%<br>
                            No Tumor: {confidence_scores['notumor']*100:.1f}%<br>
                            Pituitary: {confidence_scores['pituitary']*100:.1f}%
                        </td>
                    </tr>
                </table>
            </div>
            
            <div style="font-size: 12px; color: #7f8c8d; margin-top: 30px; border-top: 1px solid #ddd; padding-top: 10px;">
                <p style="margin: 0;">This is an AI-assisted report and should be reviewed by a qualified radiologist or neurologist before making clinical decisions.</p>
                <p style="margin: 5px 0 0 0;">Generated using Brain Tumor Detection AI v1.0 | Model: ResNet50V2 | Confidence Score: {confidence:.1f}%</p>
            </div>
        </div>
        """
        
        # Create recommendations based on the predicted class
        if predicted_class == "notumor":
            recommendations = [
                "No evidence of tumor on this MRI scan",
                "Consider follow-up if clinical symptoms persist",
                "Consult with neurologist for further evaluation if needed"
            ]
        else:
            recommendations = [
                f"Findings consistent with {predicted_class}",
                "Correlation with clinical presentation recommended",
                "Consider follow-up imaging in 2-3 months",
                "Neurosurgical consultation recommended",
                "Additional diagnostic tests may be beneficial"
            ]
        
        recommendation_html = f"""
        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 15px;">
            <h4 style="margin-top: 0; margin-bottom: 10px; color: #2c3e50;">Clinical Recommendations</h4>
            <ul style="margin: 0; padding-left: 20px;">
                {''.join(f"<li style='margin-bottom: 5px;'>{recommendation}</li>" for recommendation in recommendations)}
            </ul>
        </div>
        """
        
        return annotated_img, medical_report_html, diagnosis_label, recommendation_html, confidence_scores, fig
        
    except Exception as e:
        error_msg = str(e)
        error_html = f"""
        <div style="border: 2px solid red; padding: 20px; border-radius: 5px; background-color: #fff5f5;">
            <h3 style="color: red;">❌ ERROR OCCURRED</h3>
            <p>{error_msg}</p>
        </div>
        """
        return img, error_html, "", "", {class_name: 0 for class_name in CLASSES}, None

def update_confidence_plot(confidences):
    """Create a horizontal bar chart of confidence scores"""
    fig, ax = plt.subplots(figsize=(8, 4))
    classes = list(confidences.keys())
    scores = [confidences[c] for c in classes]
    
    # Colors for different classes
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFBE0B']
    
    # Create horizontal bars
    bars = ax.barh(classes, scores, color=colors)
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(max(width + 0.01, 0.05), 
                bar.get_y() + bar.get_height()/2, 
                f'{width*100:.1f}%', 
                va='center')
    
    # Set labels and title
    ax.set_xlabel('Confidence Score')
    ax.set_title('Classification Confidence by Category')
    ax.set_xlim(0, 1)
    
    # Formatting
    plt.tight_layout()
    return fig

# Custom CSS for the application
custom_css = """
.container {
    max-width: 1200px;
    margin: 0 auto;
}
.header {
    background: linear-gradient(135deg, #4b0082, #9370db);
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    text-align: center;
}
.content {
    display: flex;
    gap: 20px;
}
.panel {
    background-color: white;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    padding: 20px;
    margin-bottom: 20px;
}
.footer {
    text-align: center;
    margin-top: 30px;
    padding: 10px;
    background-color: #f5f5f5;
    border-radius: 10px;
    font-size: 12px;
    color: #666;
}
"""

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="Brain Tumor MRI Analysis System", css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <div class="header">
            <h1>🧠 Brain Tumor MRI Analysis System</h1>
            <p>AI-Powered MRI Analysis for Brain Tumor Classification</p>
        </div>
        """)
        
        with gr.Tabs():
            with gr.TabItem("Clinical Dashboard"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Patient Information Panel
                        gr.HTML("""<div class="panel"><h3>Patient Information</h3>""")
                        patient_name = gr.Textbox(label="Patient Name", placeholder="Enter patient name")
                        patient_id = gr.Textbox(label="Patient ID", placeholder="Enter patient ID")
                        with gr.Row():
                            patient_age = gr.Textbox(label="Age", placeholder="Age")
                            patient_gender = gr.Dropdown(label="Gender", choices=["Male", "Female", "Other"], value=None)
                        clinical_notes = gr.Textbox(label="Clinical Notes", placeholder="Enter relevant clinical information...", lines=3)
                        
                        # Image Upload Panel
                        gr.Markdown("### MRI Image")
                        image_input = gr.Image(label="Upload Brain MRI", type="pil")
                        submit_btn = gr.Button("Analyze MRI", variant="primary", size="lg")
                        gr.HTML("""</div>""")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Results Panel
                        gr.HTML("""<div class="panel"><h3>Diagnosis Results</h3>""")
                        result_image = gr.Image(label="Analyzed MRI", type="numpy")
                        diagnosis_label = gr.HTML(label="Diagnosis")
                        confidence_output = gr.Label(label="Confidence Scores", visible=False)
                        confidence_plot = gr.Plot(label="Confidence Levels")
                        recommendation = gr.HTML(label="Recommendations")
                        gr.HTML("""</div>""")
                    
                    with gr.Column(scale=1):
                        # Medical Report Panel
                        gr.HTML("""<div class="panel"><h3>Medical Report</h3>""")
                        medical_report = gr.HTML()
                        gr.HTML("""</div>""")
            
            with gr.TabItem("Sample Images"):
                gr.Markdown("### Sample MRI Scans")
                with gr.Row():
                    # Create a display for each sample image if it exists
                    for class_name in CLASSES:
                        if os.path.exists(sample_images.get(class_name, "")):
                            with gr.Column():
                                gr.Image(value=sample_images.get(class_name), label=class_name.capitalize())
                                gr.Markdown(f"**{class_name.capitalize()}**: {class_descriptions[class_name][:100]}...")
                        else:
                            with gr.Column():
                                gr.Markdown(f"Sample for {class_name} not found at {sample_images.get(class_name, '')}")
            
            with gr.TabItem("About the System"):
                gr.HTML("""
                <div class="panel">
                    <h2>About this Brain Tumor Detection System</h2>
                    <p>This application uses a ResNet50V2-based deep learning model to analyze brain MRI scans for the presence and classification of tumors.</p>
                    
                    <h3>Model Information</h3>
                    <ul>
                        <li><strong>Architecture:</strong> ResNet50V2 (transfer learning)</li>
                        <li><strong>Training Dataset:</strong> Brain MRI images with tumor and non-tumor cases</li>
                        <li><strong>Classification:</strong> Multi-class (Glioma, Meningioma, No Tumor, Pituitary)</li>
                    </ul>
                    
                    <h3>Tumor Type Information</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px;">
                            <h4 style="color: #4b0082;">Glioma</h4>
                            <p>Gliomas are tumors that occur in the brain and spinal cord. They begin in glial cells that surround and support nerve cells. 
                            These tumors can be low-grade (slow growing) or high-grade (rapidly growing).</p>
                        </div>
                        <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px;">
                            <h4 style="color: #4b0082;">Meningioma</h4>
                            <p>Meningiomas arise from the meninges, the membranes that surround your brain and spinal cord. 
                            Most meningiomas are noncancerous (benign), though some can be malignant. They generally grow slowly.</p>
                        </div>
                        <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px;">
                            <h4 style="color: #4b0082;">Pituitary</h4>
                            <p>Pituitary tumors are abnormal growths that develop in the pituitary gland at the base of the brain. 
                            Most pituitary tumors are benign and don't spread beyond the pituitary gland.</p>
                        </div>
                        <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px;">
                            <h4 style="color: #4b0082;">No Tumor</h4>
                            <p>This classification indicates that no tumor was detected in the brain MRI scan. 
                            However, this doesn't rule out other neurological conditions that might not present as tumors.</p>
                        </div>
                    </div>
                    
                    <h3>Important Clinical Note</h3>
                    <div style="border: 1px solid #9370db; background-color: #f8f4ff; padding: 15px; border-radius: 5px; margin: 20px 0;">
                        <p style="margin: 0; font-weight: bold; color: #4b0082;">⚠️ Clinical Decision Support Only</p>
                        <p style="margin-top: 10px;">This system is designed as a decision support tool and not intended to replace clinical judgment. 
                        All results should be interpreted by qualified healthcare professionals in the context of the patient's clinical presentation, 
                        additional imaging, and medical history.</p>
                    </div>
                </div>
                """)
            
        gr.HTML("""
        <div class="footer">
            <p>© 2025 - Brain Tumor Detection Model - For Clinical Research & Educational Use Only</p>
            <p>Not FDA/CE approved for clinical use | Always consult with qualified healthcare professionals</p>
        </div>
        """)
        
        # Connect the interface components to the prediction function
        submit_btn.click(
            fn=predict_brain_tumor,
            inputs=[image_input, patient_name, patient_age, patient_gender, patient_id, clinical_notes],
            outputs=[result_image, medical_report, diagnosis_label, recommendation, confidence_output, confidence_plot]
        )
        
        # Connect the confidence output to update the plot
        confidence_output.change(
            fn=update_confidence_plot,
            inputs=confidence_output,
            outputs=confidence_plot
        )
    
    return demo

# Launch the app
if __name__ == "__main__":
    if not model_loaded:
        print(f"WARNING: Model could not be loaded from {MODEL_PATH}")
        print("The application will start but predictions will not work.")
        print("Please ensure the model file exists and is in the correct location.")
    
    try:
        demo = create_interface()
        demo.launch(share=True)
    except Exception as e:
        print(f"Error launching Gradio interface: {e}")
        print("If you're experiencing circular import issues, try reinstalling Gradio.")
        print("pip uninstall -y gradio gradio-client && pip install gradio --no-cache-dir")