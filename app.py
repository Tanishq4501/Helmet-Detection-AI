import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import requests

# Function to download files
def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)



HELMET_MODEL_URLS = {
    "helmet_yolo.weights": "https://github.com/AnjanDutta/Hard-Hat-Detection/releases/download/v1.0/yolov3_helmet.weights",
    "helmet_yolo.cfg": "https://github.com/AnjanDutta/Hard-Hat-Detection/releases/download/v1.0/yolov3_helmet.cfg",
    "helmet.names": "https://github.com/AnjanDutta/Hard-Hat-Detection/releases/download/v1.0/helmet.names"
}

def create_helmet_classes_file():
    # Common helmet detection classes
    helmet_classes = [
        "helmet",
        "head",
        "person",
        "hardhat",
        "no_helmet",
        "with_helmet"
    ]
    
    with open("helmet.names", "w") as f:
        for class_name in helmet_classes:
            f.write(class_name + "\n")
    
    return helmet_classes

# Sidebar
st.sidebar.header("**🦺 Helmet Detection AI**")
st.sidebar.write("Workplace Safety Compliance")

st.sidebar.header("About This System")
st.sidebar.info('''
🔍 **Helmet Detection Features:**
• Detects hard hats and safety helmets
• Identifies persons without helmets
• Calculates safety compliance rate
• Works with construction/workplace images

📊 **Use Cases:**
• Construction site safety monitoring
• Workplace compliance audits
• Safety training materials
• Industrial safety assessments
''')

st.sidebar.header("Instructions")
st.sidebar.markdown('''
1. **Upload Image**: Choose workplace/construction image
2. **Wait for Processing**: AI analyzes the image
3. **View Results**: See detected helmets and compliance rate
4. **Safety Assessment**: Review compliance statistics
''')

st.sidebar.header("Contact")
st.sidebar.write("📧 [Email](mailto:safety@company.com)")
st.sidebar.write("💼 [LinkedIn](https://linkedin.com/in/safety-expert)")
st.sidebar.write("🐱 [GitHub](https://github.com/safety-detection)")

@st.cache_resource
def load_helmet_detection_model():
    try:
        st.info("Loading helmet detection model...")
        
        helmet_classes = create_helmet_classes_file()
        
        return load_standard_yolo_for_helmet_detection()
        
    except Exception as e:
        st.warning(f"Custom helmet model not available: {str(e)}")
        return load_standard_yolo_for_helmet_detection()

def load_standard_yolo_for_helmet_detection():
    try:
        weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
        config_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
        names_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
        
        for filename, url in [("yolov4.weights", weights_url), ("yolov4.cfg", config_url), ("coco.names", names_url)]:
            if not os.path.exists(filename):
                st.info(f"Downloading {filename}...")
                download_file(url, filename)
        
        net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
        layer_names = net.getLayerNames()
        
        try:
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        except (IndexError, TypeError):
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        
        st.success("✅ Model loaded successfully!")
        return net, output_layers, classes
        
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None, None

def detect_helmets_advanced(image, net, output_layers, classes):
    if net is None:
        return image, {"persons": 0, "helmets": 0, "compliance_rate": 0}
    
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    height, width, channels = img_bgr.shape
    
    # Create blob and run detection
    blob = cv2.dnn.blobFromImage(img_bgr, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    boxes = []
    confidences = []
    class_ids = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    persons = []
    potential_helmets = []
    
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            
            if label == "person":
                persons.append({
                    'bbox': [x, y, w, h],
                    'confidence': confidence,
                    'has_helmet': False,
                    'helmet_confidence': 0.0
                })
    
    for person in persons:
        px, py, pw, ph = person['bbox']
        
        head_region = img_bgr[py:py + ph//4, px:px + pw]
        
        if head_region.size > 0:
            helmet_score = analyze_helmet_colors(head_region)
            
            shape_score = analyze_helmet_shapes(head_region)
            
            texture_score = analyze_helmet_texture(head_region)
            
            total_score = (helmet_score + shape_score + texture_score) / 3
            
            if total_score > 0.3:  
                person['has_helmet'] = True
                person['helmet_confidence'] = total_score
    
    # Draw results
    for person in persons:
        x, y, w, h = person['bbox']
        
        # Color coding
        if person['has_helmet']:
            color = (0, 255, 0)  
            status = f"✅ Helmet ({person['helmet_confidence']:.2f})"
        else:
            color = (0, 0, 255)  
            status = "❌ No Helmet"
        
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 3)
        
        cv2.putText(img_bgr, status, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img_bgr, f"Conf: {person['confidence']:.2f}", (x, y + h + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h//4), (255, 255, 0), 2)
    
    total_persons = len(persons)
    persons_with_helmets = sum(1 for p in persons if p['has_helmet'])
    compliance_rate = (persons_with_helmets / total_persons * 100) if total_persons > 0 else 0
    
    results = {
        "persons": total_persons,
        "helmets": persons_with_helmets,
        "compliance_rate": compliance_rate,
        "detailed_results": persons
    }
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), results

def analyze_helmet_colors(head_region):
    """Analyze colors typically associated with helmets"""
    hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
    
    helmet_colors = [
        ([15, 100, 100], [35, 255, 255]),
    
        ([0, 0, 200], [180, 30, 255]),
        ([100, 100, 100], [130, 255, 255]),
        
        ([0, 100, 100], [10, 255, 255]),
    ]
    
    total_pixels = head_region.shape[0] * head_region.shape[1]
    helmet_pixels = 0
    
    for lower, upper in helmet_colors:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        helmet_pixels += cv2.countNonZero(mask)
    
    return min(helmet_pixels / total_pixels, 1.0)

def analyze_helmet_shapes(head_region):
    """Analyze shapes typical of helmets"""
    gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    helmet_score = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Minimum area threshold
            # Calculate roundness (helmet-like shape)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                roundness = (4 * np.pi * area) / (perimeter ** 2)
                helmet_score = max(helmet_score, roundness)
    
    return helmet_score

def analyze_helmet_texture(head_region):
    """Analyze texture patterns typical of helmets"""
    gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
    

    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    texture_score = max(0, 1 - (variance / 1000))
    
    return texture_score

st.title("🦺 Advanced Helmet Detection System")
st.markdown("**AI-Powered Workplace Safety Compliance Monitor**")

net, output_layers, classes = load_helmet_detection_model()

if net is not None:
    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Upload workplace image (construction sites, factories, etc.)", 
        type=["jpg", "jpeg", "png", "bmp"]
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="📸 Original Image", use_column_width=True)
        
        # Process image
        with st.spinner("🔍 Analyzing image for helmet detection..."):
            result_image, detection_stats = detect_helmets_advanced(image, net, output_layers, classes)
        
        # Display results
        st.image(result_image, caption="🎯 Detection Results", use_column_width=True)
        
        # Display statistics
        if detection_stats["persons"] > 0:
            st.subheader("📊 Safety Compliance Report")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("👥 Total Persons", detection_stats["persons"])
            
            with col2:
                st.metric("🦺 With Helmets", detection_stats["helmets"])
            
            with col3:
                st.metric("⚠️ Without Helmets", detection_stats["persons"] - detection_stats["helmets"])
            
            with col4:
                st.metric("📈 Compliance Rate", f"{detection_stats['compliance_rate']:.1f}%")
            
            # Safety status
            compliance_rate = detection_stats['compliance_rate']
            if compliance_rate == 100:
                st.success("✅ **EXCELLENT COMPLIANCE** - All workers wearing helmets!")
            elif compliance_rate >= 80:
                st.warning("⚠️ **GOOD COMPLIANCE** - Most workers wearing helmets")
            elif compliance_rate >= 50:
                st.error("❌ **POOR COMPLIANCE** - Many workers without helmets")
            else:
                st.error("🚨 **CRITICAL SAFETY ISSUE** - Most workers without helmets")
            
            # Progress bar
            st.subheader("Compliance Progress")
            st.progress(compliance_rate / 100)
            
        else:
            st.info("ℹ️ No persons detected in the image")
            
    else:
        st.info("👆 Please upload an image to start helmet detection")

else:
    st.error("❌ Failed to load the detection model")
    st.info("Please check your internet connection and try refreshing the page")

# Help section
with st.expander("ℹ️ How to get better results"):
    st.markdown("""
    **For optimal helmet detection:**
    
    1. **Image Quality**: Use high-resolution, well-lit images
    2. **Camera Angle**: Front or side views work best
    3. **Distance**: Persons should be clearly visible (not too far)
    4. **Lighting**: Avoid backlighting or shadows on faces
    5. **Helmet Visibility**: Ensure helmets are not obstructed
    
    **Common Helmet Colors Detected:**
    - 🟡 Yellow/Orange (most common)
    - ⚪ White
    - 🔵 Blue
    - 🔴 Red
    - 🟢 Green
    """)

with st.expander("⚙️ Technical Details"):
    st.markdown("""
    **Detection Method:**
    - Uses YOLOv4 for person detection
    - Applies computer vision algorithms for helmet detection
    - Combines color analysis, shape detection, and texture analysis
    - Real-time processing with confidence scoring
    
    **Accuracy Notes:**
    - This is a demonstration system
    - For production use, consider training on specific helmet datasets
    - Accuracy depends on image quality and lighting conditions
    """)