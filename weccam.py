import os
import cv2
from google.cloud import vision_v1p3beta1 as vision
from google.cloud import translate_v2 as translate

# Setup Google Cloud authentication client key
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'client_key.json'

# Specify a font file for Japanese text
japanese_font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (50, 50, 200)
line_type = 2

def recognize_objects(image_content):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_content)
    response = client.label_detection(image=image)
    labels = response.label_annotations
    detected_labels = [(label.description.lower(), round(label.score, 2)) for label in labels]
    for label in labels:
        # if len(text.description) == 10:
        desc = label.description.lower()
        score = round(label.score, 2)
        print("label: ", desc, "  score: ", score)
    return detected_labels


def translate_text(text, target_language='ja'):
    translate_client = translate.Client()
    result = translate_client.translate(text, target_language=target_language)
    return result['translatedText']

def generate_human_like_description(detected_labels):
    objects = []
    for label, score in detected_labels:
        objects.append(label.capitalize())

    # Construct the human-like explanation
    if len(objects) == 0:
        explanation = "I don't see any recognizable objects."
    elif len(objects) == 1:
        explanation = f"I see a {objects[0]}."
    else:
        last_object = objects[-1]
        other_objects = ", ".join(objects[:-1])
        explanation = f"I see {other_objects} and {last_object}."
    # Translate explanation to Japanese
    translated_explanation = translate_text(explanation, target_language='ja')
    return translated_explanation

def main():
    print('---------- Start Recognition --------')

    cap = cv2.VideoCapture(0)  # Access the webcam (change to 1 for an external camera)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        return

    # Encode frame to JPEG format
    ret, jpeg = cv2.imencode('.jpg', frame)
    if not ret:
        print("Error: Failed to encode image.")
        return

    image_content = jpeg.tobytes()
    detected_labels = recognize_objects(image_content)

    # Generate human-like description in Japanese
    human_description_jp = generate_human_like_description(detected_labels)
    print("\n", human_description_jp,"\n")

    # Display the result image with the Japanese description
    
    cv2.imshow('Recognition Result', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()
    print('------------------ End -----------------')

if __name__ == '__main__':
    main()
