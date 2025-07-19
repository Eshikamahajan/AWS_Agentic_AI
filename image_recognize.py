
import pandas as pd
import boto3
import cv2
from botocore.exceptions import ClientError

credential = pd.read_csv("aws-event/aws-event_accessKeys.csv")
access_key_id = credential['Access key ID'][0]
secret_access_key = credential['Secret access key'][0]

rekognition = boto3.client('rekognition', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key,  region_name="us-east-1")

image_bytes= ""
detected_text = []
detected_label = []

def load_image(image_path = "aws-event/assets/2.jpg"):
    # Load image bytes
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()
    return image_bytes

def get_detected_text(image_bytes):
    # ---- 1. Detect Text ----
    text_response = rekognition.detect_text(Image={"Bytes": image_bytes})
    # print("=== Detected Text ===")
    for item in text_response["TextDetections"]:
        if item["Type"] == "LINE":
            # print(f"Text: {item['DetectedText']} (Confidence: {item['Confidence']:.2f}%)")
            detected_text.append((item['DetectedText'], str(item['Confidence'])))
    return detected_text

def get_detected_label(image_bytes):
    # ---- 2. Detect Labels ----
    label_response = rekognition.detect_labels(Image={"Bytes": image_bytes}, MaxLabels=10)
    # print("\n=== Detected Labels ===")
    for label in label_response["Labels"]:
        # print(f"Label: {label['Name']} (Confidence: {label['Confidence']:.2f}%)")
        detected_label.append((label['Name'], str(label['Confidence'])))
    return detected_label

if __name__ == "__main__":
    image_bytes = load_image()
    detect_text(image_bytes)
    detect_label(image_bytes)
    print("Image processing completed.")
    print("--------------------------Detected Texts--------------------------")
    print(detected_text)
    print("\n")
    print("--------------------------Detected Labels--------------------------")
    print(detected_label)
    print("\n")