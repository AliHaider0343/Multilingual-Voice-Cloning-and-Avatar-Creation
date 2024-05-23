import json
import time
from http.client import HTTPException
import torch
from TTS.api import TTS
from fastapi import FastAPI, UploadFile, File, Form,Response
from PIL import Image
from rembg import remove
from starlette.responses import StreamingResponse
from io import BytesIO
import io
import requests
from pydantic import BaseModel
import os
import tempfile
from tempfile import NamedTemporaryFile
import cv2
import numpy as np
import zipfile


device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)
#tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

DID_token="ZWxjaW5tdXNheWV2Nzc2NUBnbWFpbC5jb20:Hg-DdQFE014P-ULD_1pJ5"
background_img = Image.open("BK4.png")
temp_dir = tempfile.gettempdir()

app = FastAPI()
class Item(BaseModel):
    name: str
    user_data: str
    image_url: str

ALLOWED_EXTENSIONS = {'wav', 'mp3'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def remove_background(image):
  output = remove(image)
  return output

def add_padding(image, max_width=1000, padding_color=(0, 0, 0, 0)):
    # Calculate desired height based on 16:9 ratio
    desired_height = int(max_width * (4 / 8))

    # Resize image if necessary to fit within desired height
    width, height = image.size
    if height > desired_height:
        new_width = int(desired_height * (width / height))
        image = image.resize((new_width, desired_height), Image.Resampling.LANCZOS)
        width, height = image.size

    # Calculate padding to ensure output width is max_width
    left_padding = (max_width - width) // 2
    right_padding = max_width - width - left_padding

    # Calculate top padding to ensure output height is desired_height and remove bottom padding
    top_padding = (desired_height - height) - 0  # No bottom padding

    # Create a new image with padding
    new_image = Image.new(image.mode, (max_width, desired_height), padding_color)
    new_image.paste(image, (left_padding, top_padding))

    return new_image

def attachbackground(image):
  global background_img
  background_img = background_img.resize(image.size, Image.Resampling.LANCZOS)
  result = Image.alpha_composite(background_img.convert("RGBA"), image)
  return result

def GetAvatarID(name,user_data,image_url,speaking):
  if speaking:
    input="Hello, my name is Elchin. Welcome to Speak Up conversational application created especially for foreign languages teachers. We have checked several similar applications in the past, but none has been more impressive than Speak Up, we really went far above and beyond to make sure that our app is not only working but far more exceptional. Are you a foreign language tutor? Say hello to Speak Up - your personal tutoring AI assistant.",
  else:
    input="<break time=\"5000ms\"/><break time=\"5000ms\"/><break time=\"5000ms\"/><break time=\"5000ms\"/><break time=\"5000ms\"/><break time=\"5000ms\"/>"

  url = "https://api.d-id.com/talks"
  payload = {
     "script": {
        "type": "text",
        "input": str(input),
        "ssml": True,
        "provider": {
            "type": "microsoft",
            "voice_id": "en-US-JacobNeural"

        },
    },

      "config": {
          "stitch": True
      },
      "source_url": image_url,
      "crop_mode": "Wide",
      "user_data": user_data,
      "name": name
  }
  headers = {
      "accept": "application/json",
      "content-type": "application/json",
      "authorization": f"Basic {DID_token}"
  }

  response = requests.post(url, json=payload, headers=headers)
  response_json = json.loads(response.text)
  try:
      id=response_json["id"]
  except Exception as e:
      raise Exception(f"{response.text}")

  return id

def getAvatarResult(TaskID):
    url = f"https://api.d-id.com/talks/{TaskID}"
    headers = {
          "accept": "application/json",
          "content-type": "application/json",
          "authorization": f"Basic {DID_token}"
      }
    while True:
        # Make the API call
        response = requests.get(url, headers=headers)
        response_json = json.loads(response.text)

        # Extract all details
        details = {
            "name": response_json.get("name", ""),
            "user_data": response_json.get("user_data", ""),
            "status": response_json.get("status", ""),
            "result_url": response_json.get("result_url", "")
        }

        if details["status"] == "done":
            if details["result_url"]:
                return details
            else:
                return None
        else:
            time.sleep(3)

def overlay_logo_on_video(video_path, output_path):
    # Load video
    cap = cv2.VideoCapture(video_path)
    # Load image with transparency
    image = cv2.imread("mask.png", cv2.IMREAD_UNCHANGED)

    # Resize image to match video dimensions
    height, width, _ = image.shape
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width != frame_width or height != frame_height:
        image = cv2.resize(image, (frame_width, frame_height))

    # Split image into channels
    b, g, r, alpha = cv2.split(image)

    # Create mask from alpha channel
    mask = cv2.merge((alpha, alpha, alpha))

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 25.0, (frame_width, frame_height))

    # Overlay image on video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Blend image and video using mask
        overlay = cv2.bitwise_and(image[:, :, :3], mask)
        background = cv2.bitwise_and(frame, cv2.bitwise_not(mask))
        result = cv2.add(overlay, background)

        out.write(result)

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def process_audio(text, input_audio_path, processed_audio_path,language_tag="en"):
    # Perform text-to-speech conversion and save to processed_audio_path

    tts.tts_to_file(text, speaker_wav=input_audio_path, language=language_tag, file_path=processed_audio_path)

@app.post("/process_image/")
async def process_image(image: UploadFile = File(...)):
    try:
        # Read image file and load it into Pillow image variable
        img = Image.open(BytesIO(await image.read()))
        resultant_image = attachbackground(add_padding(remove_background(img)))
        img_byte_array = BytesIO()
        resultant_image.save(img_byte_array, format="PNG")  # Save as PNG regardless of the format
        img_byte_array.seek(0)

        # Return success message along with the image
        return StreamingResponse(io.BytesIO(img_byte_array.read()), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

def zip_and_return(speaker_video_data, listener_video_data):
    # Create a BytesIO object to store the zip file data
    zip_data = BytesIO()

    # Create a ZipFile object to write to the BytesIO object
    with zipfile.ZipFile(zip_data, mode='w') as zip_file:
        # Add speaker video data to the zip file with appropriate filename
        zip_file.writestr("processed_speaker_video.mp4", speaker_video_data)
        # Add listener video data to the zip file with appropriate filename
        zip_file.writestr("processed_listener_video.mp4", listener_video_data)

    # Set the pointer of the zip_data BytesIO object to the beginning
    zip_data.seek(0)

    # Return the zip file as a StreamingResponse
    return StreamingResponse(zip_data, media_type="application/zip",
                              headers={"Content-Disposition": "attachment;filename=processed_videos.zip"})

@app.post("/GetAvatar/")
async def GetAvatar(item: Item):
    name = item.name
    user_data = item.user_data
    image_url = item.image_url

    try:
        speakingAvatarID=GetAvatarID(name, user_data, image_url, True)
        ListeningAvatarID=GetAvatarID(name, user_data, image_url, False)
        speakingAvatarResults=getAvatarResult(speakingAvatarID)
        listeningAvatarResults=getAvatarResult(ListeningAvatarID)

        processed_speaker_video_path = NamedTemporaryFile(suffix='.mp4', delete=False).name
        processed_listener_video_path = NamedTemporaryFile(suffix='.mp4', delete=False).name

        overlay_logo_on_video(speakingAvatarResults['result_url'],processed_speaker_video_path)
        overlay_logo_on_video(listeningAvatarResults['result_url'],processed_listener_video_path)


        # Read processed audio file into memory
        with open(processed_speaker_video_path, "rb") as speaker_file:
            speaker_video_data = speaker_file.read()

        # Read processed audio file into memory
        with open(processed_listener_video_path, "rb") as listener_file:
            listener_video_data = listener_file.read()

        os.remove(processed_speaker_video_path)
        os.remove(processed_listener_video_path)
        zip_response = zip_and_return(speaker_video_data, listener_video_data)
        return zip_response

    except Exception as e:
        return {"status": "Error", "Description": str(e)}


@app.post("/clone_audio/")
async def clone_audio(audio_file: UploadFile = File(...), text: str = Form(...),language_tag:str =Form(...)):
    # Define paths for input and processed audio files
    input_audio_path = NamedTemporaryFile(delete=False).name
    processed_audio_path = NamedTemporaryFile(suffix='.mp3', delete=False).name

    # Save uploaded audio file
    with open(input_audio_path, "wb") as audio_output_file:
        audio_output_file.write(await audio_file.read())

    # Process audio
    process_audio(text, input_audio_path, processed_audio_path,language_tag)

    # Read processed audio file into memory
    with open(processed_audio_path, "rb") as processed_audio_file:
        processed_audio_data = processed_audio_file.read()

    # Remove temporary files
    os.unlink(input_audio_path)
    os.unlink(processed_audio_path)

    return Response(content=processed_audio_data, media_type="audio/mp3", headers={"Content-Disposition": "attachment;filename=processed_audio.mp3"})