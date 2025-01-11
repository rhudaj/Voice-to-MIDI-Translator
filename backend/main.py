#python imports
from io import BytesIO
import scipy.io.wavfile as wav
import numpy as np
#external imports
from midiutil import MIDIFile
#internal imports
from MidiOutput.MidiOut import signal_to_midi
from DataStructures.AudioSignal import AudioSignal, AudioSignal_FromFile

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

# FastAPI is a modern Python framework with built-in WebSocket support.
# to run the server: uvicorn main:app --reload

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Initialize a BytesIO object to hold the incoming binary data
    audio_data = b""

    # Keep receiving data until the WebSocket is closed
    try:
        while True:
            # Receive the audio file as binary data
            chunk = await websocket.receive_bytes()
            audio_data += chunk  # Append the received chunk to audio_data
            print(f"Received {len(chunk)} bytes, total {len(audio_data)} bytes")
            if (chunk):
                break
    except Exception as e:
        print("WebSocket closed or an error occurred:", e)

     # Once all data is received, process it
    if audio_data:
        try:
            # Convert the binary audio data into a NumPy array using BytesIO and scipy.io.wavfile
            audio_file_like = BytesIO(audio_data)  # Create a file-like object from the binary data
            sample_rate, audio_signal = wav.read(audio_file_like)
            audio_signal = np.array(audio_signal, dtype=np.float64)

            # Output the shape of the NumPy array (audio samples)
            print(f"Audio file converted to numpy array with shape: {audio_signal.shape}")
            await websocket.send_text("Audio file received and converted to numpy array.")  # Send message before closing

        except Exception as e:
            print(f"Error while processing the audio data: {str(e)}")
            await websocket.send_text("Error while processing the audio file.")  # Send error message before closing

    # conver to midi:

    midi: MIDIFile = signal_to_midi(audio_signal, sample_rate)

    await websocket.send_text("converted to midi file")

    with open ("output.mid", 'wb') as file:
        midi.writeFile(file)

    # Close the WebSocket connection
    await websocket.close()