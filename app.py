# app.py
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import os
import asyncio
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import queue
from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

# 환경 변수 로드
env_path = r"C:\Users\bmc\Desktop\홍태광\workspace\.env"
load_dotenv(env_path)

app = Flask(__name__)
app.config['DEEPGRAM_API_KEY'] = os.getenv("DEEPGRAM_API_KEY")
app.config['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
app.config['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
socketio = SocketIO(app, cors_allowed_origins="*")

# SafeMicrophone 클래스 구현
class SafeMicrophone(Microphone):
    def __init__(self, send_callback, loop=None):
        super().__init__(send_callback)
        self.loop = loop or asyncio.get_event_loop()
        
    async def _send_data(self, data):
        try:
            if asyncio.iscoroutinefunction(self.send_callback):
                await self.send_callback(data)
            else:
                await self.loop.run_in_executor(None, self.send_callback, data)
        except Exception as e:
            print(f"Error sending data: {e}")
            
    def send(self, data):
        asyncio.run_coroutine_threadsafe(self._send_data(data), self.loop)

class AudioProcessor:
    def __init__(self):
        self.buffer_size = 4096
        self.is_processing = False
        self.deepgram = None
        self.dg_connection = None
        self.stop_event = asyncio.Event()
        
    async def init_deepgram(self):
        config = DeepgramClientOptions(options={"keepalive": "true"})
        self.deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"), config)
        self.dg_connection = self.deepgram.listen.asyncwebsocket.v("1")
        
        # Deepgram 이벤트 핸들러 설정
        self.dg_connection.on(LiveTranscriptionEvents.Transcript, self.on_message)
        self.dg_connection.on(LiveTranscriptionEvents.Error, self.on_error)

    async def on_message(self, self_, result, **kwargs):
        if not result or not result.channel or not result.channel.alternatives:
            return

        sentence = result.channel.alternatives[0].transcript
        
        if not sentence.strip():
            return
            
        if sentence:
            print("인식 중인 부분:", sentence)
            socketio.emit('terminal_output', {"type": "processing", "message": f"Processing message: {sentence}"})
        
        if result.is_final:
            if result.speech_final:
                socketio.emit('terminal_output', {"type": "final", "message": f"Speech Final: {sentence}"})
            else:
                socketio.emit('terminal_output', {"type": "final", "message": f"Is Final: {sentence}"})
        else:
            if sentence:
                socketio.emit('terminal_output', {"type": "interim", "message": f"Interim Results: {sentence}"})

    async def on_error(self, self_, error, **kwargs):
        error_message = f"Error occurred: {error}"
        print(error_message)
        socketio.emit('terminal_output', {"type": "error", "message": error_message})
        if not self.stop_event.is_set():
            await asyncio.sleep(1)
            await self.reconnect()

    async def reconnect(self):
        try:
            options = self.get_deepgram_options()
            addons = self.get_deepgram_addons()
            if await self.dg_connection.start(options, addons=addons) is False:
                print("Failed to reconnect to Deepgram")
                return False
            return True
        except Exception as e:
            print(f"Reconnection failed: {e}")
            return False

    def get_deepgram_options(self):
        return LiveOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
            encoding="linear16",
            channels=6,
            sample_rate=16000,
            interim_results=True,
            utterance_end_ms="2000",
            vad_events=True,
            endpointing=500,
        )

    def get_deepgram_addons(self):
        return {
            "no_delay": "true",
            "punctuate": "true"
        }

    async def process_audio(self, audio_data):
        if not self.deepgram:
            await self.init_deepgram()
            
        options = self.get_deepgram_options()
        addons = self.get_deepgram_addons()
        
        if await self.dg_connection.start(options, addons=addons) is False:
            print("Failed to connect to Deepgram")
            return

        current_loop = asyncio.get_event_loop()
        microphone = SafeMicrophone(
            send_callback=lambda data: current_loop.create_task(self.dg_connection.send(data)),
            loop=current_loop
        )

        try:
            microphone.start()
            # 오디오 데이터 처리
            await self.dg_connection.send(audio_data)
            
        except Exception as e:
            print(f"Error processing audio: {e}")
        finally:
            microphone.finish()
            await self.dg_connection.finish()

# STT/Translation 모델 구성
STT_MODELS = {
    'google': None,  # Google STT 클라이언트
    'whisper': None,  # OpenAI Whisper 클라이언트
    'nova2': None,  # Deepgram nova-2 클라이언트
    'whisper_large': None  # Deepgram whisper-large-v3 클라이언트
}

TRANSLATION_MODELS = {
    'llama_8b': None,  # Llama 3.1-8b 클라이언트
    'llama_70b': None,  # Llama 3.1-70b 클라이언트
    'gemma_7b': None  # Gemma 7b 클라이언트
}

# 오디오 처리를 위한 글로벌 큐
audio_queue = queue.Queue()
executor = ThreadPoolExecutor(max_workers=8)

@app.route('/')
def index():
    return render_template('index.html')

audio_processor = AudioProcessor()

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('audio_data')
def handle_audio_data(data):
    if not audio_processor.is_processing:
        audio_queue.put(data)
        executor.submit(process_audio_queue)

def process_audio_queue():
    while not audio_queue.empty():
        data = audio_queue.get()
        asyncio.run(audio_processor.process_audio(data))

if __name__ == '__main__':
    socketio.run(app, debug=True)