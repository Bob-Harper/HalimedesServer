import json
import os
import sys
import asyncio
import websockets
import concurrent.futures
import logging
from vosk import Model, SpkModel, KaldiRecognizer
from dotenv import load_dotenv

# ------------------------------------------------------------
# Load environment variables from .env, with fallback defaults
# ------------------------------------------------------------
load_dotenv()

MODEL_PATH = os.environ.get("VOSK_MODEL_PATH")
if not MODEL_PATH:
    raise RuntimeError("VOSK_MODEL_PATH is not set in the environment.")

INTERFACE = os.environ.get("VOSK_BIND_HOST", "0.0.0.0")
PORT = int(os.environ.get("VOSK_BIND_PORT", 2700))
SPK_MODEL_PATH = os.environ.get("VOSK_SPK_MODEL_PATH")
SAMPLE_RATE = float(os.environ.get("VOSK_SAMPLE_RATE", 44100))
MAX_ALTERNATIVES = int(os.environ.get("VOSK_ALTERNATIVES", 0))
SHOW_WORDS = os.environ.get("VOSK_SHOW_WORDS", "true").lower() == "true"

# ------------------------------------------------------------
# Globals
# ------------------------------------------------------------
model = None
spk_model = None
pool = None

# ------------------------------------------------------------
# Chunk processing (runs in thread pool)
# ------------------------------------------------------------
def process_chunk(rec, message):
    if message == '{"eof" : 1}':
        return rec.FinalResult(), True
    if message == '{"reset" : 1}':
        return rec.FinalResult(), False
    elif rec.AcceptWaveform(message):
        return rec.Result(), False
    else:
        return rec.PartialResult(), False

# ------------------------------------------------------------
# Per‑connection recognizer loop
# ------------------------------------------------------------
async def recognize(websocket):
    global model, spk_model, pool

    loop = asyncio.get_running_loop()
    rec = None
    phrase_list = None

    # These can be overridden per connection
    sample_rate = SAMPLE_RATE
    show_words = SHOW_WORDS
    max_alternatives = MAX_ALTERNATIVES

    model_changed = False

    logging.info('Connection from %s', websocket.remote_address)

    while True:
        message = await websocket.recv()

        # Handle config messages
        if isinstance(message, str) and 'config' in message:
            jobj = json.loads(message)['config']
            logging.info("Config %s", jobj)

            if 'phrase_list' in jobj:
                phrase_list = jobj['phrase_list']

            if 'sample_rate' in jobj:
                sample_rate = float(jobj['sample_rate'])

            if 'model' in jobj:
                model = Model(jobj['model'])
                model_changed = True

            if 'words' in jobj:
                show_words = bool(jobj['words'])

            if 'max_alternatives' in jobj:
                max_alternatives = int(jobj['max_alternatives'])

            continue

        # Create recognizer if needed
        if not rec or model_changed:
            model_changed = False

            if phrase_list:
                rec = KaldiRecognizer(model, sample_rate, json.dumps(phrase_list, ensure_ascii=False))
            else:
                rec = KaldiRecognizer(model, sample_rate)

            rec.SetWords(show_words)
            rec.SetMaxAlternatives(max_alternatives)

            if spk_model:
                rec.SetSpkModel(spk_model)

        # Process audio chunk in thread pool
        response, stop = await loop.run_in_executor(pool, process_chunk, rec, message)

        logging.info(f"[VoskServer] Received {len(message)} bytes")
        logging.info(f"[VoskServer] Sending: {response[:200]}")

        await websocket.send(response)

        if stop:
            break

# ------------------------------------------------------------
# Server startup
# ------------------------------------------------------------
async def start():
    global model, spk_model, pool

    logging.basicConfig(level=logging.INFO)

    model = Model(MODEL_PATH)
    spk_model = SpkModel(SPK_MODEL_PATH) if SPK_MODEL_PATH else None

    pool = concurrent.futures.ThreadPoolExecutor(os.cpu_count() or 1)

    print("Vosk loading model from:", MODEL_PATH)

    server = await websockets.serve(recognize, INTERFACE, PORT)
    await server.wait_closed()

# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == '__main__':
    asyncio.run(start())