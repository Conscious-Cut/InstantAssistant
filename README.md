# InstantAssistant

A streaming GPU voice assistant. Talk to it, end your turn with the word **`go`**, and it talks back. Every stage streams: STT pushes partial transcripts into the LLM, the LLM streams tokens into the TTS, and TTS audio is shipped to the browser as it's synthesized — even before you've finished talking.

<video src="https://raw.githubusercontent.com/Conscious-Cut/InstantAssistant/main/IMG_2129.mov" controls muted playsinline width="720"></video>

[(direct download)](https://raw.githubusercontent.com/Conscious-Cut/InstantAssistant/main/IMG_2129.mov)

## Pipeline

```
mic → 16k Float32 PCM ─┐
                       │  (per WebSocket frame)
                       ▼
                  Silero VAD
                       │
                       ▼
              faster-whisper        ── partial transcript every ~450ms
                       │
                       │   each new partial:
                       │     • cancels the in-flight Speculation
                       │     • starts a fresh one for the new text
                       ▼
        ┌──────────── Speculation ──────────────┐
        │  vLLM chat stream (Qwen3.6-35B-A3B)   │
        │           │                            │
        │           ▼ tokens                     │
        │   sentence-buffered Kokoro TTS         │
        │           │                            │
        │           ▼ 24k Float32 PCM            │
        └───────────│────────────────────────────┘
                    │
                    │  tagged binary frame
                    │  [ u32 LE spec_id ][ int16 LE PCM ]
                    ▼
           ┌─── browser ────┐
           │  pendingBySpec │   buffer per spec_id; do not play
           │       │        │
           │   "go" ⇒ spec_release → drain buffer + play live
           └────────────────┘
```

The "speculation" idea: as soon as Whisper emits a partial transcript like *"what is the capital of France"*, the server immediately pretends you're done and starts generating + speaking a reply. The audio reaches the browser but is held silent. If your next word changes the transcript, that speculation is cancelled and a new one starts. If your next word is **`go`**, the held audio is released — first-sound latency from "go" is essentially network RTT.

## Stack

- **STT** — `faster-whisper small.en`, FP16, with Silero VAD gating.
- **LLM** — `RedHatAI/Qwen3.6-35B-A3B-NVFP4` served by vLLM (`--enable-prefix-caching`, non-thinking mode via `chat_template_kwargs.enable_thinking=false`).
- **TTS** — `Kokoro-82M` (`af_heart`), 24 kHz, sentence-flushed.
- **Server** — FastAPI + uvicorn over TLS (self-signed; required because secure-context restrictions block `getUserMedia` and `audioWorklet` on plain HTTP for non-`localhost` origins).
- **Front-end** — vanilla JS, AudioWorklet capture, AudioBufferSourceNode playback, no framework.

## WebSocket protocol

- **Client → server** — binary frames of 320 samples (20 ms) Float32 mono @ 16 kHz; JSON `{type:"reset"}` to clear chat history.
- **Server → client**:
  - JSON: `ready`, `user_partial/final`, `spec_start{id,text}`, `spec_cancel{id}`, `spec_release{id}`, `assistant_start/partial/final`.
  - Binary: `[u32 LE spec_id][int16 LE PCM @ 24 kHz mono]`.

## Running

GPU required (Blackwell tested; FP4 quantization is Blackwell-native). You also need a vLLM install reachable on `LLM_BASE` (default `http://127.0.0.1:8001/v1`) serving the Qwen model under the name `qwen`.

```bash
# 1) python deps
uv venv venv
VIRTUAL_ENV=$PWD/venv uv pip install \
    fastapi 'uvicorn[standard]' websockets numpy soundfile openai httpx \
    faster-whisper kokoro silero-vad torch torchaudio --torch-backend=auto
VIRTUAL_ENV=$PWD/venv uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12  # CTranslate2 wants cu12
VIRTUAL_ENV=$PWD/venv uv pip install \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# 2) self-signed cert (so the browser allows the mic)
openssl req -x509 -nodes -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 \
  -subj "/CN=voicechat" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:<your-LAN-IP>"

# 3) start vLLM (separate venv recommended; needs trust_remote_code)
vllm serve RedHatAI/Qwen3.6-35B-A3B-NVFP4 \
    --host 127.0.0.1 --port 8001 --served-model-name qwen \
    --max-model-len 16384 --gpu-memory-utilization 0.78 \
    --enable-prefix-caching --trust-remote-code

# 4) start the web server
./run.sh

# 5) open https://<your-host>:8000  (accept the self-signed cert), click "Start mic"
```

`smoke.py` runs an end-to-end test: it synthesizes *"What is two plus two? Go."* with Kokoro, streams it into `/ws`, and verifies the speculation/release path produces audio.

## Configuration

Environment variables read by `server.py`:

| var              | default                       | meaning                          |
| ---------------- | ----------------------------- | -------------------------------- |
| `LLM_BASE`       | `http://127.0.0.1:8001/v1`    | OpenAI-compatible vLLM endpoint  |
| `LLM_MODEL`      | `qwen`                        | served model name                |
| `WHISPER_MODEL`  | `small.en`                    | faster-whisper model id          |
| `KOKORO_VOICE`   | `af_heart`                    | Kokoro voice                     |
