"""Streaming voice chat: STT (faster-whisper) -> LLM (vLLM/Qwen) -> TTS (Kokoro).

Speculative pipeline. Each new partial transcript kicks off a Speculation that
streams LLM tokens into Kokoro and ships tagged PCM straight to the browser,
which buffers per spec_id but does not play. When the next partial arrives, the
prior speculation is cancelled and the browser drops its buffer for that id.
The trigger word "go" releases the matching speculation: the browser plays
everything it has buffered for that id and any further chunks live."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

import httpx
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from kokoro import KPipeline
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("voicechat")

ROOT = Path(__file__).parent
LLM_BASE = os.environ.get("LLM_BASE", "http://127.0.0.1:8001/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen")
SAMPLE_RATE_IN = 16000           # browser sends 16k mono float32
SAMPLE_RATE_OUT = 24000          # Kokoro emits 24k float32
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small.en")
KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")
SYSTEM_PROMPT = (
    "You are a fast, friendly voice assistant. Keep replies concise and conversational, "
    "two to four short sentences unless asked for more. No markdown, no bullet lists."
)

# Qwen3 supports a "thinking" mode that wraps reasoning in <think>…</think>;
# for a low-latency voice app we always want it OFF so the first token IS speech.
LLM_EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}

# ---- shared singletons ------------------------------------------------------

log.info("Loading faster-whisper model %s ...", WHISPER_MODEL_SIZE)
whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="float16")
log.info("Loading Silero VAD ...")
vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
)
log.info("Loading Kokoro TTS ...")
kokoro = KPipeline(lang_code="a")  # American English
log.info("Connecting to vLLM at %s", LLM_BASE)
llm = AsyncOpenAI(base_url=LLM_BASE, api_key="local", timeout=httpx.Timeout(120.0))

app = FastAPI()
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

@app.get("/")
async def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


# ---- audio helpers ----------------------------------------------------------


def frame_energy(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))


def vad_speech_prob(audio: np.ndarray) -> float:
    """Silero expects 512-sample chunks at 16kHz; returns highest prob in window."""
    if audio.size < 512:
        return 0.0
    t = torch.from_numpy(audio.astype(np.float32))
    probs: list[float] = []
    for i in range(0, audio.size - 512 + 1, 512):
        chunk = t[i : i + 512]
        probs.append(float(vad_model(chunk, SAMPLE_RATE_IN).item()))
    return max(probs) if probs else 0.0


# ---- session state ----------------------------------------------------------


GO_RE = re.compile(r"\b(go|okay go|alright go)\s*[.!?]?\s*$", re.IGNORECASE)


def strip_trigger(text: str) -> str:
    return GO_RE.sub("", text).strip()


def has_trigger(text: str) -> bool:
    return bool(GO_RE.search(text.strip()))


@dataclass
class Speculation:
    """A speculative response in flight. Audio is shipped to the browser as it's
    synthesized — the browser is the buffer, and decides on release whether to play."""
    input_text: str
    spec_id: int
    task: asyncio.Task | None = None
    full_text: str = ""
    released: asyncio.Event = field(default_factory=asyncio.Event)
    completed: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class Session:
    ws: WebSocket
    history: list[dict] = field(default_factory=list)  # chat history, role/content
    audio_buf: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    speech_active: bool = False
    silence_ms: int = 0
    last_partial_text: str = ""
    last_stt_run: float = 0.0
    spec: Speculation | None = None
    spec_counter: int = 0  # monotonically increasing spec_id source
    in_turn: bool = False  # True from "go" trigger until release_spec finishes
    turn_id: int = 0


async def send_json(ws: WebSocket, **payload) -> None:
    try:
        await ws.send_text(json.dumps(payload))
    except Exception:
        pass


async def send_pcm(ws: WebSocket, spec_id: int, audio_f32: np.ndarray) -> None:
    """Convert f32 mono @ 24k to int16 LE; prefix 4-byte LE uint32 spec_id."""
    pcm = np.clip(audio_f32, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    header = int(spec_id).to_bytes(4, "little", signed=False)
    try:
        await ws.send_bytes(header + pcm.tobytes())
    except Exception:
        pass


# ---- STT loop ---------------------------------------------------------------


def transcribe(audio: np.ndarray, prompt: str = "") -> str:
    """Synchronous whisper transcribe — call via run_in_executor."""
    if audio.size < SAMPLE_RATE_IN // 2:  # <0.5s
        return ""
    segs, _ = whisper.transcribe(
        audio,
        language="en",
        beam_size=1,
        vad_filter=False,
        condition_on_previous_text=False,
        initial_prompt=prompt or None,
        without_timestamps=True,
    )
    return " ".join(s.text.strip() for s in segs).strip()


async def run_stt_partial(sess: Session) -> str | None:
    """Transcribe the active speech buffer; returns text if changed."""
    if sess.audio_buf.size < SAMPLE_RATE_IN // 2:
        return None
    loop = asyncio.get_running_loop()
    text = await loop.run_in_executor(None, transcribe, sess.audio_buf.copy(), "")
    if not text or text == sess.last_partial_text:
        return None
    sess.last_partial_text = text
    return text


# ---- LLM prefill + generation ----------------------------------------------


async def llm_stream(sess: Session, user_text: str) -> AsyncIterator[str]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, *sess.history,
            {"role": "user", "content": user_text}]
    stream = await llm.chat.completions.create(
        model=LLM_MODEL, messages=msgs, max_tokens=400, temperature=0.7, stream=True,
        extra_body=LLM_EXTRA,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta


# ---- TTS ---------------------------------------------------------------------


SENT_END = re.compile(r"([.!?,;:]|\n)\s")


def tts_chunks(text: str) -> list[np.ndarray]:
    """Run Kokoro on a text chunk; returns 24kHz float32 audio (possibly multiple)."""
    out: list[np.ndarray] = []
    for _, _, audio in kokoro(text, voice=KOKORO_VOICE, speed=1.0):
        if audio is not None:
            out.append(audio.cpu().numpy().astype(np.float32))
    return out


async def speculate_run(sess: Session, spec: Speculation) -> None:
    """LLM stream → TTS → tagged PCM straight to the browser (which buffers per spec_id)."""
    loop = asyncio.get_running_loop()
    buf = ""
    last_flush = time.time()
    await send_json(sess.ws, type="spec_start", id=spec.spec_id, text=spec.input_text)

    async def flush(piece: str) -> None:
        if not piece.strip():
            return
        chunks = await loop.run_in_executor(None, tts_chunks, piece)
        for c in chunks:
            await send_pcm(sess.ws, spec.spec_id, c)

    try:
        async for delta in llm_stream(sess, spec.input_text):
            spec.full_text += delta
            buf += delta
            now = time.time()
            m = SENT_END.search(buf)
            force = (now - last_flush) > 0.9 and len(buf.strip()) > 12
            if m or force:
                cut = (m.end() if m else len(buf))
                piece, buf = buf[:cut], buf[cut:]
                await flush(piece)
                last_flush = now
                if spec.released.is_set():
                    await send_json(sess.ws, type="assistant_partial", text=spec.full_text)
        if buf.strip():
            await flush(buf)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("speculate_run failed")
    finally:
        spec.completed.set()


async def cancel_spec(sess: Session, spec: Speculation | None) -> None:
    if spec is None:
        return
    if spec.task and not spec.task.done():
        spec.task.cancel()
        try:
            await spec.task
        except (asyncio.CancelledError, Exception):
            pass
    # tell browser to drop any audio it has buffered for this spec_id
    await send_json(sess.ws, type="spec_cancel", id=spec.spec_id)


def kick_spec(sess: Session, user_text: str) -> Speculation:
    """Start a new speculative response. Caller is responsible for cancelling any prior spec."""
    sess.spec_counter += 1
    spec = Speculation(input_text=user_text, spec_id=sess.spec_counter)
    spec.task = asyncio.create_task(speculate_run(sess, spec))
    sess.spec = spec
    return spec


async def release_spec(sess: Session, spec: Speculation, turn_id: int) -> None:
    """Promote a speculation: tell browser to play its buffered audio for this id, finish streaming."""
    sess.in_turn = True
    try:
        await send_json(sess.ws, type="assistant_start", turn=turn_id)
        await send_json(sess.ws, type="spec_release", id=spec.spec_id)
        spec.released.set()
        if spec.full_text:
            await send_json(sess.ws, type="assistant_partial", text=spec.full_text)
        try:
            await spec.task  # type: ignore[arg-type]
        except (asyncio.CancelledError, Exception):
            log.exception("released spec ended abnormally")

        await send_json(sess.ws, type="assistant_final", text=spec.full_text)

        if spec.full_text.strip():
            sess.history.append({"role": "user", "content": spec.input_text})
            sess.history.append({"role": "assistant", "content": spec.full_text})
            if len(sess.history) > 12:
                sess.history = sess.history[-12:]
    finally:
        sess.in_turn = False
        if sess.spec is spec:
            sess.spec = None


# ---- main per-connection orchestrator --------------------------------------


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    sess = Session(ws=ws)
    await send_json(ws, type="ready")
    log.info("client connected")

    SILENCE_END_MS = 700
    SILENCE_END_MS_WITH_TRIGGER = 250  # snappier when "go" is already heard
    PARTIAL_INTERVAL_S = 0.45

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"] is not None:
                # 16kHz mono Float32 PCM
                arr = np.frombuffer(msg["bytes"], dtype=np.float32)
                if arr.size == 0:
                    continue

                # gate by VAD
                prob = vad_speech_prob(arr)
                is_speech = prob > 0.5 or frame_energy(arr) > 0.02

                if is_speech:
                    sess.audio_buf = np.concatenate([sess.audio_buf, arr])
                    sess.speech_active = True
                    sess.silence_ms = 0
                elif sess.speech_active:
                    # keep a short tail
                    sess.audio_buf = np.concatenate([sess.audio_buf, arr])
                    sess.silence_ms += int(arr.size * 1000 / SAMPLE_RATE_IN)

                # cap buffer to avoid runaway
                max_samples = SAMPLE_RATE_IN * 25  # 25s
                if sess.audio_buf.size > max_samples:
                    sess.audio_buf = sess.audio_buf[-max_samples:]

                # periodic partial STT — every change kicks a speculation
                now = time.time()
                if (sess.speech_active and not sess.in_turn
                        and sess.audio_buf.size >= SAMPLE_RATE_IN // 2
                        and now - sess.last_stt_run >= PARTIAL_INTERVAL_S):
                    sess.last_stt_run = now
                    text = await run_stt_partial(sess)
                    if text:
                        await send_json(ws, type="user_partial", text=text)
                        cleaned = strip_trigger(text)
                        # restart speculation if input text changed and trigger not present
                        if cleaned and not has_trigger(text):
                            if sess.spec is None or sess.spec.input_text != cleaned:
                                await cancel_spec(sess, sess.spec)
                                kick_spec(sess, cleaned)

                # end-of-turn?
                eot_thresh = (SILENCE_END_MS_WITH_TRIGGER
                              if has_trigger(sess.last_partial_text)
                              else SILENCE_END_MS)
                if (sess.speech_active and sess.silence_ms >= eot_thresh
                        and not sess.in_turn):
                    final_audio = sess.audio_buf.copy()
                    sess.audio_buf = np.zeros(0, dtype=np.float32)
                    sess.speech_active = False
                    sess.silence_ms = 0
                    sess.last_partial_text = ""

                    loop = asyncio.get_running_loop()
                    final_text = await loop.run_in_executor(None, transcribe, final_audio, "")
                    if not final_text:
                        continue
                    await send_json(ws, type="user_final", text=final_text)

                    if not has_trigger(final_text):
                        # no trigger: drop the in-flight spec, wait for next utterance
                        await cancel_spec(sess, sess.spec)
                        sess.spec = None
                        continue

                    user_msg = strip_trigger(final_text)
                    if not user_msg:
                        continue

                    # if the running (or already-finished) spec matches, release it;
                    # otherwise cancel and start a fresh one for the corrected text
                    if sess.spec is None or sess.spec.input_text != user_msg:
                        await cancel_spec(sess, sess.spec)
                        kick_spec(sess, user_msg)

                    sess.turn_id += 1
                    asyncio.create_task(release_spec(sess, sess.spec, sess.turn_id))

            elif "text" in msg and msg["text"]:
                # JSON control: {"type":"reset"} etc.
                try:
                    data = json.loads(msg["text"])
                except Exception:
                    continue
                if data.get("type") == "reset":
                    sess.history.clear()
                    sess.audio_buf = np.zeros(0, dtype=np.float32)
                    sess.speech_active = False
                    sess.last_partial_text = ""
                    await cancel_spec(sess, sess.spec)
                    sess.spec = None
                    sess.in_turn = False
                    await send_json(ws, type="reset_ok")

    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("ws handler crashed")
    finally:
        await cancel_spec(sess, sess.spec)
        log.info("client disconnected")


if __name__ == "__main__":
    import uvicorn
    cert = ROOT / "cert.pem"
    key = ROOT / "key.pem"
    kwargs = {"host": "0.0.0.0", "port": 8000, "log_level": "info"}
    if cert.exists() and key.exists():
        kwargs["ssl_certfile"] = str(cert)
        kwargs["ssl_keyfile"] = str(key)
        log.info("TLS enabled — visit https://<host>:8000")
    uvicorn.run(app, **kwargs)
