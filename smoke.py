"""End-to-end smoke test: synth audio with Kokoro -> push to /ws -> collect reply."""
import asyncio, json, ssl, sys, time
import numpy as np
import websockets
from kokoro import KPipeline

QUESTION = "What is two plus two? Go."
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

async def main():
    print("synth question with kokoro...", flush=True)
    k = KPipeline(lang_code="a")
    audio_24k = []
    for _, _, a in k(QUESTION, voice="af_heart", speed=1.0):
        audio_24k.append(a.cpu().numpy())
    a24 = np.concatenate(audio_24k).astype(np.float32)
    # resample 24k -> 16k
    n16 = int(a24.size * 16000 / 24000)
    idx = np.linspace(0, a24.size - 1, n16)
    a16 = np.interp(idx, np.arange(a24.size), a24).astype(np.float32)
    # pad with 0.8s silence after
    a16 = np.concatenate([a16, np.zeros(int(16000*0.8), dtype=np.float32)])
    print(f"  {a16.size/16000:.2f}s of 16k audio", flush=True)

    print("connecting...", flush=True)
    received_text = []
    received_audio_bytes = 0
    saw_assistant_final = asyncio.Event()
    async with websockets.connect("wss://127.0.0.1:8000/ws", max_size=None, ssl=SSL_CTX) as ws:
        spec_audio_bytes = {}    # spec_id -> bytes received before any release
        live_spec = [None]
        async def reader():
            nonlocal received_audio_bytes
            async for msg in ws:
                if isinstance(msg, bytes):
                    spec_id = int.from_bytes(msg[:4], "little", signed=False)
                    pcm = msg[4:]
                    spec_audio_bytes[spec_id] = spec_audio_bytes.get(spec_id, 0) + len(pcm)
                    if live_spec[0] == spec_id:
                        received_audio_bytes += len(pcm)
                else:
                    j = json.loads(msg)
                    print("  <-", j, flush=True)
                    if j.get("type") == "spec_release":
                        live_spec[0] = j["id"]
                        # count anything we already received for this spec_id as "live"
                        received_audio_bytes += spec_audio_bytes.get(j["id"], 0)
                    if j.get("type") in ("user_partial","user_final","assistant_partial","assistant_final"):
                        received_text.append(j)
                    if j.get("type") == "assistant_final":
                        saw_assistant_final.set()
        rt = asyncio.create_task(reader())

        # stream in 20ms (320 samples) frames
        FR = 320
        for i in range(0, a16.size, FR):
            frame = a16[i:i+FR]
            if frame.size < FR:
                frame = np.concatenate([frame, np.zeros(FR - frame.size, dtype=np.float32)])
            await ws.send(frame.tobytes())
            await asyncio.sleep(0.02)
        # wait up to 30s for final
        try:
            await asyncio.wait_for(saw_assistant_final.wait(), timeout=30)
        except asyncio.TimeoutError:
            print("TIMEOUT waiting for assistant_final", flush=True)
        rt.cancel()

    print(f"\nreceived audio bytes: {received_audio_bytes} ({received_audio_bytes/2/24000:.2f}s)", flush=True)
    print("text events:", len(received_text), flush=True)

asyncio.run(main())
