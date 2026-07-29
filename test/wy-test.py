import asyncio, os, sys, time, wave
from wyoming.client import AsyncTcpClient
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.info import Describe, Info

async def once(path):
    w = wave.open(path,"rb"); pcm = w.readframes(w.getnframes()); rate=w.getframerate()
    async with AsyncTcpClient("127.0.0.1", int(os.environ.get("PORT", 7892))) as c:
        await c.write_event(Transcribe(language="en").event())
        await c.write_event(AudioStart(rate=rate, width=2, channels=1).event())
        t=time.time()
        for i in range(0, len(pcm), 4096):
            await c.write_event(AudioChunk(rate=rate,width=2,channels=1,audio=pcm[i:i+4096]).event())
        await c.write_event(AudioStop().event())
        while True:
            e = await c.read_event()
            if e is None: return None, 0
            if Transcript.is_type(e.type):
                return Transcript.from_event(e).text, (time.time()-t)*1000

async def main():
    async with AsyncTcpClient("127.0.0.1", int(os.environ.get("PORT", 7892))) as c:
        await c.write_event(Describe().event())
        e = await c.read_event()
        print("INFO ok:", Info.is_type(e.type))
    for p in sys.argv[1:]:
        txt, ms = await once(p)
        print(f"{p.split('/')[-1]:14s} {ms:7.0f}ms :: {txt!r}")
asyncio.run(main())
