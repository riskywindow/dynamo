#!/usr/bin/env python3
"""proxy.py LISTEN_PORT TARGET_FILE: TCP proxy; each new connection goes to 127.0.0.1:<port in TARGET_FILE>.
Used to move new requests from a killed primary to a promoted shadow."""
import asyncio, sys

LISTEN, TARGET = int(sys.argv[1]), sys.argv[2]


async def pipe(r, w):
    try:
        while data := await r.read(65536):
            w.write(data)
            await w.drain()
    except Exception:
        pass
    finally:
        w.close()


async def handle(cr, cw):
    port = int(open(TARGET).read().strip())
    try:
        ur, uw = await asyncio.open_connection("127.0.0.1", port)
    except Exception:
        cw.close()
        return
    await asyncio.gather(pipe(cr, uw), pipe(ur, cw))


async def main():
    server = await asyncio.start_server(handle, "127.0.0.1", LISTEN)
    async with server:
        await server.serve_forever()

asyncio.run(main())
