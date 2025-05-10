import asyncio
import websockets

async def echo(websocket):  # Only one argument now
    async for message in websocket:
        await websocket.send(f"Echo: {message}")

async def main():
    async with websockets.serve(echo, "localhost", 8765):
        print("WebSocket server running at ws://localhost:8765")
        await asyncio.Future()  # run forever

asyncio.run(main())
