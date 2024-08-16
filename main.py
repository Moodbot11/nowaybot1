import asyncio
from aiohttp import web
from lmnt.api import Speech
from openai import AsyncOpenAI

DEFAULT_PROMPT = 'Read me the text of a short sci-fi story in the public domain.'
VOICE_ID = 'lily'

async def main():
    async with Speech() as speech:
        connection = await speech.synthesize_streaming(VOICE_ID)
        t1 = asyncio.create_task(reader_task(connection))
        t2 = asyncio.create_task(writer_task(connection))
        await asyncio.gather(t1, t2)

async def reader_task(connection):
    with open('output.mp3', 'wb') as f:
        async for message in connection:
            f.write(message['audio'])

async def writer_task(connection):
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': DEFAULT_PROMPT}],
        stream=True)

    async for chunk in response:
        if (not chunk.choices[0] or
            not chunk.choices[0].delta or
            not chunk.choices[0].delta.content):
            continue
        content = chunk.choices[0].delta.content
        await connection.append_text(content)
        print(content, end='', flush=True)

    await connection.finish()

async def handle(request):
    return web.FileResponse('output.mp3')

app = web.Application()
app.router.add_get('/', handle)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    web.run_app(app, port=8000)