import asyncio
import ui
import sk_helper as sk


audio_file = "./audio_file_2.mp3"

async def main():
    ui.run_ui()

# NOTE THAT KERNEL RUNS ASYNCHRONOUSLY
if __name__ == "__main__":
    asyncio.run(main())