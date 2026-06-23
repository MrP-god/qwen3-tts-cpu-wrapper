import argparse
import uvicorn

def main():
    parser = argparse.ArgumentParser(description="Start the Qwen3-TTS CPU Verber Server.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind the server to.")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development.")
    
    args = parser.parse_args()
    
    print("=========================================================")
    print(f" Starting Qwen3-TTS CPU Verber on http://{args.host}:{args.port}")
    print("=========================================================")
    
    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
