import argparse
from .web.server import start_server

def main():
    parser = argparse.ArgumentParser(description="AI Subtitle Generator Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    args = parser.parse_args()
    
    print(f"Starting AI Subtitle Generator UI on port {args.port}...")
    start_server(port=args.port)

if __name__ == "__main__":
    main()
