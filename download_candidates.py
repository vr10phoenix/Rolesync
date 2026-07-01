import argparse
import gdown
import os
import sys

def download_file(url, output="candidates.jsonl"):
    if "drive.google.com" in url:
        # Extract file ID
        import re
        match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
        if match:
            file_id = match.group(1)
            url = f"https://drive.google.com/uc?id={file_id}"
        else:
            url = url
    else:
        url = url 

    print(f"Downloading from {url} ...")
    gdown.download(url, output, quiet=False)

    if os.path.exists(output):
        size = os.path.getsize(output)
        print(f"Downloaded {output} ({size} bytes).")
        with open(output, "r") as f:
            first = f.readline().strip()
        if first.startswith("<!DOCTYPE") or first.startswith("<html"):
            print("Error in Download.")
            sys.exit(1)
        else:
            print("File valid.")
    else:
        print("Download failed.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://drive.google.com/file/d/1-f-XfJ2pqYZMyl0Ru2rBCBN3Wc7861jw/view?usp=drive_link",
                        help="Google Drive share link or direct URL to the candidates.jsonl file")
    parser.add_argument("--output", default="candidates.jsonl", help="Output filename (default: candidates.jsonl)")
    args = parser.parse_args()
    download_file(args.url, args.output)