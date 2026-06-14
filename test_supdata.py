import httpx

SUPADATA_KEY = "sd_ca3ba00b5e7e557f20c7cb0dbe812233"  # paste from supadata.ai dashboard

def test(video_id: str):
    print(f"\nTesting: https://youtube.com/watch?v={video_id}")
    
    with httpx.Client(timeout=30) as client:
        r = client.get(
            "https://api.supadata.ai/v1/youtube/transcript",
            params={"videoId": video_id, "text": "true"},
            headers={"x-api-key": SUPADATA_KEY}
        )

    print(f"Status : {r.status_code}")

    if r.status_code == 200:
        data = r.json()
        content = data.get("content", "")
        lang    = data.get("lang", "unknown")
        print(f"✅  SUCCESS")
        print(f"Language : {lang}")
        print(f"Length   : {len(content)} chars")
        print(f"Preview  : {content[:300]}\n")
    else:
        print(f"❌  FAILED")
        print(f"Response : {r.text}\n")

# Test two videos
test("POQ3Vi81Djk")          # change to any video ID you want
# test("dQw4w9WgXcQ")          # classic test video