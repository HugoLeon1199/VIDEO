import json, sys
video_id = sys.argv[1] if len(sys.argv) > 1 else "what-ancient-humans-did-all-day-vi"
data = json.load(open(f'output/{video_id}/timestamps.json', encoding='utf-8'))
print(f'Total: {len(data)} segments')
for s in data:
    print(f'[{s["index"]:03d}] {s["start"]:6.1f}s - {s["end"]:6.1f}s | {s["text"][:70]}')
