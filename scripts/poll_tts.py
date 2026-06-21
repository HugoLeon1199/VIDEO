import os, httpx, json, time

env = open('d:/CODE/VIDEO/YOUTUBE/.env').read()
for line in env.splitlines():
    if line.startswith('RUNPOD_API_KEY='):
        api_key = line.split('=', 1)[1].strip()

endpoint = 'syo26j5rexxrbl'
job_id = '8d5b9f0e-c8ee-4ebb-8c69-cd2fcdca2f39-e2'
headers = {'Authorization': f'Bearer {api_key}'}

print('Polling every 15s for up to 10 minutes...')
for i in range(40):
    r = httpx.get(f'https://api.runpod.ai/v2/{endpoint}/health', headers=headers, timeout=10)
    h = r.json()
    r2 = httpx.get(f'https://api.runpod.ai/v2/{endpoint}/status/{job_id}', headers=headers, timeout=10)
    j = r2.json()
    w = h['workers']
    print(f'[{i*15}s] init={w["initializing"]} ready={w["ready"]} idle={w["idle"]} | job={j["status"]}')
    if j['status'] in ('COMPLETED', 'FAILED'):
        print(json.dumps(j, indent=2))
        break
    time.sleep(15)
