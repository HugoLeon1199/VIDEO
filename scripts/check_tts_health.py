import os, httpx, json

env = open('d:/CODE/VIDEO/YOUTUBE/.env').read()
for line in env.splitlines():
    if line.startswith('RUNPOD_API_KEY='):
        api_key = line.split('=', 1)[1].strip()

endpoint = 'syo26j5rexxrbl'
headers = {'Authorization': f'Bearer {api_key}'}

# Health
r = httpx.get(f'https://api.runpod.ai/v2/{endpoint}/health', headers=headers, timeout=10)
print('HEALTH:', json.dumps(r.json(), indent=2))

# Latest job
job_id = '548d222d-adc1-4e51-bb74-895af7069e86-e1'
r2 = httpx.get(f'https://api.runpod.ai/v2/{endpoint}/status/{job_id}', headers=headers, timeout=10)
print('JOB:', json.dumps(r2.json(), indent=2))
