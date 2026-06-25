import os, sys, time
sys.path.insert(0, 'd:/CODE/VIDEO/YOUTUBE')

env = open('d:/CODE/VIDEO/YOUTUBE/.env').read()
for line in env.splitlines():
    if line.startswith('RUNPOD_API_KEY='):
        os.environ['RUNPOD_API_KEY'] = line.split('=', 1)[1].strip()
os.environ['RUNPOD_TTS_ENDPOINT_ID'] = 'syo26j5rexxrbl'

from tts_generation.runpod_tts_client import clone_voice

print('Submitting TTS job...')
t0 = time.time()
audio = clone_voice(
    text='Chân trái của một đứa trẻ bị cắt cụt 31.000 năm trước. Và nó đã sống. Đặt tay lên ống chân của bạn. Cảm nhận khúc xương dài ngay dưới da, và xương mỏng hơn bên cạnh nó.',
    voice_id='nhat',
    ref_audio_path='d:/CODE/VIDEO/YOUTUBE/output/ancient-child-surgery-31000-years-vi/voice_ref_nhat.mp3',
    ref_text='Xin kiến chào quý bạn và các vị. Món rằng dọng nói này sẽ phù hợp với bất kỳ dự án nào của bạn. Cảm ơn đã lựa chọn. Hãy thử ngay nhé.',
    speed=0.95,
)
out = 'd:/CODE/VIDEO/YOUTUBE/output/ancient-child-surgery-31000-years-vi/test_clone.mp3'
open(out, 'wb').write(audio)
print(f'Done in {time.time()-t0:.1f}s! {len(audio)//1024} KB -> {out}')
