from gtts import gTTS
import os

texts = ['ARマーカーが認識できません。マーカーを確認してください。']

for i in range(len(texts)):
    tts = gTTS(text=texts[i],lang='ja')
    tts.save(f'{i}.mp3')
