from pydub import AudioSegment
import csv

# Load the audio file
audio = AudioSegment.from_file("LKY-speech-For Third World Leaders Hope or Despair.mp3")

# Load the timestamps
with open("LKY-speech-.tsv") as tsvfile:
    reader = csv.DictReader(tsvfile, delimiter='\t')
    for i, row in enumerate(reader):
        start_time = int(row["start"])
        end_time = int(row["end"])
        sentence_audio = audio[start_time:end_time]
        sentence_audio.export("sentence_{}_{}.wav".format(i,row["text"]), format="wav")
