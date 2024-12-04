from flask import Flask, request
from os import remove
from os.path import exists

raw_record_outfile = 'recorded_audio.raw'
if exists(raw_record_outfile):
    remove(raw_record_outfile)

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_audio():
    audio_data = request.data
    with open(raw_record_outfile, 'ab') as f: # Append binary data to the file
        f.write(audio_data)
    return 'Audio chunk received', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
