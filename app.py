from flask import Flask, render_template, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from IPython.display import YouTubeVideo
from deep_translator import GoogleTranslator
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__, template_folder="./Template")

checkpoint = "bart_samsum_model"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model= AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

summarizer_bart = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize(full_txt, min_summ_len=130):

    summarizer_ft = pipeline (
        "summarization",
        model = base_model,
        tokenizer = tokenizer
    )

    l = full_txt.split(" ")
    l_summ = []
    chunk_len = 750
    overlap = 50
    pointer = 0
    flag = True
    while(flag):
        if pointer < len(l):
            if pointer + chunk_len < len(l):
                txt = " ".join(l[pointer:pointer+chunk_len])
                pointer = pointer + chunk_len - overlap
                l_summ.append(summarizer_ft(txt, max_length=130, min_length=40, do_sample=False)[0]['summary_text'])
            else:
                txt = " ".join(l[pointer:])
                l_summ.append(summarizer_ft(txt, max_length=len(l) - pointer, min_length=40, do_sample=False)[0]['summary_text'])
                pointer = len(l)
                flag = False

    large_summ = " ".join(l_summ)
    l_large_summ = large_summ.split(" ")

    if len(large_summ.split(" ")) < chunk_len:
        summ = summarizer_bart(large_summ, max_length=300, min_length=int(min_summ_len), do_sample=False)[0]['summary_text']
    else: 
        flag = True
        pointer = 0
        final_summ = []
        while(flag):
            if pointer < len(l_large_summ):
                if pointer + chunk_len < len(l_large_summ):
                    txt = " ".join(l_large_summ[pointer:pointer+chunk_len])
                    pointer = pointer + chunk_len - overlap
                    t = summarizer_bart(txt, max_length=130, min_length=40, do_sample=False)[0]['summary_text']
                    final_summ.append(t)
                else:
                    txt = " ".join(l_large_summ[pointer:])
                    t = summarizer_bart(txt, max_length=len(l_large_summ)-pointer, min_length=40, do_sample=False)[0]['summary_text']
                    final_summ.append(t)
                    pointer = len(l_large_summ)
                    flag = False
        large_summ = " ".join(final_summ)
        summ = summarizer_bart(large_summ, max_length=100, min_length=int(min_summ_len), do_sample=False)[0]['summary_text']
    return summ
    pass

def extract_text(youtube_video_url, language, min_summ_len):
    video_id = youtube_video_url.split("=")[1]

    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    transcript = transcript_list.find_transcript(['en','hi','mr','gu','kn','ta','te','pa','ur','fr','de'])
    translated_transcript = transcript.translate('en')
    transcript_text = translated_transcript.fetch()
    
    transcript = " ".join([i["text"] for i in transcript_text])
    
    # Summarize the transcript in English
    english_summary = summarize(transcript, min_summ_len)
    
    # Translate the summary to the requested language
    translated_summary = GoogleTranslator(source='en', target=language).translate(english_summary)
    
    return translated_summary


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def get_summary():
    youtube_video_url = request.form['youtube_video_url']
    min_summ_len = int(request.form['min_summ_len'])
    language = request.form['language']
    summary = extract_text(youtube_video_url,language, min_summ_len)
    
    # Fetching video thumbnail URL using YouTube API
    video_id = youtube_video_url.split("=")[1]
    thumbnail_url = f'https://www.youtube.com/embed/{video_id}'
    
    # Returning JSON response with summary and thumbnail URL
    return jsonify(summary=summary, thumbnail_url=thumbnail_url)

if __name__ == '__main__':
    app.config['DEBUG'] = True
    app.run(debug=True)