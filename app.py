""" 
Cognitive distortion
 """


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize, pos_tag
import pandas as pd
import nltk
from nltk.corpus import wordnet
import re
import speech_recognition as sr
import sounddevice as sd
from scipy.io.wavfile import write
import os
import time
import requests 
from pydub import AudioSegment
import csv
import sqlite3
from PyDictionary import PyDictionary

# Ensure you have downloaded the necessary data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
import nltk
from nltk.corpus import wordnet
import requests

nltk.download('wordnet')

dictionary=PyDictionary()

def get_wordnet_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

def get_online_synonyms(word):
    #this recieved locally in sqllite db with thesaurus from project gutenberg
    conn = sqlite3.connect('my_database.db')
    cursor = conn.cursor()
    #print("get_online_synonyms: word")
    #print(word)
    # Insert data into the table
    cursor.execute("SELECT synonym FROM synonymslistings WHERE term='%s'" % word.replace("'","''"))
    result = cursor.fetchall()
    conn.commit()
    cursor.close() 
    conn.close()
    return {item for item in result}
    #response = requests.get(f"https://api.datamuse.com/words?rel_syn={word}")
    #return {item['word'] for item in response.json()}

def get_combined_synonyms(word):
    wordnet_synonyms = get_wordnet_synonyms(word)
    online_synonyms = get_online_synonyms(word) #dictionary.synonym(word)
    return wordnet_synonyms.union(online_synonyms)

def check_pos(word):
    text = word_tokenize(word)
    tagged = pos_tag(text)
    return tagged[0][1]

def isCertainWordType(word):
    pos = check_pos(word)
    if pos.startswith('RB'):
        return True
    elif pos.startswith('VB'):
        return True
    elif pos.startswith('JJ'):
        return True
    elif pos.startswith('NN'):
        return True
    return False
        
def are_synonyms(word1, word2): 
    data = {}
    #file = "./data/data.pkl"
    
    
    # Connect to the database (or create it)
    conn = sqlite3.connect('my_database.db')
    cursor = conn.cursor()

    # Create a table
    cursor.execute('''CREATE TABLE IF NOT EXISTS synonymslistings
                    (id INTEGER PRIMARY KEY, term TEXT, synonym Text)''')

    
    cursor.execute('''CREATE TABLE IF NOT EXISTS synonyms
                    (id INTEGER PRIMARY KEY, term TEXT, synonym Text)''')


    # Insert data into the table
    cursor.execute("SELECT * FROM synonyms WHERE term=? and synonym=?", (word1,word2))
    result = cursor.fetchone()
    if(result is not None):
        return True
    syn_word1 = get_combined_synonyms(word1)
    syn_word2 = get_combined_synonyms(word2)
    
    found = False
    for sw1 in syn_word1:
        if sw1 in syn_word2:
            found = True
            cursor.execute("SELECT * FROM synonyms WHERE term=? and synonym=?", (word1,sw1))
            result = cursor.fetchone()
            if(result is None):
                cursor.execute("INSERT INTO synonyms (term, synonym) VALUES (?, ?)", (word1,sw1))
            
    if(found):
        cursor.execute("SELECT * FROM synonyms WHERE term=? and synonym=?", (word1,sw1))
        result = cursor.fetchone()
        if(result is None):
            cursor.execute("INSERT INTO synonyms (term, synonym) VALUES (?, ?)", (word1,word2))
    conn.commit()
    cursor.close() 
    conn.close()
    if(found):
        return True 
    return False

def similarWord(sentences, terms):
    synonyms = []
    newTermsList = []
    
    for word in sentences.split(' '):
        if(not isCertainWordType(word)):
            continue
        for term in terms:
            #print("word, term")
            #print(f"{word} {term}")
            word = word.strip()
            term = term.strip()
                
            if(are_synonyms(word, term)):
                #print("synonym term")
                #print(term)
                newTermsList.append(word)
        
                if len(term) == 1:
                    continue
                wildcard_pattern = ".*"+word+".*"  # This pattern matches any term containing "an"

                # Compile the regex pattern for case-insensitive matching
                pattern = re.compile(wildcard_pattern, re.IGNORECASE)

                if pattern.match(term):
                    newTermsList.append(word)
            
    return newTermsList
        
def add_unique_element(array, element): 
    if element not in array: 
        array.append(element) 
    return array

def outputSimilarity(documents):
    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer()

    # Fit and transform the documents to TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(documents)
    # Compute cosine similarity between the sentence and all terms in the array
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Flatten the result and convert to a list
    return cosine_similarities.flatten()
    
def wait_for_file(file_path, check_interval=1):
    while not (os.path.exists(file_path)):
        time.sleep(check_interval)
def bibleVerseCBT(cogdistort):
    data = {
        "All or Nothing Thinking": "Do not judge, or you too will be judged. (Matthew 7:1) - Reminds us to avoid extreme labels and see situations with nuance.",
        "Overgeneralizing": "Romans 8:28: And we know that in all things God works for the good of those who love him, who have been called according to his purpose. This verse reminds us that not every event defines our entire life, and there is a bigger picture at play.",
        "Discounting the Positive": "Do not despise these small beginnings, for the Lord rejoices to see the work begin. (Zechariah 4:10) - Reminds us to acknowledge and appreciate positive aspects of situations.",
        "Jumping to Conclusions": "Proverbs 18:13: He who answers a matter before he hears it, it is folly and shame to him. This verse emphasizes the importance of listening carefully and gathering all the information before forming an opinion.",
        "Mind Reading": "1 Peter 5:7: Cast all your anxiety on him because he cares for you. How it relates to Mind Reading Mind Reading often fuels anxiety as we worry about what others are thinking and how they might react to us. This verse encourages us to surrender our anxieties to God, trusting in His care and protection.",
        "Fortune Telling": "(2 Timothy 1:7). There is also the theme of anxiety concerning the future in Matthew 6:25-34. Essentially, by worrying and imaging a negative future we do not improve the situation. Our thoughts should be on the present. (verse 34).",
        "Magnification Catastrophizing": "Therefore do not worry about tomorrow, for tomorrow will worry about itself. Each day has enough trouble of its own. (Matthew 6:34) - Encourages not to jump to worst-case scenarios.",
        "Emotional Reasoning": "Do not conform to the pattern of this world, but be transformed by the renewing of your mind. (Romans 12:2) - Encourages us to not let emotions dictate our thinking and to actively change our perspective",
        "Should Statements": "Matthew 11:28-30: Come to me, all you who are weary and burdened, and I will give you rest. Take my yoke upon you and learn from me, for I am gentle and humble in heart, and you will find rest for your souls. For my yoke is easy and my burden is light.",
        "Labeling": "For we are God's handiwork, created in Christ Jesus to do good works, which God prepared in advance for us to do. (Ephesians 2:10) - Reminds us of our inherent worth and not to define ourselves by negative labels"
    }
    
    return data[cogdistort]
    
data = pd.read_csv("./data/cogdistort.csv")

cogdistortlabel = data['cognitive_distortions_label']

cogTerms = []
df = pd.DataFrame(data)
 
# Sampling frequency
freq = 44100
 
# Recording duration
duration = 10
toRecord = True
# Start recorder with the given values 
# of duration and sample frequency
file = "./data/recording.wav"
while(True):   
    if(toRecord):
        print("Hit Enter then start speaking. Make sure microphone settings enabled")
        input()
        recording = sd.rec(int(duration * freq), 
                    samplerate=freq, channels=2)
        print("Recording for 10secs... start speaking")
        
        # Record audio for the given number of seconds
        sd.wait()
        
        # This will convert the NumPy array to an audio
        # file with the given sampling frequency
        write(file, freq, recording)
        wait_for_file(file, 1)

    # Load the audio file
    audio = AudioSegment.from_file(file, format="wav")

    # Export the audio file to a supported format
    audio.export(file, format="wav")

    recognizer = sr.Recognizer()

    with sr.AudioFile(file) as source:
        audio = recognizer.record(source)

    try:
        sentences = recognizer.recognize_google(audio)
        print("Transcription:", sentences)
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


    #sentences = "I ought to be able to handle this on my own"
    rating = []
    print("sentences")
    print(sentences)
    for r in df.iterrows():
        #print("r[1]")
        #print(r[1])
        label = r[1]["cognitive_distortions_label"]
        terms = r[1]["terms"].split('-')
        #print("terms")
        #print(terms)
        cogTerms.append({"label": label, "terms": terms})
        similarterms = list(set(similarWord(sentences, terms)))
        print("similarterms")
        print(similarterms)
        terms = list(set(similarterms) | set(terms))
        #print("terms")
        #print(terms)
        #print("sentences")
        #print(sentences)
        documents = terms + [sentences]
        cosine_similarities = outputSimilarity(documents)
        rating.append({"label": label, "terms_score": cosine_similarities})
        
    #high = 0
    print(sorted(rating, key=lambda x: max(x['terms_score'])))

    highestRating=max(rating, key=lambda x: max(x["terms_score"]))
    print("---------------------------------------------")
    print(highestRating['label'])
    #topThreeDistortions = sorted(rating, key=lambda x: (max(x['terms_score']), x if max(x['terms_score']) > 0.0 else 0.0), reverse=True)[:10]
    topDistortions = sorted(rating, key=lambda x: (max(x['terms_score'])), reverse=True)[:10]
    print("Top distortions")
    for d in topDistortions:
        for r in rating:
            if r["label"] == d['label']:        
                if(max(r["terms_score"]) != 0.0):
                    print(d['label'])
                    print("terms_score")
                    print(r['terms_score'])
                    print("Bible verse - Put cognitive distortion on trial - ")
                    print(bibleVerseCBT(d['label']))
