from django.shortcuts import render
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from rest_framework.viewsets import ViewSet
from rest_framework.response import Response
from rest_framework.status import HTTP_200_OK
import heapq



class Summarizer(ViewSet):

    def __init__(self):
        self.text = ''
        pass

    def summarize_code(self,request):
        self.text = request.POST['summary_text']
        stop_words = set(stopwords.words("english"))
        freq_words_table = dict()
        for word in word_tokenize(text=self.text):
            if word in stop_words:
                continue
            elif word in freq_words_table.keys():
                freq_words_table[word] += 1
            else:
                freq_words_table[word] = 1
        max_freq = max(freq_words_table.values())

        for word in freq_words_table.keys():
            freq_words_table[word] = (freq_words_table[word]/max_freq)

        sent_score = dict()
        for sentence in sent_tokenize(text=self.text):
            for word in word_tokenize(text=self.text):
                if word in freq_words_table.keys():
                    if len(sentence.split(' ')) < 30:
                        if sentence in sent_score.keys():
                            sent_score[sentence] += freq_words_table[word]
                        else:
                            sent_score[sentence] = freq_words_table[word]


        summary_sentences = heapq.nlargest(7,sent_score,key=sent_score.get)
        summary = ''.join(summary_sentences)
        return Response({'abstract':summary},HTTP_200_OK)

    def download_dependencies(self,request):
        import nltk
        nltk.download("stopwords")
        nltk.download("punkt")

