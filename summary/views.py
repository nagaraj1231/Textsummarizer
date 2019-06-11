from django.shortcuts import render
from pathlib import Path
import nltk
from .magic import methods
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.cluster.util import cosine_distance
import networkx as nx
from rest_framework.viewsets import ViewSet
from rest_framework.response import Response
from rest_framework.status import HTTP_200_OK
import numpy as np
import heapq , csv, os, pandas as pd, operator
from pathlib import Path



class Summarizer(ViewSet):

    def __init__(self):
        self.text = ''
        self.input_paragraph = ''
        pass

    def summarize_code(self,request):
        self.text = request.POST['summary_text']
        extraction = request.POST['no_lines'] if request.POST['no_lines'] is not '' else 4
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


        summary_sentences = heapq.nlargest(int(extraction),sent_score,key=sent_score.get)
        summary = ''.join(summary_sentences)

        sd = Sentiment_analysis()
        sd.prepare_dataset(summary)
        return Response({'abstract':summary},HTTP_200_OK)

    def second_summarizer(self):
        sentences = self.input_paragraph.split(". ")
        article = list(map(lambda u: u.replace("[^a-zA-Z]"," ").split(" "),sentences))
        article.pop()
        return article

    def generate_summary(self,request):

        self.input_paragraph = request.POST['summary_text']
        extraction = request.POST['no_lines'] if request.POST['no_lines'] is not '' else 4
        # Get the sopwords in English language          ---- M K N
        self.stop_words = stopwords.words("english")

        self.sentence = self.second_summarizer()
        #  Generate similarity matrix                   ----- M K N
        simi_matrix = self.similarity_matrix()

        # Generate similarity matrix from the graph using numpy     ---- M K N
        similar_matrix_graph = nx.from_numpy_array(simi_matrix)
        # Generate rank for the matrix                  ---- M K N
        total_score = nx.pagerank(similar_matrix_graph)
        # assign score for the sentence                 ---- M K N
        rank_sentence = sorted(((total_score[i],s) for i,s in enumerate(self.sentence)),reverse=True)
        # rank_sentence = sorted(list(map(lambda x,y: (total_score[x],y) ,list(l for l in range(len(self.sentence))),self.sentence)))
        summarized_text = list()
        for i in range(int(extraction)):
            summarized_text.append(" ".join(rank_sentence[i][1]))

        # summarized_text = ". ".join(list(map(lambda g: " ".join(g[1]),rank_sentence)))
        sd = Sentiment_analysis()
        sd.prepare_dataset(" ".join(summarized_text))
        return Response({'abstract': summarized_text}, HTTP_200_OK)

    def similarity_matrix(self):
        similarity = np.zeros(shape=(len(self.sentence),len(self.sentence)))
        for x in range(len(self.sentence)):
            for y in range(len(self.sentence)):
                if x == y:
                    continue
                similarity[x][y] = self.similar_sentence(sent1=self.sentence[x],sent2=self.sentence[y])

        return similarity

    def similar_sentence(self,sent1,sent2):
        if self.stop_words is None:
            self.stop_words = list()

        sent1 = list(map(lambda d: d.lower(),sent1))
        sent2 = list(map(lambda m: m.lower(),sent2))

        total_sentence = list(set(sent1+sent2))

        vect1 = [0] * len(total_sentence)
        vect2 = [0] * len(total_sentence)

        for e in sent1:
            if e in self.stop_words:
                continue
            vect1[total_sentence.index(e)] += 1

        for m in sent2:
            if m in self.stop_words:
                continue
            vect2[total_sentence.index(m)] += 1

        return 1 - cosine_distance(vect1,vect2)

    def download_dependencies(self):
        import nltk
        nltk.download("stopwords")
        nltk.download("punkt")

class Sentiment_analysis(ViewSet):
    def __int__(self):
        super().__init__()

    def prepare_dataset(self,summarized_text):
        upload_path = str(Path('./media/dataset').resolve())
        upload_file = str(Path('./media/dataset/summary.tsv').resolve())
        if not os.path.isdir(upload_path):
            os.mkdir(upload_path)

        if os.path.exists(upload_file) and os.path.isfile(upload_file):
            with open(upload_file, "r") as tsvfile:
                tsvread = csv.reader(tsvfile,delimiter="\t")
                existing_bytes = list(map(lambda c: c,tsvread))
                tsvfile.close()

        try:
            with open(upload_file, "wt") as tsvfile:
                tsv = csv.writer(tsvfile, delimiter="\t")
                y = int(existing_bytes[-2][1]) + 1
                existing_bytes.append(['ID',y])
                existing_bytes.append(["Text", summarized_text])
                for k, row in enumerate(existing_bytes):
                    tsv.writerow([row[0],row[1]])
                tsvfile.close()

            pass
        except NameError:
            with open(upload_file, "wt") as tsvfile:
                tsv = csv.writer(tsvfile, delimiter="\t")
                tsv.writerow(["ID","1"])
                tsv.writerow(["Text",summarized_text])
                tsvfile.close()
