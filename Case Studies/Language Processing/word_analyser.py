from collections import Counter

text = "This, is, some text stuff that is to test text."

def count_words(text):
    """counts the number of occurances of each word in a text string skipping punctuation"""
    
    word_counts = {}
    text = text.lower()
    
    #define chars to skip - punctuation etc
    skips = [".", ",", ";", ":", "'", '"', "?"]
    
    #loop over text and replace unwanted chars
    for i in text:
        if i in skips:
            text = text.replace(i, "")
    
    text = text.split(" ")
    
    for word in text:
        #if word is known in dict already
        if word in word_counts:
            word_counts[word] += 1
        #if word is unknown
        else:
            word_counts[word] = 1
    return word_counts


def count_words_fast(text):
    """counts the number of occurances of each word in a text string skipping punctuation"""

    text = text.lower()
    
    #define chars to skip - punctuation etc
    skips = [".", ",", ";", ":", "'", '"', "?"]
    
    #loop over text and replace unwanted chars
    for i in text:
        if i in skips:
            text = text.replace(i, "")
    
    text = text.split(" ")
    
    word_counts = Counter(text)
    
    
    return word_counts

def read_book(title_path):
    """
    read a book and dreturn it as a string
    """
    with open(title_path, "r", encoding="utf8") as current_file:
        text = current_file.read()
        text = text.replace("\n", "").replace("\r", "")
    return text


def word_stats(word_counts):
    """
    return number of unique words and their frequencies
    """
 
    num_unique = len(word_counts) #returns the number of unique keys in the dict
    counts = word_counts.values() #a list of the counts, just counts
    return (num_unique, counts) # returns a tuple


#reading multiple files in
import os
book_dir = ("C:/Users/Flat J/Documents/Programming/Python/Python for Research/Case Studies/Language Processing/Books")

import pandas as pd
stats = pd.DataFrame(columns = ("language", "author", "title", "length", "unique")) #creates empty dataframe with the defined colums
title_num = 1 #keeps track of row number

for language in os.listdir(book_dir):
    #each language has a folder in the book_dir
    #each language is organised by author
    for author in os.listdir(book_dir + "/" + language): #the file location in the listdir is given by concatinating strings
        for title in os.listdir(book_dir + "/" + language + "/" + author): #concatinates author on
            input_file = book_dir + "/" + language + "/" + author + "/" + title
            print(input_file)
            text = read_book(input_file)
            (num_unique, counts) = word_stats(count_words(text))
            
            stats.loc[title_num] = language, author.capitalize(), title.replace(".txt", ""), sum(counts), num_unique
            title_num += 1

import matplotlib.pyplot as plt
plt.plot(stats.length, stats.unique, "bo")
