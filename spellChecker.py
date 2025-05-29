# import regular expressions
import re

# import numpy
import numpy as np 

# Dict subclass for counting hashable items.
from collections import Counter 

# import the string class
import string

# import pickle to use binary files
import pickle as pk

def obtainWordOccurrences(inputText):
    # to ensure that the word will be found in the dataset use the lowercase
    lowercaseText = inputText.lower()

    # find all the unique words and return them 
    # \w is the regular expression that matches any word character
    # + is a quantifier that matches one or more occurrences
    # r'\w+' will match one or more consecutive word characters
    # therefore this will return a list of all words found in the text
    return re.findall( r'\w+', lowercaseText)

def wordProbabilityCalculation(dictionary, inputWord):
    # get the amount of times the word was found within the text
    occurrences = dictionary[inputWord]

    # get the number of unique words in the training set
    numberOfWords = sum( dictionary.values() )

    # determine the probability 
    wordProbability = occurrences/numberOfWords

    return wordProbability

def validWords(dictionary, inputText):
    valid = []
    # loop through the dictionary and check if the words are found within the dictionary
    for word in inputText:
        # if the word is found within the dictionary append
        if word in dictionary:
            valid.append(word)
    
    return valid

def split( inputword, alphabet):
    # split the input word into two halves
    return [ (inputword[:i], inputword[i:])    for i in range( len(inputword) + 1) ]

def delete(inputWord, splitWord, validCharacters):
    # in the right half of the word delete one character
    return [ wordLeftHalf + wordRightHalf[1:]               for wordLeftHalf, wordRightHalf in splitWord if wordRightHalf ]

def transpose(inputWord, splitWord, validCharacters):
    # in the right half transpose two characters
    return [ wordLeftHalf + wordRightHalf[1] + wordRightHalf[0] + wordRightHalf[2:] for wordLeftHalf, wordRightHalf in splitWord if len(wordRightHalf)>1 ]

def replace(inputWord, splitWord, validCharacters):
    # going through the alphabet replace a single character in the right half
    return [ wordLeftHalf + c + wordRightHalf[1:]           for wordLeftHalf, wordRightHalf in splitWord if wordRightHalf for c in validCharacters ]

def insert(inputWord, splitWord, validCharacters):
    # going through the alphabet add a single character in the middle of the word
    return [ wordLeftHalf + c + wordRightHalf               for wordLeftHalf, wordRightHalf in splitWord for c in validCharacters ]


def mutate(inputWord):
    # the mutate function mutates only one character
    validCharacters = 'abcdefghijklmnopqrstuvwxyz'

    # first we split the word into two halves
    splitWord = split(inputWord, validCharacters)

    # then we start mutating and getting various different spellings
    insertion = insert(inputWord, splitWord, validCharacters)
    deletion = delete(inputWord, splitWord, validCharacters)
    replacement = replace(inputWord, splitWord, validCharacters)
    transposition = transpose(inputWord, splitWord, validCharacters)

    # concatenate these mutations
    concat = insertion + deletion + replacement + transposition

    # next we return these mutations
    return concat

def mutate2(inputWord):
    # the mutate2 function mutates two characters and uses mutate()
    secondMutation = (mutation2 for mutation1 in mutate(inputWord) for mutation2 in mutate(mutation1))

    return secondMutation

def wordVariations(dictionary, inputWord):
    # generate the spelling mutations that are possible for the input word
    word1 = validWords(dictionary, [inputWord])
    # next mutate the word once and check validity
    word2 = validWords(dictionary, mutate(inputWord) )
    # next mutate the words again with edit length of two
    word3 = validWords(dictionary, mutate2(inputWord) )
    # last word is the set of the original if none are valid
    word4 = [inputWord]

    # return possible words that need to be checked for most probable word
    return (word1 or word2 or word3 or word4)


def spellingCorrection(dictionary, inputWord):
    # # calculate the probability of each word 
    probableWords = wordVariations(dictionary, inputWord)

    # create probability array
    probabilityArray = np.zeros(len(probableWords))

    counter = 0

    # determine the probability of each word 
    for i in probableWords:
        probabilityArray[counter] = wordProbabilityCalculation(dictionary, i)
        # print(i)
        # print(probabilityArray[counter] )
        counter += 1

    # take the word with the highest probability
    return probableWords[np.argmax(probabilityArray)]

def checkPunctuation(originalWord, correctedWord):
    # first we check if the first word is capitalized 
    if originalWord.istitle():
        correctedWord = correctedWord.capitalize()

    # next check if there is punctuation in the last position and add it back in
    if originalWord[-1] in string.punctuation:
        correctedWord += originalWord[-1]

    return correctedWord


def spellCheck(inputSentence, train=False, filename='trainingText.txt'):
    # get the training text
    if train:
        trainingText = open(filename).read()
        # get the occurrences of each word in the text
        occurrences = obtainWordOccurrences(trainingText)
        #  Save these values in the counter dict
        wordCount = Counter(occurrences)
        # Store the counter dict for future use
        with open('counter_dict.pkl', 'wb') as f:
            pk.dump(wordCount, f)
    else:
        # Load the counter dict if not training
        with open('counter_dict.pkl', 'rb') as f:
            wordCount = pk.load(f)

    # now start processing the words that have been classified
    # split the string into words
    wordArray = inputSentence.split(' ')

    # corrected Array
    correctedArray = []

    for i in wordArray:
        # determine the most probable spelling and append
        correctedWord = spellingCorrection(wordCount,i)

        # next re-add all the punctuation
        addedPunctuation = checkPunctuation(i,correctedWord)

        # append the new Word
        correctedArray.append( addedPunctuation )

    # create string from Array
    return ' '.join(correctedArray)

