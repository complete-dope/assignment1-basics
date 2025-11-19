# this is used to tokenize the values

from collections import defaultdict
import string
from this import d
from typing import DefaultDict
import regex as re 
import time

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
text = 'some text that Ill pretokenize'

# out = re.finditer(PAT, text)
# print(out.match())
# print(dir(out))


# -- BPE TRAINING EXAMPLE --

dataset = "lowƒ∂ low low low low lower lower widest widest widest newest newest newest newest newest newest" 

# Special tokens
eos = "<|endoftext|>"

## Tokens till now ( 256 + 1 (eos) )
 
# Pre-Tokenization
data = dataset.split(' ')

freq_table = defaultdict(int)
for word in data:
    freq_table[word] += 1 # freq table


# START 
wordBytes_freq_table = defaultdict[tuple[bytes], int](int)
for word in data:
    bytes_array = []
    for chr in word:
        bytes_array.append(chr.encode('utf-8'))

    wordBytes_freq_table[tuple(bytes_array)] += 1

# print(wordBytes_freq_table) # defaultdict(<class 'int'>, {(b'l', b'o', b'w', b'\xc6\x92', b'\xe2\x88\x82'): 1, (b'l', b'o', b'w'): 4, (b'l', b'o', b'w', b'e', b'r'): 2, (b'w', b'i', b'd', b'e', b's', b't'): 3, (b'n', b'e', b'w', b'e', b's', b't'): 6})

# import sys ; sys.exit(0)


# Doing Merges 
for _ in range(6): # max 10 only  
    merges_table = defaultdict(int)
    for bytes_tuples,freq in wordBytes_freq_table.items():
        for i in range(len(bytes_tuples)-1):
            merges_table[(bytes_tuples[i], bytes_tuples[i+1])] += 1*freq
        
    merges_list = sorted(merges_table.items(), key=lambda item:item[1], reverse =True)
    # print('\n\n', merges_list)
    # import sys ; sys.exit(0)


    # get the top one from this
    merge_chars = merges_list[0][0]
    merge_chars_count = merges_list[0][1]

    if merge_chars_count == 1: 
        break # no more merges required 

    new_freq_table = defaultdict(int) 
    for wordBytes, freq in wordBytes_freq_table.items():
        i=0
        newWordBytes = []
        while(i < len(wordBytes)):
            if i+1<len(wordBytes):
                if (wordBytes[i], wordBytes[i+1]) == merge_chars:
                    newWordBytes.append(wordBytes[i] + wordBytes[i+1])
                    i+=1
                else:
                    newWordBytes.append(wordBytes[i]) 

            
            else:
                newWordBytes.append(wordBytes[i]) 
            
            i+=1   

        new_freq_table[tuple(newWordBytes)] = freq

    wordBytes_freq_table = new_freq_table


# print("\n\nFinal\n",new_freq_table)


# use python's multiprocessing library in this and also use re.finditer

# read the file from data 
import os 
VALIDATION_FILE = '../data/TinyStoriesV2-GPT4-valid.txt'
validation_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) ,VALIDATION_FILE) 

# split this
data = ''
with open(validation_file_path , 'r', encoding = 'utf-8') as f:
    # f here is the file-path 
    data += f.read()

print('\n----\n')
print(type(data))
print('string data is : ', data[:5] + '\n\n')

#  we note that the merging part of BPE training is not parallelizable in Python.

import re
special_tokens = ["<|endoftext|>"]

# Build a regex-safe OR pattern for all special tokens
pattern = "|".join(re.escape(t) for t in special_tokens) # take in consideration all the special tokens

# Split on the special tokens
data_parts = re.split(pattern, data)

print(data_parts[:2])



class BPE_Trainer:
    def __init__(self , input_path:str, vocab_size :int, special_tokens: list[str]= []) -> None:
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

    def _pretokenization(self, FILE = False, text_data=''):
        if FILE:
            validation_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) ,VALIDATION_FILE) 
            data = ''
            with open(validation_file_path , 'r', encoding = 'utf-8') as f:
                # f here is the file-path 
                data = f.read()

        else:
            data = text_data
        
        # split this
        if self.special_tokens:
            pattern = "|".join([re.escape(t) for t in self.special_tokens]) # split based on the special tokens
            chunked_data = re.split(pattern, data) # Nx paragraphs 
        else:
            chunked_data = [data]

        print('chunk-data is ', chunked_data)
            
        # now pretokenize this data ( based on gpt-2 regex pattern )
        GPT2_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        pretokenized_data = []
        for string_para in chunked_data:
            print('string para is ', string_para)
            import regex as re
            pretokenized_data += re.findall(GPT2_REGEX , string_para)
        print('pd is ', pretokenized_data)

        # pretokenized_data = []
        # for string_para in chunked_data:
        #     words = string_para.split(' ')
        #     for word in words:
        #         pretokenized_data.append(word)
            
        # pretokenized_data = [word.split(' ') for chunk_data in chunked_data for word in chunk_data ]

        return pretokenized_data

    def train(self) -> (dict[int, bytes] , list[tuple[bytes]]):
        pretokenized_data = self._pretokenization(FILE = True)
        
        starting_range = 256 + len(self.special_tokens)
        byte_merges = self.vocab_size - starting_range # this tells total no. of merges to do 
        
        wordBytes_freq_table = defaultdict[tuple[bytes], int](int)
        for word in pretokenized_data:
            bytes_array = []
            for chr in word:
                bytes_array.append(chr.encode('utf-8'))

            wordBytes_freq_table[tuple(bytes_array)] += 1

        times = []
        vocab:dict[int, bytes] = {} 
        merges:list[bytes,bytes] = [] 

        for i in range(256):
            vocab[i] = bytes([i])

        for idx,st in enumerate(self.special_tokens):
            vocab[256+idx] = st


        for idx in range(byte_merges): #  limited merges only  
            print('Starting merge : ', idx)
            merges_table:defaultdict[tuple[bytes], int] = defaultdict(int)
            bytes_to_word:defaultdict[tuple[bytes], list[str]] = defaultdict(list)
            
            for bytes_tuples,freq in sorted(wordBytes_freq_table.items()):
                for i in range(len(bytes_tuples)-1):
                    merges_table[(bytes_tuples[i], bytes_tuples[i+1])] += 1*freq
                    bytes_to_word[(bytes_tuples[i], bytes_tuples[i+1])] += [bytes_tuples]
                
            merges_list:list[tuple(tuple[bytes], int)] = sorted(
                merges_table.items(), 
                key=lambda item: (-item[1], tuple(b.decode('utf-8') for b in item[0]))
            )
            # print('\n\n', merges_list)
            # import sys ; sys.exit(0)


            # get the top one from this
            # print('merges list first one is ', merges_list)
            merge_chars = merges_list[0][0]
            merge_chars_count = merges_list[0][1]

            # add this to vocab_table
            vocab[starting_range] = merge_chars[0] + merge_chars[1]
            starting_range +=1
            merges.append(merge_chars)

            if merge_chars_count == 1: 
                break # no more merges required 
                
            start_time = time.time()
            wordBytes_freq_table = self._cached_bpe_merging(bytes_to_word, wordBytes_freq_table, merge_chars)
            
            # wordBytes_freq_table = self._naive_bpe_merging(wordBytes_freq_table, merge_chars)
            end_time = time.time()

            times.append(end_time - start_time)
            
        print('avg time', sum(times)/len(times))
        self.vocab, self.merges = vocab, merges
        return vocab, merges

    # 10x times faster
    def _cached_bpe_merging(self, bytes_to_word, wordBytes_freq_table, merge_chars):
        
        bytes_tuples_to_change = bytes_to_word[merge_chars] # list of bytes tuples to update             
        
        for byte_tuple in bytes_tuples_to_change: # only change limited values 
            occ_freq = wordBytes_freq_table[byte_tuple]
            
            new_byte_tuple = []
            i=0
            while(i<len(byte_tuple)):
                if i+1 < len(byte_tuple) and (byte_tuple[i], byte_tuple[i+1]) == merge_chars:
                    new_byte_tuple.append(byte_tuple[i] + byte_tuple[i+1])
                    i+=1
                else:
                    new_byte_tuple.append(byte_tuple[i])
                i+=1
            
            # print('new byte tuples : ', new_byte_tuple)
            wordBytes_freq_table[tuple(new_byte_tuple)] = occ_freq # add a new key
            del wordBytes_freq_table[byte_tuple] # delete this key from dict 

        return wordBytes_freq_table


    def _naive_bpe_merging(self,wordBytes_freq_table, merge_chars):        
        new_freq_table = defaultdict[tuple[bytes], int](int)
        for wordBytes, freq in wordBytes_freq_table.items():
            i=0
            newWordBytes = []
            while(i < len(wordBytes)):
                if i+1<len(wordBytes):
                    if (wordBytes[i], wordBytes[i+1]) == merge_chars:
                        newWordBytes.append(wordBytes[i] + wordBytes[i+1])
                        i+=1
                    else:
                        newWordBytes.append(wordBytes[i]) 
                
                else:
                    newWordBytes.append(wordBytes[i]) 
                
                i+=1   

            new_freq_table[tuple(newWordBytes)] = freq

        return new_freq_table


    def encoder(self, text: str, vocab = None):
        
        if getattr(self, 'vocab', None) is None:
            if vocab is None:
                raise ValueError('provide the vocab or train it ')        
        else: 
            vocab = self.vocab

        # reverse vocab
        rvocab = defaultdict(int)
        for i,j in vocab.items():
            rvocab[j] = i # bytes to its integer value
        
        # pretokenize
        pretokenized_data = self._pretokenization(text_data = text)

        text_bytes = []
        # first convert to byte pairs
        wordBytes_to_freq:defaultdict[tuple[bytes], int] = defaultdict(int)
        for word in pretokenized_data:
            wordbyte = []
            for chr in word:
                wordbyte.append(chr.encode('utf-8'))
            
            # wordBytes_to_freq[wordbyte] += 1
            text_bytes.append(tuple(wordbyte)) # [(b'a', b'n'), (b'c', b'a' , b't')]

        def _word_wise_merge(word:tuple[bytes]) -> list[int]:
            # for i in range(len(word)):
            merge_found = True
            while(merge_found):
                updated_word = []
                merge_found = False
                i =0
                while i < len(word):
                    if merge_found is False and i+1<len(word) and rvocab[(word[i],word[i+1])] > 0: # then a potential match
                        updated_word.append(word[i] + word[i+1])
                        i+=1
                        merge_found = True
                        
                    else:
                        updated_word.append(word[i])
                    i+=1

                word = tuple(updated_word)
            
            # convert to its integer mapping 
            output_token = []
            for merged_bytes in word:
                output_token.append(rvocab[merged_bytes])

            return output_token

        final_out = []
        for word in text_bytes:
            final_out.append(_word_wise_merge(word))

        return [x for sub in final_out for x in sub]
 

    def decoder(self, x:list[int], vocab=None):
        # x : [120, 256, 42, 312, 542]
        
        if getattr(self, 'vocab',None) is None:
            if vocab is None:
                raise ValueError('provide the vocab or train it ')        
        else: 
            vocab = self.vocab
            
        output = ''
        for val in x:
            if type(vocab[val]) is bytes:
                char_str = vocab[val].decode('utf-8')
            
            output += char_str #vocab[val]

        return output
        

vocab:dict[int, bytes] = {} 
merges:list[bytes,bytes] = [] 

for i in range(256):
    vocab[i] = bytes([i])

output = BPE_Trainer(input_path = validation_file_path, vocab_size=256).encoder(text = 'I am here', vocab = vocab)
print(output)

# vocab, merges =  BPE_Trainer(input_path = validation_file_path, vocab_size = 256+1+300 , special_tokens=['<|endoftext|>']).train()

# print(vocab)
# print('\n\n\n')
# print(merges)