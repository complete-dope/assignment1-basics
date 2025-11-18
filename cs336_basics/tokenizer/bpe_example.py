# this is used to tokenize the values

from collections import defaultdict
import string
import regex as re 

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

print(wordBytes_freq_table) # defaultdict(<class 'int'>, {(b'l', b'o', b'w', b'\xc6\x92', b'\xe2\x88\x82'): 1, (b'l', b'o', b'w'): 4, (b'l', b'o', b'w', b'e', b'r'): 2, (b'w', b'i', b'd', b'e', b's', b't'): 3, (b'n', b'e', b'w', b'e', b's', b't'): 6})

# import sys ; sys.exit(0)


# Doing Merges 
for _ in range(6): # max 10 only  
    merges_table = defaultdict(int)
    for bytes_tuples,freq in wordBytes_freq_table.items():
        for i in range(len(bytes_tuples)-1):
            merges_table[(bytes_tuples[i], bytes_tuples[i+1])] += 1*freq
        
    print(merges_table)
    # import sys ; sys.exit(0)

    merges_list = sorted(merges_table.items(), key=lambda item:item[1], reverse =True)
    print('\n\n', merges_list)
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


print("\n\nFinal\n",new_freq_table)


# use python's multiprocessing library in this and also use re.finditer
