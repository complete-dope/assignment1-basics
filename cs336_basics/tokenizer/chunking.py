# loading / reading a file in chunks 
# delimiter aware stream parser 
# delimiter : A delimiter is a character or sequence of characters used to separate distinct regions of data

import os
import io 
from typing import Generator


VALIDATION_FILE = '../data/TinyStoriesV2-GPT4-valid.txt'
validation_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) ,VALIDATION_FILE) 
CHUNK_SIZE = 10 # 1 Kb

def text_in_bytes(file_path): # generator function now my goal to read from this buffer 
    with open(file_path, 'rb') as f: # this 'rb' converts a file to bytes if not 
        file_buffer = io.BytesIO(f.read())
        yield file_buffer # bytes object


# read the file in streaming mode 

byte_array = bytearray(CHUNK_SIZE) # since this size is fixed we are not loading any extra memory 
def text_in_bytes_buffered(file_path): # generator function now my goal to read from this buffer 
    with open(file_path, 'rb') as f: # this 'rb' converts a file to bytes if not 
        while True:
            n_bytes = f.readinto(byte_array) # this tells the actual no. of bytes read 

            if n_bytes == 0:
                break 
            
            yield byte_array[:n_bytes]

            # file_buffer = io.BufferedIOBase().readinto(byte_array)
            # yield file_buffer # bytes object


# for i in text_in_bytes_buffered(validation_file_path):
#     print(i.decode('utf-8'))
#     print(f'So this seems like we loaded {CHUNK_SIZE} no. of bytes in our memory !')

# del byte_array # to avoid memory leak !


# --- Configuration ---
CHUNK_SIZE = 250  # Size of raw reads from the file

def read_file_chunked(file_path):
    """
    1. Generator that reads the file in raw byte chunks (like our previous example).
    """
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            yield chunk


def read_up_to_last_delimiter(file_path, delimiter: bytes, chunk_size: int = CHUNK_SIZE)-> Generator:

    # Initialize the buffer with any leftover bytes from the previous cycle
    buffer = b''
    
    with open(file_path, 'rb') as f:
        while True:
            # Read a large chunk (e.g., 1MB)
            new_chunk = f.read(chunk_size)
            
            # --- EOF Check and Final Yield ---
            if not new_chunk:
                # If we hit End of File, yield any incomplete data left in the buffer
                if buffer:
                    yield buffer
                break # Exit the main loop
                
            # Combine current buffer with the newly read chunk
            buffer += new_chunk
            
            # --- Reverse Search for Delimiter ---
            # rfind() searches from the end of the byte string backward
            last_index = buffer.rfind(delimiter)
            
            if last_index != -1:
                # Delimiter Found:
                
                # 1. Get the data to yield (from start up to the delimiter's position)
                data_to_yield = buffer[:last_index]
                
                # 2. Update the buffer to only contain the remainder
                # The remainder starts immediately after the delimiter ends.
                buffer = buffer[last_index + len(delimiter):]
                
                # 3. Yield the complete, large chunk
                yield data_to_yield
                
            # If last_index == -1, the loop continues, reading another chunk
            # into the buffer, waiting to form a complete segment that contains the delimiter.


    del buffer # avoiding memory leak 

if __name__ == '__main__':
    
    file_path = VALIDATION_FILE
    iterator = read_up_to_last_delimiter(file_path, delimiter=b'<|endoftext>')

    iters = 0
    for i in iterator:
        print(i)
        print('\n\n')
        iters += 1

        if iters > 2:
            break


