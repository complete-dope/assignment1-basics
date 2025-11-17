def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

# output = decode_utf8_bytes_to_str_wrong("hello ß".encode('utf-8'))
# print(output)


unicode_list = []
string = ''
for i in "hello world ∂".encode('utf-8'):
    string += bytes([i]).decode('utf-8')
    unicode_list.append(i)

print(string)
