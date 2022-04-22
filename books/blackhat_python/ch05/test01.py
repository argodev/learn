import httplib

conn = httplib.HTTPSConnection('nostarch.com')
conn.request('GET', '/')
response = conn.getresponse()
print(response.status, response.reason)
data = response.read()
print(data)


