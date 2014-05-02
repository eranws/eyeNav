from httplib2 import Http

h = Http()

resp, content = h.request("http://localhost:8889/a/message/new", "POST", "up")
