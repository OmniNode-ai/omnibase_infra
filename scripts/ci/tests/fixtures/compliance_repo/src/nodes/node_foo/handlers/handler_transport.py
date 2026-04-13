import httpx

def fetch(url):
    return httpx.get(url)
