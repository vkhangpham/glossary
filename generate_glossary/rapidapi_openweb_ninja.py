import requests, json, os
from dotenv import load_dotenv

load_dotenv()

WEB_SEARCH_URL = "https://google-api31.p.rapidapi.com/websearch"
IMAGE_SEARCH_URL = "https://google-api31.p.rapidapi.com/imagesearch"

headers = {
    "content-type": "application/json",
    "X-RapidAPI-Key": os.environ["RAPID_API_KEY"],
    "X-RapidAPI-Host": "real-time-web-search.p.rapidapi.com",
}


def web_search(query, max_results=3):
    url = "https://real-time-web-search.p.rapidapi.com/search"
    querystring = {"q": query, "limit": f"{max_results}"}

    response = requests.get(url, headers=headers, params=querystring)
    return response.json()


def web_search_bulk(queries, max_results=3):
    url = "https://real-time-web-search.p.rapidapi.com/search"
    payload = {"queries": queries, "limit": f"{max_results}"}
    response = requests.post(url, json=payload, headers=headers)

    return response.json()
