# Semantic-Search-Engine

## Objective:
Develop an advanced search engine algorithm that efficiently retrieves subtitles based on user queries, with a specific emphasis on subtitle content. The primary goal is to leverage natural language processing and machine learning techniques to enhance the relevance and accuracy of search results.

## Technologies Used

Python 
Flask
HTML



## Semantic Search Engines: 
Semantic search engines go beyond simple keyword matching to understand the meaning and context of user queries and documents.

## Steps:

1) The web scrapped data is pre-processed and converted into a pandas Data Frame
2) The subtitle data is converted into Embedding Vectors
3) Since the dimensions of vectors is below 1000, chunking is applied to to counter the loss of information
4) Cosine similarity is used as a metic to for the best result
5) The whole code is deployed on a local server with the help of Flask

