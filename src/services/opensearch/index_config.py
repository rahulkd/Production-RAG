ARXIV_PAPERS_INDEX = "arxiv-papers"

# Index mapping configuration for arXiv papers
ARXIV_PAPERS_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "standard_analyzer": {"type": "standard", "stopwords": "_english_"},
                "text_analyzer": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "stop", "snowball"]},
            }
        },
        ## what does text analyzer do? It is a custom analyzer that uses the standard tokenizer and applies lowercase, stop word removal, 
        # and snowball stemming filters to the text. This helps improve search relevance by normalizing the text and reducing it to its 
        # base form.
        ## what is snowball stemming? Snowball stemming is a process of reducing words to their base or root form. 
        # It is a more aggressive form of stemming than the Porter stemmer and is designed to handle a wider range of languages. 
        # The snowball filter in Elasticsearch applies this stemming process to the text, which can help improve search relevance by 
        # allowing different forms of a word to match the same search query.
    },
    ## what does these mappings do? The mappings define the structure of the documents that will be indexed in Elasticsearch.
    ## "arxiv_id": {"type": "keyword"}, this means that the arxiv_id field will be treated as a keyword, 
    # which is a type of data that is not analyzed and is stored as-is.
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "arxiv_id": {"type": "keyword"},
            "title": {
                "type": "text",
                "analyzer": "text_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "authors": {
                "type": "text",
                "analyzer": "standard_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "abstract": {"type": "text", "analyzer": "text_analyzer"},
            "categories": {"type": "keyword"},
            "raw_text": {"type": "text", "analyzer": "text_analyzer"},
            "pdf_url": {"type": "keyword"},
            "published_date": {"type": "date"},
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"},
        },
    },
}
