import torch

from src.data.loader import build_graph_from_event

torch.manual_seed(42)


def test_build_graph_with_full_csv_row():
    row = {
        "news_id": "test123",
        "title": "Example News",
        "url": "http://example.com",
        "publish_date": "2023-01-01",
        "source": "Twitter",
        "text": "This is a news content",
        "labels": "true",
        "n_tweets": "3",
        "n_retweets": "2",
        "n_replies": "1",
        "n_users": "3",
        "tweet_ids": "1001,1002,1003",
        "retweet_ids": "1002,1003",
        "reply_ids": "",
        "user_ids": "2001,2002,2003",
        "retweet_relations": "1002-1001-2002-2001,1003-1001-2003-2001",
        "reply_relations": "",
        "data_name": "politics",
    }
    data = build_graph_from_event(**row)
    assert data is not None
    assert data.x.shape == (3, 64)
    assert data.edge_index.shape[0] == 2
    assert data.y.item() == 1


def test_filter_single_tweet_event():
    row = {
        "labels": "false",
        "n_tweets": "1",
        "tweet_ids": "1001",
        "retweet_relations": "",
        "reply_relations": "",
    }
    data = build_graph_from_event(**row)
    assert data is None


def test_empty_relations():
    row = {
        "labels": "true",
        "n_tweets": "2",
        "tweet_ids": "1001,1002",
        "retweet_relations": "",
        "reply_relations": "",
    }
    data = build_graph_from_event(**row)
    assert data is not None
    assert data.edge_index.shape[1] == 0

