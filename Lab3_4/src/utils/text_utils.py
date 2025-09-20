import re
URL_RE = re.compile(r"http\S+|www\S+|https\S+")
MENTION_RE = re.compile(r"@[\w_]+")
HASH_RE = re.compile(r"#\w+")
SPACES_RE = re.compile(r"\s+")
def clean_text(text: str, remove_urls=True, remove_hashtags=True, remove_mentions=True, to_lower=True) -> str:
    if remove_urls: text = URL_RE.sub("", text)
    if remove_mentions: text = MENTION_RE.sub("", text)
    if remove_hashtags: text = HASH_RE.sub("", text)
    text = SPACES_RE.sub(" ", text).strip()
    return text.lower() if to_lower else text
