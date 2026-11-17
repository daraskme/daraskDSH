"""DDGS web search and indexed public X post search, one JSON request per process."""
import json
import re
import sys
from urllib.parse import urlsplit, urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError
from lxml import html

from ddgs import DDGS
from ddgs.exceptions import DDGSException


def public_post(url):
    try:
        parsed = urlsplit(url)
        return (parsed.scheme == 'https' and parsed.hostname in
                {'x.com', 'www.x.com', 'twitter.com', 'www.twitter.com', 'mobile.twitter.com'}
                and re.fullmatch(r'/(?:[\w]+|i/web)/status/\d+/?', parsed.path))
    except ValueError:
        return False


def realtime_posts(query, limit):
    url = 'https://search.yahoo.co.jp/realtime/search?' + urlencode({'p': query})
    with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=7) as response:
        document = html.fromstring(response.read(4 * 1024 * 1024))
    data = json.loads(document.xpath('string(//script[@id="__NEXT_DATA__"])'))
    timeline = data['props']['pageProps']['pageData']['timeline']['entry']
    posts = {}
    for tweet in timeline:
        url = tweet.get('url', '').split('?', 1)[0]
        if public_post(url):
            text = re.sub(r'\t(?:START|END)\t', '', tweet.get('displayTextBody') or tweet.get('displayText', ''))
            posts[url] = {'url': url, 'title': tweet.get('name', '') + ' (@' + tweet.get('screenName', '') + ')', 'snippet': text[:700]}
    return {'sources': list(posts.values())[:limit], 'truncated': len(posts) > limit}


def search(query, limit, x_only):
    if x_only:
        try:
            recent = realtime_posts(query, limit)
            if recent['sources']:
                return recent
        except (URLError, TimeoutError, OSError, ValueError, KeyError, TypeError):
            pass
    client = DDGS(timeout=7)
    searches = [query] if not x_only else [f'site:x.com {query}', f'site:twitter.com {query}']
    sources = {}
    errors = []
    succeeded = False
    attempts = [(s, b) for b in (['yahoo','auto'] if x_only else ['auto','yahoo']) for s in searches]
    for term, backend in attempts:
        try:
            rows = client.text(term, max_results=min(30,max(10,limit*2)) if x_only else limit, backend=backend)
            succeeded = True
        except DDGSException as exc:
            if 'No results found' in str(exc) and not exc.__cause__:
                succeeded = True
            else:
                errors.append(type(exc).__name__)
            continue
        for row in rows:
            url = row.get('href', '')
            if url.startswith(('https://', 'http://')) and (not x_only or public_post(url)):
                sources[url] = {'url': url, 'title': row.get('title', ''), 'snippet': row.get('body', '')}
        if len(sources) >= limit:
            break
    if not succeeded:
        raise RuntimeError('DDGS search failed: ' + ', '.join(errors))
    return {'sources': list(sources.values())[:limit], 'truncated': len(sources) > limit}


if __name__ == '__main__':
    args = json.load(sys.stdin)
    print(json.dumps(search(args['query'], args['limit'], args['xOnly']), ensure_ascii=False))
