"""Explicit ntfy delivery with retries and observable failures; no secrets logged."""
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
from urllib.request import Request,urlopen


def validate_destination(topic,server,token_file=None):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',topic):raise ValueError('Invalid ntfy topic.')
    url=urlsplit(server)
    if url.scheme!='https' or not url.netloc or url.username or url.password or url.query or url.fragment:
        raise ValueError('Use an HTTPS ntfy server without embedded credentials.')
    headers={'Content-Type':'application/json'}
    if token_file:
        token=Path(token_file).read_text().strip()
        if not token or '\n' in token or '\r' in token:raise ValueError('The ntfy token file must contain one nonempty token.')
        headers['Authorization']='Bearer '+token
    return headers


def publish(topic,title,message,*,server='https://ntfy.sh',token_file=None,attempts=3):
    headers=validate_destination(topic,server,token_file)
    payload=json.dumps(dict(topic=topic,title=title,message=message,priority=3),ensure_ascii=False).encode('utf-8')
    last=None
    for attempt in range(attempts):
        try:
            request=Request(server.rstrip('/')+'/',data=payload,headers=headers,method='POST')
            with urlopen(request,timeout=12) as response:
                result=json.load(response)
                if response.status!=200 or result.get('event')!='message':raise ValueError('Unexpected ntfy response.')
                return dict(deliveredToServer=True,messageId=result.get('id'),attempts=attempt+1)
        except Exception as error:
            # Store only an exception class/code; response/URL can contain secrets.
            last=type(error).__name__
            if attempt+1<attempts:time.sleep(min(2**attempt,4))
    return dict(deliveredToServer=False,error=last,attempts=attempts)
