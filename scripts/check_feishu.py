#!/usr/bin/env python3
"""Read-only Feishu API connectivity check. Never prints document contents or secrets."""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://open.feishu.cn/open-apis"

def api(method, url, *, body=None, token=None):
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if token:
        headers["Authorization"] = "Bearer " + token
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        # Avoid emitting detailed server responses or request credentials.
        raise RuntimeError(f"HTTP {exc.code} calling Feishu API") from None
    except urllib.error.URLError:
        raise RuntimeError("Network error reaching Feishu API") from None
    if result.get("code") != 0:
        raise RuntimeError(f"Feishu API error code={result.get('code')} msg={result.get('msg','')}")
    return result.get("data", {}), result

def main():
    app_id = os.environ.get("FEISHU_APP_ID")
    app_secret = os.environ.get("FEISHU_APP_SECRET")
    node_token = os.environ.get("FEISHU_WIKI_NODE_TOKEN", "W1jhwJs53ifE6RkhloOc3XtHnmg")
    if not app_id or not app_secret:
        sys.exit("Missing FEISHU_APP_ID or FEISHU_APP_SECRET environment variables.")
    _, response = api("POST", BASE + "/auth/v3/tenant_access_token/internal", body={"app_id": app_id, "app_secret": app_secret})
    token = response.get("tenant_access_token")
    if not token:
        sys.exit("Auth response missing tenant_access_token.")
    print("PASS: Application authentication")
    node_data, _ = api("GET", BASE + "/wiki/v2/spaces/get_node?" + urllib.parse.urlencode({"token": node_token}), token=token)
    node = node_data.get("node", {})
    obj_token = node.get("obj_token")
    obj_type = node.get("obj_type")
    if not obj_token:
        sys.exit("Wiki node found but missing obj_token.")
    print(f"PASS: Wiki node access (type={obj_type})")
    if obj_type != "docx":
        print("SKIP: Document blocks test supports docx nodes only.")
        return
    doc_data, _ = api("GET", BASE + "/docx/v1/documents/" + urllib.parse.quote(obj_token, safe="") + "/blocks?page_size=100", token=token)
    print(f"PASS: Docx blocks read (first page block count={len(doc_data.get('items', []))}; has_more={doc_data.get('has_more', False)})")
    print("No document content was printed or published.")

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        sys.exit("FAIL: " + str(exc))
