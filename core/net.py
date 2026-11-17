# AreaGIS Kit
# Copyright (C) 2026 Pradeepika Kaushik
# SPDX-License-Identifier: GPL-3.0-or-later
# Additional terms under GPL section 7 (attribution and name): see NOTICE.
"""Small helpers for web requests and for optional steps that may fail.

Every download goes through open_url, which accepts only http and https addresses, so a crafted address can never
make the plugin read a local file. Optional steps that fail (a missing scene, a service that is down) are logged with
skipped() and the study carries on; the report then says what is missing.
"""
import logging
import urllib.parse
import urllib.request

LOG = logging.getLogger("areagiskit")
_OPENER = urllib.request.build_opener()


def open_url(req, timeout=60):
    """Open an http(s) request or address and return the response."""
    url = req.full_url if isinstance(req, urllib.request.Request) else str(req)
    if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
        raise ValueError(f"Only http and https addresses can be opened: {url[:80]}")
    return _OPENER.open(req, timeout=timeout)


def skipped(exc):
    """Note an optional step that failed, with the function it happened in."""
    LOG.info("optional step skipped: %r", exc, stacklevel=2)
