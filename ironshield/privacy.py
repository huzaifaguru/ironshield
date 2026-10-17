"""Privacy layer: replace raw account IDs with keyed hashes before any analysis.

HMAC-SHA256 with a secret salt means the same account always maps to the same
pseudonym (so behaviour can still be tracked), but the original ID cannot be
recovered without the secret.
"""
import hashlib
import hmac
from functools import lru_cache

from . import config

_KEY = config.PRIVACY_SALT.encode()


@lru_cache(maxsize=500_000)
def pseudonymise(account_id: str) -> str:
    digest = hmac.new(_KEY, str(account_id).encode(), hashlib.sha256).hexdigest()
    return "acct_" + digest[:12]


def protect(txn: dict) -> dict:
    """Return a copy of a raw transaction with identities replaced by pseudonyms."""
    safe = dict(txn)
    safe["src"] = pseudonymise(txn["src"])
    safe["dst"] = pseudonymise(txn["dst"])
    return safe
