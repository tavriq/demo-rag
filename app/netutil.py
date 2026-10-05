"""Client IP resolution for rate limiting.

X-Forwarded-For is honoured only when the direct peer is a configured trusted
proxy (TRUSTED_PROXY). Otherwise anyone could rotate the header to dodge the
rate limit.
"""

from __future__ import annotations

from ipaddress import IPv4Network, IPv6Address, IPv6Network, ip_address, ip_network

# One IPv6 subscriber usually gets a whole /64, so a per-address limit is no limit.
IPV6_PREFIX = 64


def _in_networks(ip: str, networks: tuple[IPv4Network | IPv6Network, ...]) -> bool:
    try:
        addr = ip_address(ip)
    except ValueError:
        return False
    return any(addr in net for net in networks)


def client_ip(
    peer: str | None,
    forwarded_for: str | None,
    trusted: tuple[IPv4Network | IPv6Network, ...],
) -> str:
    peer = peer or "unknown"
    if not trusted or not forwarded_for or not _in_networks(peer, trusted):
        return peer
    # Walk from the right: the rightmost entries were appended by our own proxies.
    hops = [h.strip() for h in forwarded_for.split(",") if h.strip()]
    for hop in reversed(hops):
        try:
            ip_address(hop)
        except ValueError:
            return peer  # garbage in the chain: fall back to the proxy address
        if not _in_networks(hop, trusted):
            return hop
    return peer


def rate_limit_key(ip: str) -> str:
    """Bucket for the per-client rate limit: the IPv4 address, or the IPv6 /64 network."""
    try:
        addr = ip_address(ip)
    except ValueError:
        return ip
    if isinstance(addr, IPv6Address):
        if addr.ipv4_mapped is not None:
            return str(addr.ipv4_mapped)
        return str(ip_network(f"{addr}/{IPV6_PREFIX}", strict=False))
    return str(addr)
