"""Request-scoped outbound network policy."""

from __future__ import annotations

from ipaddress import (
    IPv4Address,
    IPv4Network,
    IPv6Address,
    IPv6Network,
    ip_address,
    ip_network,
)
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

from .client_exceptions import ClientAddressNotAllowedError

IPNetwork = Union[IPv4Network, IPv6Network]
IPAddress = Union[IPv4Address, IPv6Address]
AllowNetworksInput = Optional[Union["NetworkPolicy", Iterable[Union[str, IPNetwork]]]]


class NetworkPolicy:
    """An immutable allow-list of IP networks for one logical request.

    ``networks is None`` means that no request-level policy was supplied and
    all of aiohttp's legacy connection behaviour is preserved.  An empty tuple
    is different: it is an explicit deny-all policy.
    """

    __slots__ = ("_networks",)

    def __init__(self, networks: Optional[Sequence[IPNetwork]]) -> None:
        self._networks = None if networks is None else tuple(networks)

    @classmethod
    def parse(cls, allow_networks: Any) -> "NetworkPolicy":
        if isinstance(allow_networks, NetworkPolicy):
            return allow_networks
        if allow_networks is None:
            return UNRESTRICTED_NETWORK_POLICY
        if isinstance(allow_networks, (str, bytes)):
            raise TypeError("allow_networks must be an iterable of CIDR networks")

        parsed = []
        for network in allow_networks:
            if isinstance(network, (IPv4Network, IPv6Network)):
                parsed.append(network)
            elif isinstance(network, str):
                # Accept e.g. 127.0.0.1/8 as 127.0.0.0/8 rather than making
                # the security API unnecessarily strict about host bits.
                parsed.append(ip_network(network, strict=False))
            else:
                raise TypeError(
                    "allow_networks entries must be CIDR strings or "
                    "ipaddress network objects"
                )
        return cls(tuple(parsed))

    @property
    def enabled(self) -> bool:
        return self._networks is not None

    @property
    def networks(self) -> Optional[Tuple[IPNetwork, ...]]:
        return self._networks

    def __repr__(self) -> str:
        if self._networks is None:
            return "NetworkPolicy(unrestricted)"
        return f"NetworkPolicy({[str(network) for network in self._networks]!r})"

    def allows_address(self, address: Union[str, IPAddress]) -> bool:
        if self._networks is None:
            return True

        if isinstance(address, str):
            try:
                ip = ip_address(address)
            except ValueError:
                # URL literals may carry a fully-qualified trailing dot.
                if address.endswith("."):
                    try:
                        ip = ip_address(address.rstrip("."))
                    except ValueError:
                        return False
                else:
                    return False
        else:
            ip = address

        candidates = [ip]
        if isinstance(ip, IPv6Address) and ip.ipv4_mapped is not None:
            # A dual-stack socket may report an IPv4 peer as
            # ::ffff:127.0.0.1.  Treat it as the IPv4 address for IPv4 CIDRs
            # while also retaining the original value for IPv6 CIDRs.
            candidates.append(ip.ipv4_mapped)

        for candidate in candidates:
            for network in self._networks:
                if candidate.version == network.version and candidate in network:
                    return True
        return False

    def allows_peer(self, peername: Any) -> bool:
        if self._networks is None:
            return True
        address = address_from_peername(peername)
        return address is not None and self.allows_address(address)

    def allows_transport(self, transport: Any) -> bool:
        if self._networks is None:
            return True
        if transport is None:
            return False
        try:
            peername = transport.get_extra_info("peername")
        except AttributeError:
            return False
        return self.allows_peer(peername)

    def ensure_allowed_transport(
        self,
        transport: Any,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        if self._networks is None:
            return

        peername: Any = None
        if transport is not None:
            try:
                peername = transport.get_extra_info("peername")
            except AttributeError:
                peername = None

        address = address_from_peername(peername)
        if address is None or not self.allows_address(address):
            raise ClientAddressNotAllowedError(
                address=address if address is not None else peername,
                host=host,
                port=port,
                allow_networks=[str(network) for network in self._networks],
            )

    def filter_resolve_results(self, resolve_results: Any) -> list[Any]:
        if self._networks is None:
            # Preserve the exact return object/type for the legacy path.
            return resolve_results  # type: ignore[no-any-return]
        return [
            result for result in resolve_results if self.allows_address(result["host"])
        ]


UNRESTRICTED_NETWORK_POLICY = NetworkPolicy(None)


def address_from_peername(peername: Any) -> Optional[IPAddress]:
    """Extract an IP address from an asyncio socket peername."""
    if isinstance(peername, tuple) and peername:
        raw_address = peername[0]
    elif isinstance(peername, str):
        # Unix socket paths and other non-IP names are deliberately rejected
        # by an enabled IP policy.
        raw_address = peername
    else:
        return None

    if not isinstance(raw_address, str):
        return None
    try:
        return ip_address(raw_address)
    except ValueError:
        return None
