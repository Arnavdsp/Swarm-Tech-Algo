"""
Peer-to-peer mesh link between the six Raspberry Pis.

Every drone broadcasts a small JSON packet at ``heartbeat_hz`` and keeps a table
of what it last heard from each peer.  There is no master: a drone that hears
nobody still flies its own slot from the shared virtual-structure description,
it just stops contributing to consensus.

Two implementations share one interface:
  * ``UdpMeshLink``  — real UDP broadcast over the Pis' Wi-Fi/ESP mesh.
  * ``LoopbackLink`` — in-process, used by the simulator and the tests.
"""
import json
import socket
import threading
import time


class BaseLink:
    """Interface: ``send(msg)``, ``poll()``, ``peers()``, ``close()``."""

    def __init__(self, drone_id, cfg):
        self.drone_id = drone_id
        self.cfg = cfg
        self._peers = {}          # drone_id -> (timestamp, payload)
        self._inbox = []

    def send(self, msg):
        raise NotImplementedError

    def poll(self):
        """Drain and return everything received since the last call."""
        msgs, self._inbox = self._inbox, []
        for m in msgs:
            src = m.get("id")
            if src is not None and src != self.drone_id:
                self._peers[src] = (time.time(), m)
        return msgs

    def peers(self, now=None):
        """Peers heard within ``peer_timeout_s``."""
        now = time.time() if now is None else now
        return {k: v[1] for k, v in self._peers.items()
                if now - v[0] <= self.cfg.peer_timeout_s}

    def lost_peers(self, expected_ids, now=None):
        alive = set(self.peers(now)) | {self.drone_id}
        return sorted(set(expected_ids) - alive)

    def close(self):
        pass


class LoopbackLink(BaseLink):
    """Shared in-process bus. All links built from the same ``bus`` see each other."""

    def __init__(self, drone_id, cfg, bus=None):
        super().__init__(drone_id, cfg)
        self.bus = bus if bus is not None else []
        self.bus.append(self)

    def send(self, msg):
        msg = dict(msg, id=self.drone_id, t=time.time())
        for link in self.bus:
            if link is not self:
                link._inbox.append(msg)
        return msg


class UdpMeshLink(BaseLink):
    """UDP broadcast link. One background thread does the receiving."""

    def __init__(self, drone_id, cfg):
        super().__init__(drone_id, cfg)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        # In flight each drone is its own machine, but bench-testing several
        # nodes on one host means several sockets on one port. Without
        # SO_REUSEPORT only one of them receives, and the mesh looks broken for
        # reasons that have nothing to do with the code under test.
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        self.sock.bind((cfg.bind_host, cfg.port))
        self.sock.settimeout(0.2)
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._thread.start()

    def _rx_loop(self):
        while self._running:
            try:
                data, _ = self.sock.recvfrom(8192)
            except (socket.timeout, OSError):
                continue
            try:
                msg = json.loads(data.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue          # a malformed packet is never worth a crash
            with self._lock:
                self._inbox.append(msg)

    def send(self, msg):
        msg = dict(msg, id=self.drone_id, t=time.time())
        payload = json.dumps(msg, separators=(",", ":")).encode("utf-8")
        try:
            self.sock.sendto(payload, (self.cfg.broadcast_addr, self.cfg.port))
        except OSError:
            pass                  # a dropped heartbeat is recoverable; a crash isn't
        return msg

    def poll(self):
        with self._lock:
            msgs, self._inbox = self._inbox, []
        for m in msgs:
            src = m.get("id")
            if src is not None and src != self.drone_id:
                self._peers[src] = (time.time(), m)
        return msgs

    def close(self):
        self._running = False
        try:
            self._thread.join(timeout=1.0)
        finally:
            self.sock.close()


def make_link(drone_id, cfg, kind="udp", bus=None):
    if kind == "udp":
        return UdpMeshLink(drone_id, cfg)
    if kind == "loopback":
        return LoopbackLink(drone_id, cfg, bus=bus)
    raise ValueError(f"unknown link kind: {kind}")
