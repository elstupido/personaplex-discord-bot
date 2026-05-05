"""
WHY THIS FILE EXISTS:
The Surgical Intervention (Monkey-Patching for Resilience).

WHY:
Standard library voice layers are often built for 'Best Effort' 
communication. In a PersonaPlex environment, where every lost packet 
degrades the identity model, we must intervene in the internal mechanics.

1. THE KEY-ROTATION RACE: Discord rotates E2EE keys mid-stream. There is 
   a temporal window where the key on the wire and the key in the CPU 
   don't match. We implement a 'Grace Window' to retry decryption with 
   the previous key, preventing audio 'Glitching' during rotation.

2. NAT PERSISTENCE: Network mappings behind Docker bridges or corporate 
   firewalls can time out during periods of AI-induced silence. We 
   tighten the keepalive heartbeat to ensure the 'Warm Path' remains 
   open through the NAT device.
"""

import logging

logger = logging.getLogger("voice.patches")

def apply_reader_patches():
    """
    Apply monkey-patches to discord.voice.receive.reader to improve 
    network resilience and handle key rotations.
    """
    from nacl.exceptions import CryptoError

    try:
        from discord.voice.receive.reader import PacketDecryptor, UDPKeepAlive
    except ImportError:
        logger.warning("[Patch] Could not import PacketDecryptor — skipping reader patches")
        return

    # ── Patch 1: key-rotation retry ─────────────────────────────────────────
    # WHY THIS PATCH?
    # Discord rotates UDP keys mid-stream. In the millisecond window where 
    # the bot hasn't processed the new key yet, incoming packets fail AEAD 
    # decryption (CryptoError). By caching the previous key and retrying, 
    # we prevent the 'robotic stutter' caused by packet loss during rotation.
    
    if hasattr(PacketDecryptor, "_antigravity_patched"):
        return
        
    _orig_update_secret_key = PacketDecryptor.update_secret_key
    _orig_decrypt_rtp = PacketDecryptor.decrypt_rtp

    def _patched_update_secret_key(self, secret_key: bytes) -> None:
        """Cache the outgoing key before replacing it."""
        self._prev_box = getattr(self, 'box', None)  # snapshot before overwrite
        _orig_update_secret_key(self, secret_key)
        logger.debug("[Patch] Secret key rotated — previous box cached for retry")

    def _patched_decrypt_rtp(self, packet):
        """
        Try decryption with the current key; on CryptoError retry once with the
        previous key (covers the in-flight packets during a key rotation).
        """
        try:
            return _orig_decrypt_rtp(self, packet)
        except CryptoError:
            prev_box = getattr(self, '_prev_box', None)
            if prev_box is None:
                raise
            # Swap in the previous box, retry, then restore the current box
            current_box = self.box
            self.box = prev_box
            try:
                result = _orig_decrypt_rtp(self, packet)
                logger.debug(
                    "[Patch] Recovered packet with previous key (key-rotation window)"
                )
                return result
            except CryptoError:
                raise  # genuinely corrupt — let the caller discard it
            finally:
                self.box = current_box

    PacketDecryptor.update_secret_key = _patched_update_secret_key
    PacketDecryptor.decrypt_rtp = _patched_decrypt_rtp
    PacketDecryptor._antigravity_patched = True
    logger.info("[Patch] PacketDecryptor: key-rotation retry enabled")

    # ── Patch 2: tighter UDP keepalive ──────────────────────────────────────
    # WHY? NAT devices and Docker bridges are aggressive. If the GPU is 
    # churning and the bot stops sending for 5s, the mapping might die. 
    # 2.5s is the 'sweet spot' for keeping the tunnel warm.
    UDPKeepAlive.delay = 2500  # ms — was 5000; keeps NAT mapping alive under GPU load
    logger.info("[Patch] UDPKeepAlive.delay set to 2500 ms")
