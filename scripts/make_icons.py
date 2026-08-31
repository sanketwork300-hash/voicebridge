#!/usr/bin/env python3
"""Generate the extension's PNG icons.

Kept as a script rather than committing opaque binaries people cannot review or
regenerate. Pure stdlib: no Pillow dependency for a build step this small.
"""

import struct
import sys
import zlib
from pathlib import Path


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_icon(path: Path, size: int) -> int:
    """A rounded blue disc with a lighter 'bridge' arc through the middle."""
    cx = cy = size / 2.0
    radius = size * 0.46
    rows = bytearray()
    for y in range(size):
        rows.append(0)  # PNG filter type 0
        for x in range(size):
            dx, dy = x + 0.5 - cx, y + 0.5 - cy
            dist = (dx * dx + dy * dy) ** 0.5
            # Arc: a band curving across the disc.
            arc = abs(dy + (dx * dx) / (size * 0.5) - size * 0.10) < size * 0.075
            if dist <= radius:
                if arc and abs(dx) < radius * 0.72:
                    rows += bytes((235, 243, 255, 255))
                else:
                    t = y / size
                    rows += bytes((int(74 + 46 * t), int(140 + 40 * t), 255, 255))
            elif dist <= radius + 1.0:
                # Cheap antialiased edge.
                alpha = max(0, int(255 * (radius + 1.0 - dist)))
                rows += bytes((74, 140, 255, alpha))
            else:
                rows += bytes((0, 0, 0, 0))

    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + _chunk(b"IEND", b"")
    )
    path.write_bytes(png)
    return len(png)


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "apps/browser-extension/icons")
    out.mkdir(parents=True, exist_ok=True)
    for size in (16, 48, 128):
        n = write_icon(out / f"icon{size}.png", size)
        print(f"icon{size}.png  {n} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
