#!/usr/bin/env python3
"""Seg4 wrapper node that reuses the original YOLOE detection service behavior."""

from __future__ import annotations

from yoloe_detection_service.yoloe_detection_service_node import (
    YoloeDetectionServiceNode,
    main as _original_main,
)


def main(args=None) -> None:
    # Reuse the original node implementation exactly so TF naming, base_link
    # transforms, and service behavior match the existing production node.
    _original_main(args)


if __name__ == "__main__":
    main()
