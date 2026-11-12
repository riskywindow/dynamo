#!/usr/bin/env python3
"""Prefix each stdin line with its arrival time (epoch seconds, ms)."""
import sys, time
for line in sys.stdin:
    sys.stdout.write("%.3f %s" % (time.time(), line))
    sys.stdout.flush()
