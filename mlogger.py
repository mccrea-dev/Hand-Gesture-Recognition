#!/usr/bin/env python3
#This program should be able to detect the OS that you are using
#You must have pyserial: pip3 install pyserial
#plug chip in then run

import argparse, csv, sys, time, os
from datetime import datetime
from typing import Optional, Tuple, List
import serial
from serial.tools import list_ports

# ---------- Port discovery ----------
LIKELY_KEYS = ("usbmodem", "usbserial", "nrf", "xiao", "seeed", "com", "ttyacm", "ttyusb")

def discover_ports() -> List[serial.tools.list_ports_common.ListPortInfo]:
    return list(list_ports.comports())

def score_port(p) -> int:
    s = f"{p.device} {p.description} {p.hwid}".lower()
    return sum(k in s for k in LIKELY_KEYS)

def pick_port(prefer: Optional[str]) -> Optional[str]:
    if prefer:
        return prefer
    ports = discover_ports()
    if not ports:
        return None
    ports.sort(key=score_port, reverse=True)
    return ports[0].device

# ---------- Serial open with safe toggles ----------
def open_serial(port: str, baud: int, timeout: float = 1.0) -> serial.Serial:
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = timeout
    # macOS sometimes buffers until DTR/RTS toggled:
    ser.dtr = False
    ser.rts = False
    ser.open()
    time.sleep(0.25)
    ser.reset_input_buffer()
    ser.dtr = True
    ser.rts = True
    return ser

# ---------- CSV helpers ----------
EXPECTED_MIN_COLS = 7  # timestamp_ms, ax, ay, az, gx, gy, gz
EXPECTED_MAX_COLS = 8  # + tempC (optional)

def parse_line(line: str) -> Optional[List[str]]:
    # Accepts lines like: "12345,0.01,..." (7 or 8 columns)
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < EXPECTED_MIN_COLS or len(parts) > EXPECTED_MAX_COLS:
        return None
    if not parts[0].isdigit():
        return None
    # Validate numeric body quickly
    try:
        for x in parts[1:]:
            float(x)
    except ValueError:
        return None
    return parts

def header_for(cols: int) -> List[str]:
    base = ["timestamp_ms","ax","ay","az","gx","gy","gz"]
    return base if cols == EXPECTED_MIN_COLS else base + ["tempC"]

# ---------- Reconnect logic ----------
def try_connect(port: Optional[str], baud: int, auto: bool, verbose: bool) -> Tuple[Optional[serial.Serial], Optional[str]]:
    if auto:
        chosen = pick_port(port)
    else:
        chosen = port
    if not chosen:
        if verbose:
            print("[logger] No serial ports detected. Plug the device in.", file=sys.stderr)
        return None, None
    try:
        ser = open_serial(chosen, baud)
        if verbose:
            print(f"[logger] Connected on {chosen} @ {baud}")
        return ser, chosen
    except Exception as e:
        if verbose:
            print(f"[logger] Failed to open {chosen}: {e}", file=sys.stderr)
        return None, None

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Platform-independent IMU serial logger")
    ap.add_argument("--port", "-p", help="Serial port (COM5, /dev/cu.usbmodem14101, etc.)")
    ap.add_argument("--baud", "-b", type=int, default=115200, help="Baud rate (default: 115200)")
    ap.add_argument("--out", "-o", help="Output CSV filename (default: imu_session_YYYYmmdd_HHMMSS.csv)")
    ap.add_argument("--append", action="store_true", help="Append to output file if it exists")
    ap.add_argument("--duration", type=float, help="Stop after N seconds (default: run until Ctrl-C)")
    ap.add_argument("--no-auto", action="store_true", help="Do NOT auto-pick a port; require --port")
    ap.add_argument("--show-ports", action="store_true", help="List all ports and exit")
    ap.add_argument("--quiet", "-q", action="store_true", help="Less console output")
    args = ap.parse_args()

    if args.show_ports:
        print("[logger] Available serial ports:")
        for p in discover_ports():
            print(f"  {p.device:>20}  |  {p.description}")
        return

    if args.no_auto and not args.port:
        print("[logger] --no-auto requires explicit --port", file=sys.stderr)
        sys.exit(2)

    out_name = args.out or f"imu_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    mode = "a" if args.append and os.path.exists(out_name) else "w"

    if not args.quiet:
        print("[logger] Ports (most likely first):")
        ports = discover_ports()
        ports.sort(key=score_port, reverse=True)
        for p in ports:
            print(f"  {p.device:>20}  |  {p.description}")

    ser, current_port = try_connect(args.port, args.baud, auto=not args.no_auto, verbose=not args.quiet)

    # Prepare file
    f = open(out_name, mode, newline="")
    writer = csv.writer(f)
    wrote_header_cols = 0
    if mode == "w":
        # We don’t know yet if tempC exists—write header after first valid row
        pass

    if not args.quiet:
        print(f"[logger] Writing to {out_name}  (Ctrl+C to stop)")

    start_t = time.time()
    lines_ok = 0
    lines_bad = 0
    last_warn = 0.0

    try:
        while True:
            # Connect/reconnect loop
            if ser is None:
                time.sleep(0.5)
                ser, current_port = try_connect(args.port, args.baud, auto=not args.no_auto, verbose=not args.quiet)
                continue

            try:
                raw = ser.readline()
            except Exception as e:
                if not args.quiet:
                    print(f"\n[logger] Read error on {current_port}: {e}. Reconnecting...", file=sys.stderr)
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
                continue

            if not raw:
                # No data for a moment → print gentle hint once in a while
                now = time.time()
                if now - last_warn > 3 and lines_ok == 0 and not args.quiet:
                    print("[logger] No data yet… Check: correct sketch? 115200 baud? Serial Monitor closed?")
                    last_warn = now
                # also check duration if set
                if args.duration and (now - start_t) >= args.duration:
                    break
                continue

            line = raw.decode(errors="ignore").strip()
            parsed = parse_line(line)
            if parsed:
                # Write header lazily based on columns we actually saw
                if wrote_header_cols == 0:
                    wrote_header_cols = len(parsed)
                    writer.writerow(header_for(wrote_header_cols))
                writer.writerow(parsed)
                lines_ok += 1
                if not args.quiet and lines_ok % 200 == 0:
                    print(f"[logger] {lines_ok} rows captured (bad: {lines_bad})", end="\r")
            else:
                lines_bad += 1
                # Show a few unparsed examples early on
                if not args.quiet and (lines_ok < 5 and lines_bad <= 10):
                    print(f"[debug] Unparsed line: {line}")

            if args.duration and (time.time() - start_t) >= args.duration:
                break

    except KeyboardInterrupt:
        if not args.quiet:
            print("\n[logger] Stopping (Ctrl-C).")
    finally:
        try:
            if ser:
                ser.close()
        except Exception:
            pass
        f.flush(); f.close()
        if not args.quiet:
            print(f"[logger] Done. Rows: {lines_ok}, Unparsed: {lines_bad}")
            print("[logger] File saved:", os.path.abspath(out_name))

if __name__ == "__main__":
    main()