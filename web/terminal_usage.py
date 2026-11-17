"""Read provider-reported CLI quotas from their terminal interfaces."""
import datetime as dt
import fcntl
import json
import os
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time

import pyte


def read_panel(agent, program, directory):
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 48, 128, 0, 0))
    screen = pyte.Screen(128, 48)
    stream = pyte.ByteStream(screen)
    args = [program, '--no-alt-screen'] if agent == 'grok' else [
        program, '--process-model', 'in-process', '--respect-workspace-trust', 'false']
    proc = subprocess.Popen(args, cwd=directory, stdin=slave, stdout=slave, stderr=slave,
                            start_new_session=True, env={**os.environ, 'TERM': 'xterm-256color', 'LC_ALL': 'C.UTF-8'})
    os.close(slave)

    def pump(seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if select.select([master], [], [], .15)[0]:
                data = os.read(master, 65536)
                stream.feed(data)
                if b'\x1b[6n' in data:
                    os.write(master, b'\x1b[1;1R')
    try:
        pump(2)
        if agent == 'grok':
            os.write(master, b'/usage')
            pump(.4)
            os.write(master, b'\r')
        for _ in range(16):
            pump(.5)
            lines = '\n'.join(row.strip(' │') for row in screen.display if row.strip(' │'))
            if agent == 'devin':
                status = re.search(r'([\d.]+)% remaining(?: \(resets in ([^)]*)\))?', lines, re.I)
                if status:
                    duration = re.findall(r'(\d+)\s*([dhms])', status[2] or '')
                    seconds = sum(int(n)*{'d':86400,'h':3600,'m':60,'s':1}[unit] for n,unit in duration)
                    reset = dt.datetime.now().astimezone() + dt.timedelta(seconds=seconds)
                    return [{'label':'Devin CLI', 'remaining':float(status[1]), 'reset':reset.isoformat() if duration else None, 'approximate':bool(duration)}]
                continue
            panels = re.findall(r'([^\n]+ limit)\b[^\n]*\n[^\n]*?([\d.]+)%\s*\nResets:\s*([A-Za-z]+ \d+, \d\d?:\d\d)', lines, re.I)
            if panels:
                return panels
        raise RuntimeError('CLI usage unavailable')
    finally:
        os.close(master)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


with tempfile.TemporaryDirectory() as directory:
    agent, program = sys.argv[1:3]
    data = read_panel(agent, program, directory)
    if agent == 'devin':
        print(json.dumps(data))
    else:
        windows = []
        for label, used, date in data:
            now = dt.datetime.now()
            reset = dt.datetime.strptime(f'{now.year} {date}', '%Y %B %d, %H:%M')
            if reset < now - dt.timedelta(days=1):
                reset = reset.replace(year=now.year + 1)
            windows.append({'label': label.strip(), 'remaining': 100 - float(used), 'reset': reset.astimezone().isoformat()})
        print(json.dumps(windows))
