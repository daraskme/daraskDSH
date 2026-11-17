#!/usr/bin/env python3
"""Exercise all five CLI wire formats through the installed Harness without a network connection."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('muse', 'grok', 'cursor', 'codex', 'claude', 'devin')


def fixture(model):
    if model == 'devin' and 'models' in sys.argv:
        print(json.dumps({'families':[{'variants':[{'model_uid':'swe-2-high','label':'SWE-2 High','cost_tier':'Free'},{'model_uid':'fusion-paid','label':'Paid Fusion','cost_summary':'$8 / 1M Input · Sidekick: Free'}]}]}))
        return
    if '--prompt-file' in sys.argv:
        prompt = Path(sys.argv[sys.argv.index('--prompt-file') + 1]).read_text()
    else:
        prompt = sys.stdin.read()
    request = json.loads(prompt.splitlines()[-1])
    messages = request['messages']
    texts = [block['text'] for message in messages if message['role'] == 'user'
             for block in message['content'] if block['type'] == 'text']
    if 'TEST_MATRIX' in texts:
        sent = any(block['type'] == 'tool-call' for message in messages for block in message['content'])
        if sent:
            answer = 'Parent received ' + ','.join(m for m in MODELS if 'CHILD_OK_' + m in str(texts))
        else:
            answer = json.dumps([{'dsh_tool': 'subagent', 'arguments':
                {'description': 'Check child response', 'prompt': 'TEST_CHILD', 'provider': 'local-cli', 'model': m}}
                for m in MODELS])
    elif 'TEST_CHILD' in texts:
        answer = 'CHILD_OK_' + model
    elif 'TEST_SEARCH' in texts:
        results = [m for m in messages if m['role'] == 'tool']
        if not results:
            answer = json.dumps({'dsh_tool': 'session_search', 'arguments': {'query': 'recall-token-6ce2'}})
        else:
            answer = 'Found' if not results[-1].get('isError') and 'recall-token-6ce2' in str(results[-1]['content']) else 'Absent'
    elif 'TEST_SEED' in texts:
        answer = 'recall-token-6ce2'
    elif 'TEST_RECALL' in texts:
        answer = 'Remembered' if 'recall-token-6ce2' in str(texts) else 'No prior token'
    else:
        answer = 'Verification'
    if model == 'muse':
        print(json.dumps({'payload_type': 'run.terminal.completed', 'payload':
              {'terminal': 'completed', 'text': answer}}))
    elif model == 'codex':
        Path(sys.argv[sys.argv.index('--output-last-message') + 1]).write_text(answer)
    elif model in ('grok','devin'):
        print(answer)
    else:
        print(json.dumps({'type': 'result', 'subtype': 'success', 'result': answer}))


def main():
    with tempfile.TemporaryDirectory(prefix='dsh-verification-') as tmp:
        base = Path(tmp)
        binaries = base / 'bin'
        binaries.mkdir()
        for model in MODELS:
            (binaries / ('cursor-agent' if model == 'cursor' else model)).symlink_to(Path(__file__).resolve())

        def ask(model, task, workspace, state):
            workspace.mkdir(exist_ok=True)
            env = {**os.environ, 'PATH': f'{binaries}:{os.environ["PATH"]}', 'DSH_HOME': str(state)}
            result = subprocess.run([ROOT / 'dsh', 'ask', model, task], cwd=workspace,
                                    env=env, capture_output=True, text=True, timeout=90)
            assert result.returncode == 0, (model, result.stderr[-1200:])
            return result.stdout.strip()

        def matrix(model):
            result = ask(model, 'TEST_MATRIX', base / model, base / (model + '-state'))
            assert result == 'Parent received ' + ','.join(MODELS), (model, result)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(matrix, MODELS))
        print('36/36 parent-child routes returned their child results.')
        assert ask('devin-free','TEST_CHILD',base/'devin-free',base/'devin-free-state') == 'CHILD_OK_devin'
        denied = subprocess.run([ROOT/'dsh','ask','devin-free','--model','fusion-paid','TEST_CHILD'],
                                cwd=base/'devin-free', env={**os.environ,'DSH_HOME':str(base/'devin-free-state'),
                                'PATH':f'{binaries}:{os.environ["PATH"]}'},capture_output=True,text=True,timeout=40)
        assert denied.returncode != 0 and '無料' in denied.stderr
        print('Devin free-only accepts a Free model and rejects a paid model with a free sidekick.')
        workspace, state = base / 'shared', base / 'persisted'
        assert ask('grok', 'TEST_SEED', workspace, state) == 'recall-token-6ce2'
        assert ask('cursor', 'TEST_RECALL', workspace, state) == 'Remembered'
        assert ask('muse', 'TEST_RECALL', base / 'other', state) == 'No prior token'
        assert ask('muse', 'TEST_SEARCH', workspace, state) == 'Found'
        assert ask('claude', 'TEST_SEARCH', base / 'other', state) == 'Absent'
        print('Cross-process recall, older-session search and workspace separation passed.')


if __name__ == '__main__':
    name = Path(sys.argv[0]).name
    if name == 'agents.py':
        main()
    else:
        fixture('cursor' if name == 'cursor-agent' else name)
