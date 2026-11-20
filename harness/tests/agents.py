#!/usr/bin/env python3
"""Exercise all five CLI wire formats through the installed Harness without a network connection."""
import base64
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('muse', 'grok', 'cursor', 'codex', 'claude', 'devin', 'google')


def assert_astra_subscription():
    assert sys.argv[sys.argv.index('--model')+1] == 'gpt-6.2-astra'
    assert 'model_reasoning_effort="high"' in sys.argv
    assert 'forced_login_method="chatgpt"' in sys.argv
    assert 'model_provider="openai"' in sys.argv
    assert 'cli_auth_credentials_store="file"' in sys.argv
    assert not any(name.startswith('OPENAI_') for name in os.environ)


def fixture(model):
    if model == 'google' and 'models' in sys.argv:
        print('gemini-test-medium\tGemini Test (Medium)\ngemini-test-low\tGemini Test (Low)\ngemini-test-high\tGemini Test (High)\norchestrator-test\tOrchestrator')
        return
    if model == 'cursor' and '--list-models' in sys.argv:
        print('auto - Auto\ncursor-test - Cursor Test')
        return
    if model == 'cursor':
        assert sys.argv[sys.argv.index('--mode') + 1] == 'ask'
        assert '--sandbox' not in sys.argv  # Respect the user's Cursor CLI sandbox setting.
    if model == 'devin' and 'models' in sys.argv:
        print(json.dumps({'families':[{'variants':[{'model_uid':'swe-2-high','label':'SWE-2 High','cost_tier':'Free'},{'model_uid':'fusion-paid','label':'Paid Fusion','cost_summary':'$8 / 1M Input · Sidekick: Free'}]}]}))
        return
    if '--prompt-file' in sys.argv:
        prompt = Path(sys.argv[sys.argv.index('--prompt-file') + 1]).read_text()
    else:
        prompt = sys.stdin.read()
    if model == 'google':
        assert '--input-format' in sys.argv and '--disable-slash-commands' in sys.argv
        assert 'command(*)' in sys.argv[sys.argv.index('--settings') + 1]
        prompt = json.loads(prompt)['message']['content']
    request = json.loads(prompt.splitlines()[-1])
    messages = request['messages']
    texts = [block['text'] for message in messages if message['role'] == 'user'
             for block in message['content'] if block['type'] == 'text']
    managed = next((text for text in texts if text.startswith('DSH_MANAGED_TASK:')), None)
    if managed:
        task = json.loads(managed.splitlines()[-1])['task']
        assert model != 'devin', 'Devin must never receive managed work'
        if task['stage']==0:
            assert model in ('cursor','grok'), 'Only Cursor/Grok may receive initial managed work'
        if task['prompt'].startswith('NATIVE_CONCURRENCY_PROBE '):
            path=Path(task['prompt'].split(' ',1)[1])
            def count(delta):
                with path.open('r+') as file:
                    fcntl.flock(file,fcntl.LOCK_EX)
                    value=json.load(file)
                    value['active']+=delta
                    value['peak']=max(value['peak'],value['active'])
                    file.seek(0);json.dump(value,file);file.truncate()
            count(1)
            time.sleep(0.6)
            count(-1)
        if model == 'claude':
            assert sys.argv[sys.argv.index('--model')+1] == 'claude-opus-5-5'
        if model == 'codex':
            assert_astra_subscription()
        unresolved = task['id'] == 'escalate' and model != 'claude'
        answer = json.dumps({'dsh_tool':'structured_output','arguments':{
            'status':'unresolved' if unresolved else 'resolved','summary':'fixture '+model,
            'evidence':['fixture evidence'],'attempts':['approach one','approach two'],
            'blockers':['fixture problem'] if unresolved else [],'changedFiles':[]}})
    elif 'TEST_SUBSCRIPTION_CHILD' in texts:
        assert model == 'codex'
        assert_astra_subscription()
        answer = 'SUBSCRIPTION_CHILD_OK'
    elif 'TEST_DIRECT_CODEX_SUBSCRIPTION' in texts:
        assert model == 'codex'
        assert_astra_subscription()
        sent = any(block['type'] == 'tool-call' for message in messages for block in message['content'])
        if sent:
            answer = 'DIRECT_SUBSCRIPTION_OK' if 'SUBSCRIPTION_CHILD_OK' in str(messages) else 'WAITING_FOR_SUBSCRIPTION_CHILD'
        else:
            answer = json.dumps({'dsh_tool':'subagent','arguments':{'description':'Verify subscription-only child',
                'provider':'local-cli','model':'codex','prompt':'TEST_SUBSCRIPTION_CHILD'}})
    elif 'TEST_AUTO' in texts:
        assert prompt.startswith('DSH routing context:')
        if model == 'cursor':
            sys.stderr.write('Cursor quota exhausted (fixture)')
            sys.exit(1)
        if model == 'codex':
            assert_astra_subscription()
            answer = json.dumps({'dsh_tool':'task_queue','arguments':{'action':'escalate',
                'reason':'Two fixture approaches failed','evidence':['Approach A failed','Approach B failed']}})
        else:
            assert model == 'claude'
            assert sys.argv[sys.argv.index('--model')+1] == 'claude-opus-5-5'
            assert any(m['role']=='tool' for m in messages)
            answer = 'AUTO_OPUS_OK'
    elif 'TEST_QUEUE' in texts:
        assert model == 'codex', 'TWA coordinator must stay on subscription Astra'
        assert_astra_subscription()
        calls = [b for m in messages for b in m['content'] if b['type']=='tool-call' and b['name']=='task_queue']
        latest = messages[-1]
        if not calls:
            args = {'action':'import','planPath':'plan.json'}
        elif latest['role'] != 'tool':
            args = {'action':'status','limit':4}
        else:
            raw = ''.join(b.get('text','') for b in latest['content'])
            # Harness truncates large tool results. Fetch a bounded page rather
            # than treating its human-readable truncation marker as JSON.
            try:
                status = json.loads(raw)
            except json.JSONDecodeError:
                status = None
            last_args = json.loads(calls[-1]['arguments']) if isinstance(calls[-1]['arguments'],str) else calls[-1]['arguments']
            if last_args['action'] == 'import':
                args = {'action':'run'}
            elif status is None:
                assert last_args != {'action':'status','limit':4}, raw
                args = {'action':'status','limit':4}
            elif status.get('counts',{}).get('accepted') == status.get('total'):
                args = None
                answer = 'QUEUE_OK'
            elif any(t['status']=='review' for t in status.get('tasks',[])):
                ready = next(t for t in status['tasks'] if t['status']=='review')
                args = {'action':'review','taskId':ready['id'],'accepted':True,'evidence':['Independent fixture verification']}
            else:
                args = None
                answer = 'QUEUE_RUNNING'
        if args is not None:
            answer = json.dumps({'dsh_tool':'task_queue','arguments':args})
    elif 'TEST_MIXED_TOOL' in texts or 'TEST_MIXED_STUCK' in texts:
        results = [m for m in messages if m['role'] == 'tool']
        if results:
            answer = 'MIXED_TOOL_OK' if not results[-1].get('isError') and 'fixture-tool-ok' in str(results[-1]['content']) else 'MIXED_TOOL_FAILED'
        else:
            call = json.dumps({'dsh_tool':'bash','arguments':{'description':'Fixture command','command':'printf fixture-tool-ok'}})
            answer = call if 'DSH_PROTOCOL_REPAIR:' in prompt and 'TEST_MIXED_STUCK' not in texts else 'まず現状を確認します。' + call
    elif 'TEST_OPTIONS' in texts:
        answer = 'OPTIONS ' + base64.b64encode(json.dumps({'service':model,'args':sys.argv[1:]}).encode()).decode()
    elif 'TEST_MATRIX' in texts:
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
    elif model == 'google':
        print(json.dumps({'event':'result','result':{'status':'SUCCESS','response':answer}}))
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
            (binaries / ('agy' if model == 'google' else 'cursor-agent' if model == 'cursor' else model)).symlink_to(Path(__file__).resolve())

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
        print(f'{len(MODELS)**2}/{len(MODELS)**2} parent-child routes returned their child results.')
        assert ask('devin-free','TEST_CHILD',base/'devin-free',base/'devin-free-state') == 'CHILD_OK_devin'
        assert ask('devin-free','TEST_MIXED_TOOL',base/'mixed-tool',base/'mixed-state') == 'MIXED_TOOL_OK'
        print('Devin free-only repairs mixed prose/JSON and returns the real Harness tool result.')
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
        fixture('google' if name == 'agy' else 'cursor' if name == 'cursor-agent' else name)
