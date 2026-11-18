"""Real Harness managed children, fixed fallback model IDs and durable acceptance."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='dsh-router-integration-') as temp:
    base=Path(temp)
    (base/'bin').mkdir()
    (base/'codex-home').mkdir()
    (base/'codex-home/auth.json').write_text(json.dumps({'auth_mode':'chatgpt','tokens':{'access_token':'fixture','refresh_token':'fixture','id_token':'fixture','account_id':'fixture'}}))
    env={**os.environ,'CODEX_HOME':str(base/'codex-home'),'OPENAI_API_KEY':'fixture-must-not-be-used','PATH':str(base/'bin')+':'+os.environ['PATH']}
    for name in ('devin','claude','codex','cursor-agent','grok'):
        (base/'bin'/name).symlink_to(root/'tests/agents.py')
    result=subprocess.run([root/'dsh','ask','cursor','TEST_DIRECT_CODEX_SUBSCRIPTION'],cwd=base,
        env={**env,'DSH_HOME':str(base/'direct-state'),'DSH_TWA_ROOT':str(base)},
        capture_output=True,text=True,timeout=45)
    assert result.returncode==0 and result.stdout.strip().splitlines()[-1:] == ['DIRECT_SUBSCRIPTION_OK'],(result.stdout,result.stderr[-4000:])
    assert not result.stderr.strip(),result.stderr[-3000:]
    print('Direct plain-Codex children in TWA are pinned to subscription Astra/high with API credentials excluded.')
    if os.environ.get('DSH_ROUTER_INTEGRATION_ONLY')=='direct-subscription':
        sys.exit(0)
    tasks=[{'id':name,'phase':'analysis','title':name,'prompt':'Complete the fixture task.',
            'acceptance':['fixture proof'],'dependsOn':['escalate'] if name=='dependent' else [],
            'reads':[],'writes':[name]} for name in ('escalate','independent','normal-third','dependent')]
    (base/'plan.json').write_text(json.dumps({'version':1,'objective':'Fixture plan','inventoryComplete':True,
        'sources':['fixture checkpoint'],'phases':[{'id':'analysis','title':'Analysis'}],'tasks':tasks}))
    result=subprocess.run([root/'dsh','ask','devin-free','TEST_QUEUE'],cwd=base,
        env={**env,'DSH_HOME':str(base/'state'),'DSH_TWA_ROOT':str(base),'DSH_TASK_MAX_PARALLEL':'3','PATH':str(base/'bin')+':'+os.environ['PATH']},
        capture_output=True,text=True,timeout=80)
    assert result.returncode==0,(result.stdout,result.stderr[-6000:])
    files=list((base/'state/task-runs').glob('*/state.json'))
    assert len(files)==1,(result.stdout,result.stderr[-3000:])
    state=json.loads(files[0].read_text())
    assert all(t['status']=='accepted' for t in state['tasks']),(state,result.stdout,result.stderr[-3000:])
    attempts=next(t['attempts'] for t in state['tasks'] if t['id']=='escalate')
    assert [a['route'] for a in attempts]==['Cursor','Codex Astra 6.2','Claude Opus 5.5']
    assert {t['baseService'] for t in state['tasks']}=={'cursor','grok'}
    assert all(a.get('childId') for a in attempts)
    assert all(t['attempts'][0]['route'] in ('Cursor','Grok') for t in state['tasks'])
    assert not result.stderr.strip(),result.stderr[-3000:]
    routing=list((base/'state/routing').glob('*.json'))
    assert not routing, 'TWA parent must stay on subscription Astra'
    print('Native Cursor/Grok-only pool with fixed subscription Astra coordinator escalates unresolved work -> Astra 6.2 -> Opus 5.5; all tasks independently accepted and persisted.')
    result=subprocess.run([root/'dsh','ask','auto','TEST_AUTO'],cwd=base,
        env={**env,'DSH_HOME':str(base/'auto-state'),'PATH':str(base/'bin')+':'+os.environ['PATH']},
        capture_output=True,text=True,timeout=50)
    assert result.returncode==0 and result.stdout.strip()=='AUTO_OPUS_OK',(result.stdout,result.stderr[-4000:])
    routing=list((base/'auto-state/routing').glob('*.json'))
    assert len(routing)==1
    state=json.loads(routing[0].read_text())
    assert state['stage']==2 and len(state['history'])==2
    assert not result.stderr.strip(),result.stderr[-3000:]
    print('General auto service starts at Cursor; quota failure and explicit escalation preserve the conversation through Opus 5.5.')
    counter=base/'concurrency.json'
    counter.write_text(json.dumps({'active':0,'peak':0}))
    tasks=[{'id':'wide-'+str(i),'phase':'analysis','title':'Parallel '+str(i),
            'prompt':'NATIVE_CONCURRENCY_PROBE '+str(counter),'acceptance':['fixture proof'],
            'dependsOn':[],'reads':[],'writes':['scope-'+str(i)]} for i in range(32)]
    (base/'plan.json').write_text(json.dumps({'version':1,'objective':'Wide fixture plan','inventoryComplete':True,
        'sources':['fixture checkpoint'],'phases':[{'id':'analysis','title':'Analysis'}],'tasks':tasks}))
    result=subprocess.run([root/'dsh','ask','devin-free','TEST_QUEUE'],cwd=base,
        env={**env,'DSH_HOME':str(base/'wide-state'),'DSH_TWA_ROOT':str(base),'DSH_TASK_MAX_PARALLEL':'16','PATH':str(base/'bin')+':'+os.environ['PATH']},
        capture_output=True,text=True,timeout=90)
    assert result.returncode==0,(result.stdout,result.stderr[-4000:])
    state=json.loads(next((base/'wide-state/task-runs').glob('*/state.json')).read_text())
    assert all(t['status']=='accepted' and len(t['attempts'])==1 for t in state['tasks']),state
    peak=json.loads(counter.read_text())['peak']
    assert 8<peak<=16,peak
    assert {t['baseService'] for t in state['tasks']}=={'cursor','grok'}
    assert not result.stderr.strip(),result.stderr[-3000:]
    print(f'Native parallel admission exceeded the old 8-child cap: peak {peak}; 32 tasks completed without spurious fallback.')
