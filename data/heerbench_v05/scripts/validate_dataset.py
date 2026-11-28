"""Validate released construction records, references, graph membership and source integrity. Standard library only."""
import collections,hashlib,json,pathlib,sys
R=pathlib.Path(__file__).resolve().parents[1]
def rd(n):return [json.loads(s) for s in (R/n).read_text(encoding='utf-8').splitlines() if s.strip()]

def extended_validation(root=R):
 """Record/graph/source checks only; no entailment or military-validity claim."""
 root=pathlib.Path(root); errors=[]
 def ck(value,message):
  if not value:errors.append(message)
 def rows(name):return [json.loads(s) for s in (root/name).read_text(encoding='utf-8').splitlines() if s.strip()]
 def unique(values,label):ck(len(values)==len(set(values)),label+' has duplicate IDs/values')
 def index(records,key,label):
  unique([r[key] for r in records],label)
  return {r[key]:r for r in records}
 def edgekey(e):return (tuple(sorted(e['parent_ids'])),e['child_id'],e.get('alternative_group',''))
 q=index(rows('reference/questions.jsonl'),'question_id','Questions')
 m=index(rows('model_input/questions.jsonl'),'question_id','Model questions')
 orig=index(rows('provenance/rubric_items.jsonl'),'item_uid','Original items')
 ledger=index(rows('reference/item_ledger.jsonl'),'item_uid','Ledger')
 accepted=index(json.loads((root/'provenance/accepted_v04_items.json').read_text(encoding='utf-8')),'item_uid','Accepted baseline')
 cases=index(rows('provenance/cases.jsonl'),'case_id','Cases')
 docs=rows('provenance/documents.jsonl')
 docids=[f"{d['case_id']}:D{d['doc_order']:02}" for d in docs];unique(docids,'Documents');docmap=dict(zip(docids,docs))
 chains=index(rows('reference/chains.jsonl'),'chain_id','Chains')
 edges=index(rows('reference/dependencies.jsonl'),'dependency_id','Dependencies')
 criteria=index(rows('reference/criteria.jsonl'),'criterion_id','Criteria')
 sources=index(rows('reference/sources.jsonl'),'source_link_id','Sources')
 links=index(rows('reference/error_checks.jsonl'),'error_link_id','Error links')
 alternatives=index(rows('reference/acceptable_alternatives.jsonl'),'alternative_id','Alternatives')
 local_errors=index(rows('reference/local_error_conditions.jsonl'),'local_error_id','Local errors')
 memberships=rows('reference/chain_memberships.jsonl')
 ck(set(orig)==set(ledger)==set(accepted),'Original, accepted and ledger universes differ')
 ck(set(cases)=={f'C{i}' for i in range(1,11)},'Incomplete original scenario universe')
 ck({x['case_id'] for x in q.values()}==set(cases),'Constructed question set does not cover all scenarios')
 for cid,c in cases.items():
  dd=sorted((d for d in docs if d['case_id']==cid),key=lambda d:d['doc_order'])
  ck(len(dd)==c['n_documents'],f'{cid} original document-count mismatch')
  ck(sum(x['case_id']==cid for x in orig.values())==c['n_rubric_items'],f'{cid} original item-count mismatch')
  packet=f"# {c['operation']} — Scenario source packet\n"+''.join(f"\n\n## {cid}:D{d['doc_order']:02} — {d['title']}\n\n{d['text']}\n" for d in dd)
  path=root/'model_input/packets'/f'{cid}.md';ck(path.is_file(),f'{cid} source packet missing')
  if path.is_file():ck(path.read_text(encoding='utf-8')==packet,f'{cid} model-visible packet differs from complete unchanged source reconstruction')
 # Independent release hashes anchor all originals, including uncited documents.
 manifest=json.loads((root/'provenance/MA_Bench_MANIFEST.json').read_text(encoding='utf-8'))['files']
 for upstream,local in [('data/cases.jsonl','cases.jsonl'),('data/documents.jsonl','documents.jsonl'),('data/rubric_items.jsonl','rubric_items.jsonl'),('README.md','MA_Bench_README.md')]:
  data=(root/'provenance'/local).read_bytes();expected=manifest[upstream]
  ck(len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256'],f'{local} original release-manifest mismatch')
 for qid,x in m.items():
  if qid in q:ck(x['case_id']==q[qid]['case_id'],f'{qid} model/evaluator case mismatch')
 member=collections.defaultdict(set);expected_members=set();chain_edges=collections.defaultdict(set)
 for cid,ch in chains.items():
  ck(ch['case_id'] in cases,f'{cid} unknown case')
  ck(ch['new_reference_status']=='constructed_pending_expert_validation',f'{cid} validation status changed without adjudication')
  ids=ch['a_ids']+ch['b_ids']+ch['f_ids'];nodes=set(ids);unique(ids,cid+' members')
  adj=collections.defaultdict(set);rev=collections.defaultdict(set)
  for field,stage in [('a_ids','A'),('b_ids','B'),('f_ids','F')]:
   for qid in ch[field]:
    ck(qid in q and q[qid]['case_id']==ch['case_id'] and q[qid]['abf_stage']==stage,f'{cid} invalid member {qid}')
    member[qid].add(cid);expected_members.add((cid,ch['case_id'],qid,stage))
  unique([edgekey(e) for e in ch['dependencies']],cid+' dependencies')
  for e in ch['dependencies']:
   chain_edges[edgekey(e)].add(cid);unique(e['parent_ids'],cid+' parents')
   ck(e['logic']=='ALL' and bool(e['parent_ids']),f'{cid} invalid joint dependency')
   ck(e['child_id'] in nodes and set(e['parent_ids'])<=nodes,f'{cid} dependency outside chain membership')
   ck(bool(e['rationale'].strip()) and bool(e['counterfactual_test'].strip()),f'{cid} missing dependency explanation')
   for p in e['parent_ids']:adj[p].add(e['child_id']);rev[e['child_id']].add(p)
  for bid in ch['b_ids']:
   ck(bool(rev[bid]&set(ch['a_ids'])),f'{cid} B {bid} has no A predecessor in this chain')
   ck(bool(adj[bid]&set(ch['f_ids'])),f'{cid} B {bid} does not lead to F in this chain')
  for fid in ch['f_ids']:ck(bool(rev[fid]&set(ch['b_ids'])),f'{cid} F {fid} has no B predecessor in this chain')
  for aid in ch['a_ids']:
   abf=any(b in adj[aid] and bool(adj[b]&set(ch['f_ids'])) for b in ch['b_ids'])
   direct=any(aid in e['parent_ids'] and e['child_id'] in ch['f_ids'] and bool(set(e['parent_ids'])&set(ch['b_ids'])) for e in ch['dependencies'])
   ck(abf or direct,f'{cid} A {aid} has no ABF path or joint F prerequisite role')
 actual_members=[(x['chain_id'],x['case_id'],x['question_id'],x['stage']) for x in memberships]
 unique(actual_members,'Normalized memberships');ck(set(actual_members)==expected_members,'Normalized memberships differ from chains')
 unique([edgekey(e) for e in edges.values()],'Normalized dependency relationships')
 ck({edgekey(e) for e in edges.values()}==set(chain_edges),'Normalized and chain-contained dependency universes differ')
 for eid,e in edges.items():
  unique(e['parent_ids'],eid+' parents');unique(e['chain_ids'],eid+' chain IDs')
  ck(set(e['chain_ids'])==chain_edges[edgekey(e)],f'{eid} dependency/chain association mismatch')
  if e['child_id'] in q:ck(e['case_id']==q[e['child_id']]['case_id'],f'{eid} dependency case mismatch')
 expected_cr={};expected_src={};expected_alt={};expected_err={};embedded_ids=[]
 for qid,x in q.items():
  ck(x['abf_stage'] in {'A','B','F'},f'{qid} invalid stage')
  unique(x['chain_ids'],qid+' memberships');unique(x['source_item_uids'],qid+' direct original IDs');unique(x['lineage_item_uids'],qid+' lineage IDs')
  ck(x['source_answer_status']==('explicit' if x['abf_stage']=='A' else 'derived'),f'{qid} source-answer status mismatch')
  ck(bool(x['question'].strip()) and bool(x['expected_answer'].strip()),f'{qid} blank question or exemplar')
  ck(all(bool(x[k]) for k in ['criteria','source_refs','acceptable_alternatives','error_conditions']),f'{qid} missing scoring/reference component')
  ck(bool(x['source_item_uids']) or x['origin']=='constructed_companion',f'{qid} adapted node has no original association')
  ck(set(x['original_tiers_by_item'])==set(x['source_item_uids'])==set(x['original_adaptation_cases_by_item']),f'{qid} per-item metadata keys differ')
  for uid in x['source_item_uids']:
   item=ledger.get(uid);ck(item is not None,f'{qid} unknown original {uid}')
   if item:
    ck(item['construction_status']=='RETAINED_IN_CHAIN' and item['original_polarity']=='positive' and item['case_id']==x['case_id'] and qid in item['question_ids'],f'{qid} non-reciprocal or invalid direct original association {uid}')
   if uid in orig and uid in accepted:
    ck(x['original_tiers_by_item'].get(uid)==orig[uid]['tier'] and x['original_adaptation_cases_by_item'].get(uid)==accepted[uid]['accepted_case'],f'{qid} inherited metadata altered for {uid}')
  for uid in x['lineage_item_uids']:ck(uid in orig and orig[uid]['case_id']==x['case_id'] and orig[uid]['polarity']=='positive',f'{qid} invalid ancestral/anchor provenance {uid}')
  for c in x['criteria']:
   embedded_ids.append(c['criterion_id']);expected_cr[c['criterion_id']]=(qid,x['case_id'],x['abf_stage'],c['text'],c['required'])
  for n,s in enumerate(x['source_refs'],1):expected_src[f'{qid}-S{n}']=(qid,x['case_id'],s['doc_id'],s['line_start'],s['line_end'],s['support'])
  for n,a in enumerate(x['acceptable_alternatives'],1):expected_alt[f'{qid}-ALT{n}']=(qid,a)
  for n,e in enumerate(x['error_conditions'],1):expected_err[f'{qid}-E{n}']=(qid,e)
 unique(embedded_ids,'Embedded criterion IDs')
 ck({i:(x['question_id'],x['case_id'],x['abf_stage'],x['criterion'],x['required']) for i,x in criteria.items()}==expected_cr,'Normalized criteria differ from embedded question criteria')
 ck({i:(x['question_id'],x['case_id'],x['doc_id'],x['line_start'],x['line_end'],x['support']) for i,x in sources.items()}==expected_src,'Normalized source links differ from embedded question references')
 ck({i:(x['question_id'],x['text']) for i,x in alternatives.items()}==expected_alt,'Normalized alternatives differ from questions')
 ck({i:(x['question_id'],x['text']) for i,x in local_errors.items()}==expected_err,'Normalized local errors differ from questions')
 for sid,s in sources.items():
  d=docmap.get(s['doc_id'])
  if d:
   ck(s['question_id'] in q and d['case_id']==s['case_id']==q[s['question_id']]['case_id'],f'{sid} source/question case mismatch')
   ck(s['doc_order']==d['doc_order'] and s['doc_number']==d['doc_number'] and s['document_title']==d['title'] and s['original_source_file']==d['source_file'],f'{sid} source metadata mismatch')
 pairs=[(e['error_item_uid'],e['question_id']) for e in links.values()];unique(pairs,'Error item/question associations')
 linked=collections.defaultdict(set)
 for eid,e in links.items():
  item=orig.get(e['error_item_uid']);ck(item is not None and e['question_id'] in q,f'{eid} unknown error item or question')
  if item and e['question_id'] in q:
   ck(item['polarity']==e['original_polarity']=='negative' and item['case_id']==e['case_id']==q[e['question_id']]['case_id'],f'{eid} negative polarity or case mismatch')
   ck(e['original_item_text']==item['question'] and e['original_error_presence_condition']==item['pass_condition'],f'{eid} original negative wording altered')
  ck(bool(e['trigger'].strip()) and bool(e['applicability'].strip()),f'{eid} blank trigger or application reason')
  linked[e['error_item_uid']].add(e['question_id'])
 statuses={'BASELINE_EXCLUDED','RETAINED_IN_CHAIN','EXCLUDED_NO_FULL_CHAIN','ERROR_CHECK_LINKED','ERROR_CHECK_NOT_APPLICABLE'}
 for uid,x in ledger.items():
  o,a=orig.get(uid),accepted.get(uid)
  if o is None or a is None:continue
  ck(x['case_id']==o['case_id'] and x['original_item_id']==o['item_id'],f'{uid} original identity altered')
  for k,j in [('accepted_v04_decision','accepted_decision'),('accepted_v04_case','accepted_case'),('accepted_v04_role','accepted_role')]:ck(x[k]==a[j],f'{uid} accepted baseline {j} altered')
  status=x['construction_status'];ck(status in statuses,f'{uid} unknown construction status')
  unique(x['question_ids'],uid+' question mappings');unique(x['chain_ids'],uid+' chain mappings')
  cc=set().union(*(member[qi] for qi in x['question_ids'])) if x['question_ids'] else set()
  ck(set(x['chain_ids'])==cc,f'{uid} item/chain memberships disagree')
  ck(bool(x['construction_rationale'].strip()),f'{uid} missing disposition rationale')
  ck(x['coverage'] in {'full','partial','none'},f'{uid} invalid coverage')
  if x['coverage']=='partial':ck(bool(x['uncovered_content'].strip()),f'{uid} partial coverage lacks omitted-content explanation')
  if a['accepted_decision']!='INCLUDE':ck(status=='BASELINE_EXCLUDED',f'{uid} baseline exclusion overwritten')
  elif o['polarity']=='positive':ck(status in {'RETAINED_IN_CHAIN','EXCLUDED_NO_FULL_CHAIN'},f'{uid} positive item has invalid disposition')
  else:ck(status in {'ERROR_CHECK_LINKED','ERROR_CHECK_NOT_APPLICABLE'},f'{uid} negative item has invalid disposition')
  if status=='RETAINED_IN_CHAIN':
   ck(bool(x['question_ids']) and x['coverage'] in {'full','partial'},f'{uid} retained item not represented')
   for qi in x['question_ids']:
    if qi in q:ck(uid in q[qi]['source_item_uids'],f'{uid} question {qi} lacks reciprocal provenance')
  elif status in {'BASELINE_EXCLUDED','EXCLUDED_NO_FULL_CHAIN','ERROR_CHECK_NOT_APPLICABLE'}:ck(not x['question_ids'] and not x['chain_ids'] and x['coverage']=='none',f'{uid} unmapped status has retained content')
  if o['polarity']=='negative':
   ck(set(x['question_ids'])==linked[uid],f'{uid} negative links disagree with disposition')
   if status=='ERROR_CHECK_LINKED':ck(bool(linked[uid]),f'{uid} linked negative has no applicable question')
 return errors

def main():
 errors=extended_validation(R)
 def ck(x,msg):
  if not x:errors.append(msg)
 q={x['question_id']:x for x in rd('reference/questions.jsonl')};m={x['question_id']:x for x in rd('model_input/questions.jsonl')}
 original={x['item_uid']:x for x in rd('provenance/rubric_items.jsonl')};ledger={x['item_uid']:x for x in rd('reference/item_ledger.jsonl')}
 documents={f"{x['case_id']}:D{x['doc_order']:02}":x for x in rd('provenance/documents.jsonl')}
 chains=rd('reference/chains.jsonl');edges=rd('reference/dependencies.jsonl');criteria=rd('reference/criteria.jsonl');src=rd('reference/sources.jsonl');err=rd('reference/error_checks.jsonl')
 ck(len(original)==1257 and set(original)==set(ledger),'Original item universe is incomplete')
 ck(len(documents)==112,'Source-document universe is incomplete')
 ck(set(q)==set(m),'Model/evaluator question universes differ')
 for i,x in m.items():
  ck(set(x)=={'question_id','case_id','question'},f'{i} model-facing metadata leakage')
  ck(x['question']==q[i]['question'],f'{i} inconsistent prompt wording')
 members=collections.defaultdict(set)
 for ch in chains:
  for f in ['a_ids','b_ids','f_ids']:ck(bool(ch[f]),f'{ch["chain_id"]} missing ABF stage')
  for f,stage in [('a_ids','A'),('b_ids','B'),('f_ids','F')]:
   for i in ch[f]:
    ck(i in q and q[i]['abf_stage']==stage,f'{ch["chain_id"]} invalid member {i}');members[i].add(ch['chain_id'])
 for i,x in q.items():
  ck(bool(members[i]) and set(x['chain_ids'])==members[i],f'{i} orphan/membership mismatch')
  ck(x['new_reference_status']=='constructed_pending_expert_validation',f'{i} validation status changed without release adjudication')
 for e in edges:
  ck(e['logic']=='ALL' and bool(e['parent_ids']),f'{e["dependency_id"]} invalid joint logic')
  ck(e['child_id'] in q,f'{e["dependency_id"]} child missing')
  for parent in e['parent_ids']:
   ck(parent in q,f'{e["dependency_id"]} parent missing')
   if parent in q and e['child_id'] in q:
    child=q[e['child_id']];pa=q[parent]
    ck(pa['case_id']==child['case_id'],f'{e["dependency_id"]} cross-case edge')
    ck('ABF'.index(pa['abf_stage'])<'ABF'.index(child['abf_stage']),f'{e["dependency_id"]} backward or cyclic edge')
  if e['child_id'] in q:
   stage=q[e['child_id']]['abf_stage'];ps={q[p]['abf_stage'] for p in e['parent_ids'] if p in q}
   ck((stage=='B' and 'A' in ps) or (stage=='F' and 'B' in ps),f'{e["dependency_id"]} missing preceding stage')
 for x in src:
  d=documents.get(x['doc_id']);ck(d is not None,f'{x["source_link_id"]} document missing')
  if d:
   lines=d['text'].splitlines();a,b=x['line_start'],x['line_end'];ck(1<=a<=b<=len(lines),f'{x["source_link_id"]} source range invalid')
   ck('\n'.join(lines[a-1:b])==x['source_excerpt'],f'{x["source_link_id"]} excerpt changed')
   ck(hashlib.sha256(d['text'].encode()).hexdigest()==x['document_sha256'],f'{x["source_link_id"]} source checksum mismatch')
 for i,x in ledger.items():
  o=original[i]
  for k in ['question','pass_condition','fail_condition','tier','polarity','severity','weight','step','step_name','source_fact','artifact_anchor','source_documents']:
   ck(x['original_'+k]==o[k],f'{i} original {k} altered')
  if x['construction_status']=='RETAINED_IN_CHAIN':ck(bool(x['question_ids']),f'{i} retained item has no question')
  for qi in x['question_ids']:ck(qi in q,f'{i} missing linked question')
 for e in err:ck(e['question_id'] in q and original[e['error_item_uid']]['polarity']=='negative',f'{e["error_link_id"]} invalid error check')
 ck(len({x['criterion_id'] for x in criteria})==len(criteria),'Duplicate criterion IDs')
 report={'questions':len(q),'chains':len(chains),'criteria':len(criteria),'items':len(ledger),'documents':len(documents),'dependency_sets':len(edges),'source_links':len(src),'error_links':len(err),'source_packets_checked':10,'errors':errors}
 print(json.dumps(report,indent=2));sys.exit(bool(errors))
if __name__=='__main__':
 try:main()
 except (OSError,KeyError,TypeError,ValueError) as exc:
  print(json.dumps({'errors':[f'Malformed or missing release record: {exc}']},indent=2));sys.exit(1)
