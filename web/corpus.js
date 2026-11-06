/* Corpus UI contract, aligned with corpus_api.py / corpus.py / instructions.py.
 * GET lists: {items,total,page,pages}; instructions/jobs and nodes: arrays.
 * Corpus id IS frozen version id. Detail: {id,document_id,kind,status,title,
 * units,manifest:{unit_count,image_count,pages,index},source_path,
 * actual_source_path,source_valid,source_error,source_url,markdown}.
 * Listing currently lacks counts: cells say detail-only instead of invented totals.
 * Kinds: paper/web/image/data; kind:null lets the server infer from the document.
 * GET /corpus/{id}/annotations supplies string label/caption values and normalized
 * bbox {coords,coordinate_system}; POST uses {label}/{caption}/{box,coordinates}.
 * Bulk import accepts records, explicitly defaults verified=false, requires notes.
 * Corpus retry re-enqueues POST /documents/{document_id}/corpus with original
 * metadata + generate_after (there is no corpus/jobs/{id}/retry route).
 * Instruction detail: {row,data,hash,status,subset,source_kind,review:{passed,check},
 * evidence:[{version_id,document_id,units}],sources:[{version_id,unit_ids}],
 * source_valid,source_error,review_history}. All decisions bind expected_hash=hash.
 * POST /instructions/{id}/edit: {expected_hash,input,thinking,output,language,
 * task_type}; server automatically enqueues re-review. POST /{id}/review and
 * /{id}/commit or reject: {expected_hash}. Jobs retry: /jobs/{job_id}/retry.
 * ZIP instruction export requires one explicit subset; Corpus export uses id.
 * Existing api() handles session headers; standalone fallback GETs bootstrap.
 * Opening/polling panels only reads metadata. No automatic generation/demo data.
 */
(() => {
  'use strict';
  const $c = s => document.querySelector(s);
  const html = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const idpath = v => encodeURIComponent(String(v));
  const items = data => Array.isArray(data) ? data : data?.items || [];
  const text = v => typeof v === 'object' && v !== null ? JSON.stringify(v, null, 2) : String(v ?? '');
  const statuses = {queued:'等待处理',running:'处理中',pending:'待审核',reviewing:'审核中',approved:'审核通过',committed:'已入库',rejected:'已拒绝',failed:'处理失败',completed:'已完成',ready:'已就绪',parsing:'读取证据',extracting:'整理中',generating:'生成中',cancelled:'已取消',insufficient:'证据不足'};
  Object.assign(statuses,{corpus_preparing:'准备资料',corpus_archiving:'保存成果',corpus_completed:'整理完成',preparing:'校验证据',repairing:'修订中'});
  function jobMessage(j){
    if(j.error)return j.error;
    if(j.progress){try{const p=JSON.parse(j.progress);return p.percent===100?'成果已保存':`已准备 ${p.units||0} 个证据单元${p.pages?' · '+p.pages+' 页':''}`}catch{return j.progress}}
    return j.note||'';
  }
  const kinds = {paper:'论文 / 文档',data:'数据集',web:'网页',document:'文档',pdf:'PDF',image:'图片',dataset:'数据集',text:'文本'};
  let sessionPromise, dialogSeq = 0, dialogBusy = false, taskMode = false;
  let corpusTab = 'results', instructionTab = 'bilingual', taskCount = null;
  const lists = {corpus:{page:1,pages:1,seq:0},instructions:{page:1,pages:1,seq:0}};
  let jobBusy = {}, nodeSeq = 0;
  const corpusJobs=new Map();
  function notice(message) {
    if (typeof toast === 'function') toast(message);
    else {const el=$c('#toast');el.textContent=message;el.hidden=false;setTimeout(()=>el.hidden=true,6500)}
  }
  function human(error) {
    if (error.status === 409) return '资料或审核版本已更新，请刷新详情后重试；当前输入已保留。';
    if (error.status === 404 || error.status === 405 || error.status === 501) return '此功能接口暂不可用，请更新服务后重试。';
    if (error.status === 401 || error.status === 403) return '会话已失效，请刷新页面后重试。';
    if (error instanceof TypeError) return '服务连接中断，请检查连接后重试。';
    const message=error.message || '';
    return /[\u3400-\u9fff]/.test(message) ? message : '操作未成功，请稍后刷新重试。';
  }
  async function ensureSession() {
    if (typeof bootstrap !== 'undefined' && bootstrap.session) return bootstrap.session;
    if (!sessionPromise) sessionPromise=fetch('/api/bootstrap',{credentials:'same-origin'}).then(async r=>{
      if(!r.ok) throw Object.assign(Error('服务尚未连接'),{status:r.status});
      const data=await r.json();if(!data.session)throw Error('未取得有效会话，请刷新页面。');
      if(typeof bootstrap!=='undefined') bootstrap.session=data.session;
      return data.session;
    }).catch(e=>{sessionPromise=null;throw e});
    return sessionPromise;
  }
  async function request(path,options={}) {
    const session=await ensureSession();
    if(typeof api==='function') return api(path,options);
    const r=await fetch(path,{...options,credentials:'same-origin',headers:{'X-QA-Session':session,...options.headers}});
    if(!r.ok){const b=await r.json().catch(()=>({}));throw Object.assign(Error(typeof b.detail==='string'?b.detail:'请求失败'),{status:r.status})}
    return r.json();
  }
  const post=(path,body)=>request(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  function modeQuery(){return typeof mode==='undefined'?'live':mode}
  async function downloadZip(path,name) {
    try {
      const session=await ensureSession(),r=await fetch(path+(path.includes('?')?'&':'?')+'mode='+encodeURIComponent(modeQuery()),{credentials:'same-origin',headers:{'X-QA-Session':session}});
      if(!r.ok){const b=await r.json().catch(()=>({}));throw Object.assign(Error(b.detail||'下载未成功'),{status:r.status})}
      const url=URL.createObjectURL(await r.blob()),a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),3000);
    } catch(e){notice(human(e))}
  }
  const empty=(message,cols)=>`<tr><td colspan="${cols}"><div class="empty-state">${html(message)}</div></td></tr>`;
  const button=(label,action,value,disabled=false)=>`<button type="button" class="button ghost" data-c-action="${action}" data-c-id="${html(value)}" ${disabled?'disabled':''}>${html(label)}</button>`;
  const badge=s=>`<span class="status ${s==='failed'||s==='rejected'?'failed':['queued','running','reviewing','generating','parsing'].includes(s)?'running':''}">${html(statuses[s]||s||'状态未提供')}</span>`;
  const field=(label,value)=>`<div class="corpus-field"><dt>${html(label)}</dt><dd>${html(text(value)||'未提供')}</dd></div>`;
  function safeURL(value) {try{const u=new URL(value);return ['https:','http:'].includes(u.protocol)?u.href:''}catch{return ''}}
  const sourceURL=v=>safeURL(v)?`<a class="corpus-link" href="${html(safeURL(v))}" target="_blank" rel="noopener noreferrer">${html(v)}</a>`:html(v||'未提供');
  function openDialog(title,body) {
    dialogSeq++;$c('#corpus-dialog-title').textContent=title;$c('#corpus-dialog-body').innerHTML=body;$c('#corpus-dialog-status').textContent='';
    const d=$c('#corpus-dialog');if(!d.open)d.showModal();d.scrollTop=0;return dialogSeq;
  }
  function closeDialog(){if(dialogBusy)return;dialogSeq++;$c('#corpus-dialog').close()}
  function input(label,name,options={}) {
    const {type='text',required=false,value='',placeholder=''}=options;
    return `<label>${html(label)}<input name="${html(name)}" type="${html(type)}" value="${html(value)}" placeholder="${html(placeholder)}" ${required?'required':''}></label>`;
  }
  const submit=label=>`<div class="corpus-form-actions"><button type="button" class="button secondary" data-c-action="close">取消</button><button type="submit" class="button primary">${label}</button></div>`;
  async function submitForm(form,work) {
    if(dialogBusy)return;dialogBusy=true;form.setAttribute('aria-busy','true');
    const buttons=[...form.querySelectorAll('button')];buttons.forEach(b=>b.disabled=true);const status=$c('#corpus-dialog-status');status.textContent='正在提交，请稍候…';status.dataset.kind='info';
    try{await work(new FormData(form))}catch(e){status.textContent=human(e);status.dataset.kind='error';notice(human(e))}
    finally{dialogBusy=false;buttons.forEach(b=>b.disabled=false);form.removeAttribute('aria-busy')}
  }
  function tabs(group,value,attr) {
    const buttons=[...document.querySelectorAll(`[${attr}]`)];
    buttons.forEach(b=>{const active=b.getAttribute(attr)===value;b.classList.toggle('active',active);b.setAttribute('aria-selected',String(active));b.tabIndex=active?0:-1;$c('#'+b.getAttribute('aria-controls')).hidden=!active});
  }
  function selectCorpusTab(value) {corpusTab=value;tabs('corpus',value,'data-corpus-tab');if(value==='results')loadList('corpus');if(value==='nodes')loadNodes();if(value==='entries'&&typeof loadKnowledge==='function')loadKnowledge()}
  function selectInstructionTab(value) {instructionTab=value;tabs('instructions',value,'data-instruction-tab');if(value==='instructions'){loadList('instructions');loadJobs('instructions')}else if(typeof loadQAs==='function')loadQAs()}
  function showTasks() {
    taskMode=true;if(typeof processingTab!=='undefined')processingTab='corpus';
    ['#qa-records-panel','#knowledge-records-panel'].forEach(s=>$c(s).hidden=true);
    $c('#corpus-jobs-panel').hidden=false;$c('#export-all').hidden=true;
    document.querySelectorAll('[data-processing-tab]').forEach(b=>{b.classList.remove('active');b.setAttribute('aria-selected','false');b.tabIndex=-1});
    $c('#corpus-jobs-tab').classList.add('active');$c('#corpus-jobs-tab').setAttribute('aria-selected','true');$c('#corpus-jobs-tab').tabIndex=0;
    $c('#processing-description').textContent='查看Corpus登记、处理进度与失败原因；整理完成后在资料成果中查看。';
    if(taskCount!==null)$c('#document-count').textContent=taskCount;loadJobs('corpus');
  }
  async function loadList(type) {
    const state=lists[type],seq=++state.seq,isCorpus=type==='corpus',cols=isCorpus?6:5;
    const params=new URLSearchParams({q:$c('#'+type+'-query').value.trim(),page:state.page,page_size:20});
    if(isCorpus)params.set('kind',$c('#corpus-kind').value);else{params.set('subset',$c('#instructions-subset').value);params.set('status',$c('#instructions-status').value)}
    const info=$c('#'+type+'-page-info');info.textContent='正在读取…';
    try {
      const data=await request('/api/'+type+'?'+params);if(seq!==state.seq)return;
      if(!Array.isArray(data.items))throw Error('服务响应不完整，请刷新重试。');
      state.page=Number(data.page)||1;state.pages=Math.max(1,Number(data.pages)||Math.ceil(data.total/20)||1);
      $c('#'+type+'-total').textContent=data.total??data.items.length;
      $c('#'+type+'-rows').innerHTML=data.items.map(r=>isCorpus?`<tr><td><strong>${html(r.title||'未命名资料')}</strong><small class="corpus-muted">${html(r.source_label||'')}</small></td><td>${html(kinds[r.kind]||r.kind||'—')}</td><td><span class="corpus-muted">${html(typeof r.version==='object'?r.version?.number:r.version||r.version_number||'冻结版')}</span>${badge(r.status)}</td><td>${html(r.unit_count??r.units_count??'详情查看')} / ${html(r.image_count??'—')}</td><td><div class="corpus-clamp corpus-path">${html(r.source_path||'未提供路径')}</div></td><td><div class="corpus-row-actions">${button('查看','detail',r.id)}${button('生成指令','generate',r.id)}${button('下载','download',r.id)}</div></td></tr>`:`<tr><td><div class="corpus-clamp">${html(r.instruction||r.input||r.row?.instruction||r.row?.input||r.data?.input||r.title||'查看指令内容')}</div></td><td>${html(r.subset)}</td><td>${html(r.source_label||r.row?.source||(r.source_kind==='user_metadata'?'项目来源标识':'系统来源关系'))}</td><td>${badge(r.status)}</td><td>${button('查看与审核','instruction',r.id)}</td></tr>`).join('')||empty(isCorpus?'暂无资料成果，可从文件处理接入资料。':'暂无匹配指令，可在资料成果中发起加工。',cols);
      info.textContent=`共 ${data.total??data.items.length} 项 · 第 ${state.page} / ${state.pages} 页`;
      $c('#'+type+'-prev').disabled=state.page<=1;$c('#'+type+'-next').disabled=state.page>=state.pages;
    }catch(e){if(seq!==state.seq)return;info.textContent=human(e);$c('#'+type+'-rows').innerHTML=empty('读取失败，请点击刷新重试。',cols);notice(human(e))}
  }
  async function loadJobs(type) {
    if(jobBusy[type])return;jobBusy[type]=true;
    const container=$c('#'+type+'-jobs-list');
    try {
      const data=await request('/api/'+type+'/jobs');const jobs=items(data);
      if(type==='corpus'){jobs.forEach(j=>corpusJobs.set(String(j.document_id),j));taskCount=data.total??jobs.length;if(taskMode)$c('#document-count').textContent=taskCount}
      container.innerHTML=jobs.map(j=>`<article class="corpus-job"><div><strong>${html(j.title||j.name||j.document_name||(j.subset?j.subset+' 指令任务':'资料任务'))}</strong><p>${html(jobMessage(j))}${j.stage?` · ${html(statuses[j.stage]||j.stage)}`:''}</p>${j.error_category?`<small>${html(j.error_category)}</small>`:''}</div>${badge(j.status)}${j.corpus_id||j.version_id?button('查看成果','detail',j.corpus_id||j.version_id):''}${j.status==='failed'?button('重试',type==='corpus'?'retry-corpus':'retry-instructions',type==='corpus'?j.document_id:j.job_id||j.id):''}</article>`).join('')||'<div class="empty-state">暂无处理任务</div>';
    }catch(e){container.innerHTML=`<div class="empty-state">${html(human(e))}</div>`}finally{jobBusy[type]=false}
  }
  function normalizeCorpus(data) {
    const base=data.corpus||data,version=typeof data.version==='object'?data.version:{};
    return {...base,...version,id:base.id,version_id:data.version_id||version.id||base.version_id||base.id,version:version.number||version.version||base.version||'冻结版本',original_url:base.original_url||base.source_url,unit_count:base.unit_count??data.manifest?.unit_count,image_count:base.image_count??data.manifest?.image_count,units:data.units||version.units||base.units||[],annotations:data.annotations||base.annotations||[],markdown:data.markdown??version.markdown??base.markdown};
  }
  function evidenceView(units) {
    return (Array.isArray(units)?units:[]).map(u=>`<article class="corpus-evidence"><div class="corpus-evidence-heading"><strong>${html(u.locator||u.unit_id||u.id||'证据单元')}</strong><span class="tag">${html(kinds[u.kind]||u.kind||'文字')}</span></div>${u.asset_id?`<a href="/api/assets/${idpath(u.asset_id)}?mode=${encodeURIComponent(modeQuery())}" target="_blank" rel="noopener"><img loading="lazy" src="/api/assets/${idpath(u.asset_id)}?mode=${encodeURIComponent(modeQuery())}" alt="${html(u.caption||u.locator||'原始图片证据')}"></a>`:''}<pre class="corpus-pre">${html(u.text||u.markdown||u.quote||u.content||'')}</pre>${(u.annotations||[]).map(annotationView).join('')}</article>`).join('')||'<p class="corpus-muted">服务器未提供可展示的证据单元。</p>';
  }
  function annotationView(a){return `<article class="corpus-annotation"><strong>${html(a.unit_id||'')} · ${html(a.kind||'标注')}</strong><pre class="corpus-pre">${html(text(a.value))}</pre><span class="tag">${a.verified===true?'用户已确认':'未人工确认'}</span><p>${html(a.source_note||'未提供来源说明')}</p></article>`}
  async function openCorpus(id) {
    const seq=openDialog('资料详情','<p class="corpus-muted">正在读取真实资料与证据…</p>');
    try {
      const [raw,annotations]=await Promise.all([request('/api/corpus/'+idpath(id)),request('/api/corpus/'+idpath(id)+'/annotations')]);const c=normalizeCorpus(raw);c.annotations=items(annotations);if(seq!==dialogSeq)return;
      $c('#corpus-dialog-title').textContent=c.title||'资料详情';
      $c('#corpus-dialog-body').innerHTML=`<div class="corpus-detail-actions">${button('生成指令','generate',id)}${button('下载资料包','download',id)}${['paper','web'].includes(c.kind)?button('导出CSV','csv',id):''}${button('添加标注','annotate',id)}${button('导入标注JSON','annotation-import',id)}</div><dl class="corpus-metadata">${field('类型',kinds[c.kind]||c.kind)}${field('版本',c.version||'未提供')}${field('状态',statuses[c.status]||c.status)}${field('证据单元',c.unit_count??c.units.length)}${field('图片数',c.image_count??'未提供')}${field('项目来源标识',c.source_label)}${field('实际服务器路径',c.actual_source_path||c.source_path)}${c.actual_source_path&&c.actual_source_path!==c.source_path?field('登记来源路径',c.source_path):''}<div class="corpus-field"><dt>原始网址</dt><dd>${sourceURL(c.original_url)}</dd></div></dl>${c.source_error?`<p class="corpus-hint">${html(c.source_error)}</p>`:''}<h3>Markdown原文</h3><pre class="corpus-pre corpus-markdown">${html(c.markdown||'暂无Markdown正文，请查看下方冻结证据。')}</pre><h3>真实证据与原图</h3>${evidenceView(c.units)}<h3>资料标注</h3>${c.annotations.map(annotationView).join('')||'<p class="corpus-muted">暂无资料级标注。</p>'}`;
    }catch(e){if(seq===dialogSeq)$c('#corpus-dialog-status').textContent=human(e);notice(human(e))}
  }
  function openWeb() {
    openDialog('接入网页资料',`<p class="corpus-hint">提交网页地址后由后台整理。来源标识用于本项目导出，请填写你实际使用的来源名称。</p><form id="corpus-web-form" class="corpus-form">${input('网页URL','url',{type:'url',required:true,placeholder:'https://…'})}${input('标题（可选）','title')}${input('项目来源标识','source_label',{required:true})}<p class="corpus-muted">同时提取知识条目；是否成功以后台任务与审核结果为准。</p>${submit('接入并排队')}</form>`);
    $c('#corpus-web-form').onsubmit=e=>{e.preventDefault();submitForm(e.currentTarget,async f=>{
      const url=f.get('url').trim();if(!safeURL(url))throw Error('请输入有效的HTTP或HTTPS网页地址。');
      await post('/api/corpus/web',{url,title:f.get('title').trim(),source_label:f.get('source_label').trim(),extract_knowledge:true});
      if(typeof refresh==='function')await refresh();
      $c('#corpus-dialog').close();notice('网页已登记，正在等待后台处理。');navigateTasks();
    })};
  }
  function navigateTasks(){if(typeof setView==='function')setView('files');showTasks()}
  async function actualVersions() {
    let page=1,pages=1,all=[];
    do {const data=await request('/api/corpus?page='+page+'&page_size=100');const batch=items(data);all.push(...batch);pages=Number(data.pages)||Math.ceil((data.total||0)/100)||1;page++;}while(page<=pages);
    return all.filter(c=>c.version_id||c.version?.id||c.id).map(c=>({...c,version_id:c.version_id||c.version?.id||c.id}));
  }
  const versionOptions=(versions,selected='')=>versions.map(c=>`<option value="${html(c.version_id)}" ${String(c.version_id)===String(selected)?'selected':''}>${html(c.title||c.id)} · ${html(typeof c.version==='object'?c.version.number:c.version||'当前版本')}</option>`).join('');
  async function openGenerate(id) {
    const seq=openDialog('从Corpus生成指令','<p class="corpus-muted">正在读取可用版本与知识层级…</p>');
    try {
      const [detail,versions,nodesData]=await Promise.all([request('/api/corpus/'+idpath(id)),actualVersions(),request('/api/corpus/nodes')]);if(seq!==dialogSeq)return;
      const c=normalizeCorpus(detail);if(c.source_valid===false)throw Error(c.source_error||'当前原件失效，不能生成指令。');if(!c.version_id)throw Error('当前成果尚未提供可生成的版本，请等待整理完成或刷新。');
      const nodes=items(nodesData),supplements=versions.filter(v=>String(v.version_id)!==String(c.version_id));
      $c('#corpus-dialog-body').innerHTML=`<p class="corpus-hint">以「${html(c.title)}」的当前冻结版本为来源，目标5条，实际数量取决于证据与审核。</p><form id="corpus-generate-form" class="corpus-form"><label>指令子集<select name="subset" required><option>Science</option><option>MultiModel-Science</option><option>Sonar-Open</option><option>Bio</option></select></label>${input('用于导出的本项目来源标识','source_label',{required:true,value:c.source_label||''})}<p class="corpus-muted">子集是本项目的整理分组；来源标识应填写真实来源，不代表官方数据集。</p><label>知识节点（可多选）<select name="node_ids" multiple size="4">${nodes.map(n=>`<option value="${html(n.id)}">${html(n.name)}</option>`).join('')}</select></label><label>补充成果版本（可多选）<select name="supplement_version_ids" multiple size="4">${versionOptions(supplements)}</select></label><p class="corpus-muted">可按住Ctrl或⌘多选，留空则仅使用当前成果。</p>${submit('生成并审核')}</form>`;
      $c('#corpus-generate-form').onsubmit=e=>{e.preventDefault();submitForm(e.currentTarget,async f=>{if(f.getAll('node_ids').length>20||f.getAll('supplement_version_ids').length>5)throw Error('最多选择20个知识节点与5个补充成果版本。');await post('/api/instructions',{version_id:c.version_id,subset:f.get('subset'),target:5,source_label:f.get('source_label').trim(),node_ids:f.getAll('node_ids'),supplement_version_ids:f.getAll('supplement_version_ids')});$c('#corpus-dialog').close();notice('指令加工任务已登记，生成与审核完成后显示结果。');if(typeof setView==='function')setView('qas');selectInstructionTab('instructions')})};
    }catch(e){if(seq===dialogSeq)$c('#corpus-dialog-status').textContent=human(e);notice(human(e))}
  }
  async function loadNodes() {
    const seq=++nodeSeq;
    try{const nodes=items(await request('/api/corpus/nodes'));if(seq!==nodeSeq)return;
      const index=new Map(nodes.map(n=>[String(n.id),n]));
      $c('#corpus-node-tree').innerHTML=nodes.map(n=>{let depth=0,p=n.parent_id,seen=new Set([String(n.id)]);while(p&&index.has(String(p))&&!seen.has(String(p))){seen.add(String(p));depth++;p=index.get(String(p)).parent_id}return `<article class="corpus-node" style="--node-depth:${Math.min(depth,4)}"><div><strong>${html(n.name)}</strong><p>${html(n.definition||'未提供定义')}</p><small>${n.parent_id?'上级：'+html(index.get(String(n.parent_id))?.name||'未知节点'):'根节点'} · ${n.source_version_id?'关联实际成果版本':'未提供来源版本'}</small></div></article>`}).join('')||'<div class="empty-state">暂无知识层级，请选择实际成果添加节点。</div>';
    }catch(e){if(seq===nodeSeq)$c('#corpus-node-tree').innerHTML=`<div class="empty-state">${html(human(e))}</div>`}
  }
  async function openNode() {
    const seq=openDialog('添加知识节点','<p class="corpus-muted">正在读取实际成果…</p>');
    try{const [versions,nodeData]=await Promise.all([actualVersions(),request('/api/corpus/nodes')]);if(seq!==dialogSeq)return;
      if(!versions.length)throw Error('暂无可关联的实际成果，请先整理Corpus资料。');
      $c('#corpus-dialog-body').innerHTML=`<form id="corpus-node-form" class="corpus-form">${input('节点名称','name',{required:true})}<label>上级节点<select name="parent_id"><option value="">无上级（根节点）</option>${items(nodeData).map(n=>`<option value="${html(n.id)}">${html(n.name)}</option>`).join('')}</select></label><label>定义<textarea name="definition" rows="4" required></textarea></label><label>来源成果版本<select name="source_version_id" required><option value="">请选择实际成果</option>${versionOptions(versions)}</select></label>${submit('保存知识节点')}</form>`;
      $c('#corpus-node-form').onsubmit=e=>{e.preventDefault();submitForm(e.currentTarget,async f=>{await post('/api/corpus/nodes',{name:f.get('name').trim(),parent_id:f.get('parent_id')||null,definition:f.get('definition').trim(),source_version_id:f.get('source_version_id')});$c('#corpus-dialog').close();notice('知识节点已保存。');loadNodes()})};
    }catch(e){if(seq===dialogSeq)$c('#corpus-dialog-status').textContent=human(e)}
  }
  async function openAnnotation(id,bulk=false) {
    const seq=openDialog(bulk?'导入图片标注JSON':'添加证据标注','<p class="corpus-muted">正在读取可标注单元…</p>');
    try{const c=normalizeCorpus(await request('/api/corpus/'+idpath(id)));if(seq!==dialogSeq)return;
      if(bulk){
        $c('#corpus-dialog-body').innerHTML=`<p class="corpus-hint">导入JSON记录数组，每条包含unit_id、kind、value与真实来源说明source_note，可含verified。未明确提供verified=true的记录保持未人工确认。</p><form id="corpus-annotation-form" class="corpus-form"><label>选择JSON文件<input type="file" id="corpus-annotation-file" accept=".json,application/json"></label><label>标注记录<textarea name="records" rows="12" required placeholder="粘贴JSON数组"></textarea></label><details><summary>记录字段说明</summary><p class="corpus-muted">kind为label、bbox或caption。value包含label、box:[x1,y1,x2,y2]及coordinates（normalized或pixel）、或caption。unit_id须对应此成果真实单元。verified缺省为false。</p><p class="corpus-muted">当前单元：${html(c.units.map(u=>u.unit_id||u.id).join('、'))}</p></details>${submit('导入标注')}</form>`;
        $c('#corpus-annotation-file').onchange=async e=>{const file=e.target.files[0];if(!file)return;try{if(file.size>5*1024*1024)throw Error('标注文件超过5MB，请分批导入。');const data=await file.text();if(seq===dialogSeq)$c('#corpus-annotation-form').elements.records.value=data}catch(err){notice(human(err))}};
        $c('#corpus-annotation-form').onsubmit=e=>{e.preventDefault();submitForm(e.currentTarget,async f=>{
          let records;try{const data=JSON.parse(f.get('records'));records=Array.isArray(data)?data:data.records}catch{throw Error('JSON格式无效，请检查后重试。')}
          if(!Array.isArray(records)||!records.length||records.length>100)throw Error('请提供1–100条标注记录，每批最多100条。');
          if(records.some(r=>!r||typeof r.source_note!=='string'||!r.source_note.trim()))throw Error('每条标注必须填写真实来源说明source_note。');
          const known=new Set(c.units.filter(u=>u.asset_id).map(u=>String(u.unit_id||u.id)));
          records=records.map(r=>{if(!r||typeof r!=='object'||!known.has(String(r.unit_id))||!['label','bbox','caption'].includes(r.kind)||r.value==null)throw Error('标注须包含当前成果的真实unit_id、kind与value。');return {unit_id:r.unit_id,kind:r.kind,value:typeof r.value==='string'?{[r.kind]:r.value}:r.kind==='bbox'?{box:r.value.box||r.value.coords,coordinates:r.value.coordinates||r.value.coordinate_system,...(r.value.label?{label:r.value.label}:{})}:r.value,verified:r.verified===true,source_note:typeof r.source_note==='string'?r.source_note:''}});
          const result=await post('/api/corpus/'+idpath(id)+'/annotations/import',{records});
          if(result.failed){throw Error(`已保存 ${result.imported} 条，${result.failed} 条未导入：`+(result.items||[]).filter(r=>r.error).map(r=>`第${r.index+1}条 ${r.error}`).join('；'))}
          notice('标注导入已保存，人工确认状态按记录保留。');openCorpus(id);
        })};return;
      }
      c.units=c.units.filter(u=>u.asset_id);if(!c.units.length)throw Error('暂无可标注的真实图片单元，请先接入含图资料。');
      $c('#corpus-dialog-body').innerHTML=`<form id="corpus-annotation-form" class="corpus-form"><div id="annotation-image-preview" class="corpus-evidence"></div><label>真实证据单元<select name="unit_id" required>${c.units.map(u=>`<option value="${html(u.unit_id||u.id)}">${html(u.locator||u.unit_id||u.id)}${u.asset_id?' · 图片':''}</option>`).join('')}</select></label><label>标注类型<select name="kind"><option value="label">标签 label</option><option value="bbox">边界框 bbox</option><option value="caption">图片说明 caption</option></select></label><div id="annotation-label">${input('标签内容','label')}</div><div id="annotation-box" hidden><label>坐标单位<select name="coordinates"><option value="normalized">归一化（0–1）</option><option value="pixel">像素</option></select></label><label>边界框坐标 [x1,y1,x2,y2]<input name="box" placeholder="例如：0.1,0.1,0.8,0.8"></label></div><label id="annotation-caption" hidden>图片说明<textarea name="caption" rows="3"></textarea></label><label>来源说明<textarea name="source_note" rows="2" required></textarea></label><label class="corpus-check"><input type="checkbox" name="verified">我已查看真实证据并人工确认本条标注</label>${submit('保存标注')}</form>`;
      const form=$c('#corpus-annotation-form');const preview=()=>{const unit=c.units.find(u=>String(u.unit_id||u.id)===form.elements.unit_id.value);$c('#annotation-image-preview').innerHTML=unit?evidenceView([unit]):''};form.elements.unit_id.onchange=preview;preview();form.elements.kind.onchange=()=>{$c('#annotation-label').hidden=form.elements.kind.value!=='label';$c('#annotation-box').hidden=form.elements.kind.value!=='bbox';$c('#annotation-caption').hidden=form.elements.kind.value!=='caption'};
      form.onsubmit=e=>{e.preventDefault();submitForm(form,async f=>{
        const kind=f.get('kind'),value={},caption=f.get('caption').trim();
        if(kind==='label'){value.label=f.get('label').trim();if(!value.label)throw Error('请填写标签内容。')}
        if(kind==='bbox'){const box=f.get('box').replace(/[\[\]]/g,'').split(/[,，\s]+/).filter(Boolean).map(Number),normalized=f.get('coordinates')==='normalized';if(box.length!==4||box.some(n=>!Number.isFinite(n)||n<0||(normalized&&n>1))||box[2]<=box[0]||box[3]<=box[1])throw Error('请填写有效的四个边界框坐标，右下角须大于左上角。');value.box=box;value.coordinates=f.get('coordinates')}
        if(kind==='caption'&&!caption)throw Error('请填写图片说明。');if(kind==='caption')value.caption=caption;
        await post('/api/corpus/'+idpath(id)+'/annotations',{unit_id:f.get('unit_id'),kind,value,verified:f.get('verified')==='on',source_note:f.get('source_note').trim()});notice('标注已保存。');openCorpus(id);
      })};
    }catch(e){if(seq===dialogSeq)$c('#corpus-dialog-status').textContent=human(e)}
  }
  async function openInstruction(id) {
    const seq=openDialog('指令详情与审核','<p class="corpus-muted">正在读取指令及真实证据…</p>');
    try {
      const d=await request('/api/instructions/'+idpath(id));if(seq!==dialogSeq)return;
      const row=d.row||d.fields||{},data=d.data||row,hash=d.hash||d.expected_hash||d.row_hash;
      const canCommit=(d.can_commit??d.status==='approved')&&d.source_valid!==false&&d.review?.passed===true;
      const canEdit=d.status!=='committed'&&d.source_valid!==false;
      const sources=Array.isArray(d.sources)?d.sources:[];
      const versions=Array.isArray(d.evidence)?d.evidence:[];
      const review=d.review||{},note=review.check?.note||review.note||(review.passed===true?'审核通过':'尚无审核通过结果');
      const checkNames={supported:'事实依据',scope_correct:'适用范围',independent:'问题自含',task_correct:'任务类型',annotation_correct:'图片标注',language_correct:'语言表达',no_duplicates:'重复检查',format_correct:'输出格式'};
      $c('#corpus-dialog-body').innerHTML=`<div class="corpus-detail-actions">${badge(d.status)}<span class="tag">${html(d.subset)}</span></div>${d.source_error?`<p class="corpus-hint">${html(d.source_error)}</p>`:''}<h3>导出字段</h3><dl class="corpus-metadata">${Object.entries(row).map(([k,v])=>field(k,v)).join('')}</dl><h3>来源Corpus</h3>${sources.map(v=>`<div class="corpus-source-row">${button('查看来源成果','detail',v.version_id)}<span class="corpus-muted">引用 ${html((v.unit_ids||[]).length)} 个真实证据单元</span></div>`).join('')||'<p class="corpus-muted">暂无通过当前校验的来源关系，请查看历史冻结证据。</p>'}<h3>审核说明</h3><p class="corpus-hint">${html(note)}</p><div class="corpus-detail-actions">${Object.entries(review.check||{}).filter(([k,v])=>checkNames[k]&&typeof v==='boolean').map(([k,v])=>`<span class="tag">${html(checkNames[k])}：${v?'通过':'未通过'}</span>`).join('')}</div><h3>真实证据</h3>${versions.map(v=>`<section class="corpus-evidence-version">${v.version_id?button('查看此Corpus版本','detail',v.version_id):''}${evidenceView(v.units||[])}</section>`).join('')||'<p class="corpus-muted">未提供证据，不能提交。</p>'}${(d.annotations||[]).map(annotationView).join('')}<div class="corpus-detail-actions"><button class="button primary" id="instruction-commit" ${!hash||!canCommit?'disabled':''}>审核通过，提交入库</button><button class="button secondary" id="instruction-reject" ${!hash||d.status==='committed'||d.status==='rejected'?'disabled':''}>拒绝此候选</button><button class="button secondary" id="instruction-review" ${!hash||!['pending','failed'].includes(d.status)||d.source_valid===false?'disabled':''}>重新审核</button></div>${canEdit?`<details class="corpus-edit-details"><summary>编辑未入库候选</summary><p class="corpus-muted">修改后保存为待审核内容，由后台重新审核，审核通过后才可入库。</p><form id="instruction-edit-form" class="corpus-form"><label>问题<input name="input" value="${html(data.input||'')}" required maxlength="8000"></label><label>依据摘要<textarea name="thinking" rows="3" maxlength="2000">${html(data.thinking||'')}</textarea></label><label>答案<textarea name="output" rows="6" required maxlength="8000">${html(data.output||'')}</textarea></label><label>语言<select name="language"><option value="zh" ${data.language==='zh'?'selected':''}>中文</option><option value="en" ${data.language==='en'?'selected':''}>English</option></select></label><label>任务类型<select name="task_type">${[['science','科学问答'],['description','图片描述'],['classification','类别识别'],['localization','位置定位']].map(([v,n])=>`<option value="${v}" ${data.task_type===v?'selected':''}>${n}</option>`).join('')}</select></label><div class="corpus-form-actions"><button class="button primary" type="submit">保存并重新审核</button></div></form></details>`:'<p class="corpus-muted">已入库内容或来源失效的候选不能直接修改。</p>'}<details class="corpus-edit-details"><summary>审核历史（${html((d.review_history||[]).length)}）</summary>${(d.review_history||[]).map(h=>`<p class="corpus-muted">${html(h.review?.check?.note||h.review?.note||(h.review?.passed?'审核通过':'审核未通过'))}</p>`).join('')||'<p class="corpus-muted">暂无历史记录</p>'}</details>`;
      const editForm=$c('#instruction-edit-form');
      const protectDirty=()=>{if(editForm?.dataset.dirty==='true')throw Error('你有尚未保存的编辑，请先保存或重新打开详情后再操作。')};
      if(editForm){editForm.addEventListener('input',()=>editForm.dataset.dirty='true');editForm.onsubmit=e=>{e.preventDefault();submitForm(editForm,async f=>{
        await post('/api/instructions/'+idpath(id)+'/edit',{expected_hash:hash,input:f.get('input').trim(),thinking:f.get('thinking').trim(),output:f.get('output').trim(),language:f.get('language'),task_type:f.get('task_type')});notice('修改已保存并加入重新审核队列。');openInstruction(id);loadList('instructions');loadJobs('instructions');
      })}}
      for(const action of ['commit','reject','review'])$c('#instruction-'+action).onclick=async()=>{
        if(dialogBusy)return;try{protectDirty()}catch(e){notice(e.message);return}dialogBusy=true;
        const controls=[...$c('#corpus-dialog-body').querySelectorAll('button')],disabled=controls.map(b=>b.disabled);controls.forEach(b=>b.disabled=true);
        try{await post('/api/instructions/'+idpath(id)+'/'+action,{expected_hash:hash});notice({commit:'指令已通过受控提交入库。',reject:'候选已拒绝。',review:'已加入重新审核队列。'}[action]);openInstruction(id);loadList('instructions');loadJobs('instructions')}
        catch(e){$c('#corpus-dialog-status').textContent=human(e);notice(human(e))}
        finally{dialogBusy=false;controls.forEach((b,i)=>b.disabled=disabled[i])}
      };
    }catch(e){if(seq===dialogSeq)$c('#corpus-dialog-status').textContent=human(e);notice(human(e))}
  }
  function init() {
    // Wrap existing entry points without replacing any legacy event handlers/IDs.
    if(typeof setView==='function') {
      const oldSetView=setView;
      setView=function(next){oldSetView(next);if(next==='knowledge'){$c('#crumb').textContent='资料汇总';if(corpusTab==='results')loadList('corpus');else if(corpusTab==='nodes')loadNodes()}if(next==='qas'&&instructionTab==='instructions'){loadList('instructions');loadJobs('instructions')}if(next==='files'&&taskMode)showTasks()};
    }
    if(typeof setProcessingTab==='function') {
      const oldSetProcessingTab=setProcessingTab;
      setProcessingTab=function(next){if(next==='corpus'){showTasks();return}taskMode=false;$c('#corpus-jobs-panel').hidden=true;$c('#corpus-jobs-tab').classList.remove('active');$c('#corpus-jobs-tab').setAttribute('aria-selected','false');$c('#corpus-jobs-tab').tabIndex=-1;oldSetProcessingTab(next)};
    }
    if(typeof renderDocuments==='function') {
      const oldRenderDocuments=renderDocuments;renderDocuments=function(){if(!taskMode)oldRenderDocuments()};
    }
    if(typeof renderStats==='function') {
      const oldRenderStats=renderStats;renderStats=function(){oldRenderStats();if(taskMode&&taskCount!==null)$c('#document-count').textContent=taskCount};
    }
    if(!$c('#knowledge-view').hidden)$c('#crumb').textContent='资料汇总';
    $c('#corpus-web-open').onclick=openWeb;$c('#corpus-add-open').onclick=openWeb;
    $c('#corpus-dialog-close').onclick=closeDialog;
    $c('#corpus-dialog').addEventListener('cancel',e=>{e.preventDefault();closeDialog()});
    $c('#corpus-jobs-tab').onclick=showTasks;$c('#corpus-jobs-refresh').onclick=()=>loadJobs('corpus');
    $c('#corpus-refresh').onclick=()=>loadList('corpus');$c('#corpus-nodes-refresh').onclick=loadNodes;$c('#corpus-node-open').onclick=openNode;
    $c('#instructions-refresh').onclick=()=>{loadList('instructions');loadJobs('instructions')};
    $c('#instructions-export-csv').onclick=()=>{const subset=$c('#instructions-subset').value;if(!subset){notice('请先选择需要导出的指令子集。');return}downloadZip('/api/instructions/csv?subset='+encodeURIComponent(subset),subset+'-data.csv')};
    $c('#instructions-export').onclick=()=>{const subset=$c('#instructions-subset').value;if(!subset){notice('请先选择需要导出的指令子集。');return}downloadZip('/api/instructions/export?subset='+encodeURIComponent(subset),'instructions-'+subset+'.zip')};
    for(const type of ['corpus','instructions']) {
      $c('#'+type+'-filters').onsubmit=e=>{e.preventDefault();lists[type].page=1;loadList(type)};
      for(const dir of ['prev','next'])$c('#'+type+'-'+dir).onclick=()=>{const s=lists[type];const next=s.page+(dir==='prev'?-1:1);if(next<1||next>s.pages)return;s.page=next;loadList(type)};
    }
    for(const selector of ['#corpus-kind','#instructions-subset','#instructions-status'])$c(selector).onchange=()=>{const type=selector.startsWith('#corpus')?'corpus':'instructions';lists[type].page=1;loadList(type)};
    for(const attr of ['data-corpus-tab','data-instruction-tab']) {
      const buttons=[...document.querySelectorAll(`[${attr}]`)];
      const change=b=>attr==='data-corpus-tab'?selectCorpusTab(b.getAttribute(attr)):selectInstructionTab(b.getAttribute(attr));
      buttons.forEach((b,i)=>{b.onclick=()=>change(b);b.onkeydown=e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const next=e.key==='Home'?0:e.key==='End'?buttons.length-1:(i+(e.key==='ArrowRight'?1:-1)+buttons.length)%buttons.length;change(buttons[next]);buttons[next].focus()}});
    }
    const recordTabs=[...document.querySelectorAll('.processing-tabs [role=tab]')];
    // Capture keyboard arrows so the original two-tab handler cannot exclude Corpus.
    document.querySelector('.processing-tabs').addEventListener('keydown',e=>{const i=recordTabs.indexOf(e.target);if(i<0||!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();e.stopImmediatePropagation();const next=e.key==='Home'?0:e.key==='End'?recordTabs.length-1:(i+(e.key==='ArrowRight'?1:-1)+recordTabs.length)%recordTabs.length;recordTabs[next].click();recordTabs[next].focus()},true);
    document.addEventListener('click',async e=>{
      const b=e.target.closest('[data-c-action]');if(!b||b.disabled)return;
      const action=b.dataset.cAction,id=b.dataset.cId;
      if(action==='close'){closeDialog();return}
      if(action==='detail')return openCorpus(id);
      if(action==='generate')return openGenerate(id);
      if(action==='csv')return downloadZip('/api/corpus/csv/'+idpath(id),'corpus.csv');
      if(action==='download')return downloadZip('/api/corpus/export/'+idpath(id),'corpus.zip');
      if(action==='instruction')return openInstruction(id);
      if(action==='annotate'||action==='annotation-import')return openAnnotation(id,action==='annotation-import');
      if(action.startsWith('retry-')){const type=action==='retry-corpus'?'corpus':'instructions';b.disabled=true;try{if(type==='corpus'){const j=corpusJobs.get(String(id));if(!j)throw Error('任务记录已更新，请刷新后重试。');await post('/api/documents/'+idpath(id)+'/corpus',{kind:j.metadata?.kind||null,title:j.metadata?.title||'',source_label:j.metadata?.source_label||'',generate_after:j.generate_after===true})}else await post('/api/instructions/jobs/'+idpath(id)+'/retry',{});notice('任务已重新排队，历史结果与错误保留。');loadJobs(type)}catch(err){notice(human(err))}finally{b.disabled=false}}
    });
    // Lazy initial reads and quiet polling; no demo data and no model calls.
    if(!$c('#knowledge-view').hidden)loadList('corpus');
    setInterval(()=>{if(document.hidden)return;if(!$c('#files-view').hidden&&taskMode)loadJobs('corpus');if(!$c('#qas-view').hidden&&instructionTab==='instructions'){loadJobs('instructions');loadList('instructions')}},6000);
    window.CorpusUI=Object.freeze({showTasks,openCorpus,openWeb,openGenerate,openInstruction,selectCorpusTab,selectInstructionTab,refresh:()=>loadList('corpus')});
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init,{once:true});else init();
})();
