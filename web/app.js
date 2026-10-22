const icons={files:'<rect x="4" y="3" width="13" height="17" rx="2"/><path d="M8 8h5M8 12h5M8 16h3M17 7h3v14H7"/>',search:'<circle cx="10" cy="10" r="6"/><path d="m15 15 5 5"/>',chevron:'<path d="m7 9 5 5 5-5"/>',upload:'<path d="M12 15V3m-4 4 4-4 4 4M4 15v5h16v-5"/>',plus:'<path d="M12 5v14M5 12h14"/>',download:'<path d="M12 3v12m-4-4 4 4 4-4M4 17v4h16v-4"/>',topic:'<path d="m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5M3 16l9 5 9-5"/>',qa:'<path d="M4 4h16v12H9l-5 4V4Z"/><path d="M8 8h8M8 12h5"/>',check:'<path d="m5 12 4 4L19 6"/>',info:'<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7h.01"/>'};
const svg=name=>`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name]||icons.files}</svg>`;
const esc=value=>String(value??'').replace(/[&<>"']/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
const $=s=>document.querySelector(s);
document.querySelectorAll('[data-icon]').forEach(el=>el.innerHTML=svg(el.dataset.icon));
let semanticTimer;
const committingDocuments=new Set();
let mode='live',view='files',documents=[],topics=[],bootstrap={},opened=null,toastTimer,searchTimer,refreshSeq=0,searchSeq=0,detailSeq=0;
const labels={queued:'等待处理',parsing:'读取证据',analyzing:'分析主题',generating:'生成问答',reviewing:'核验依据',repairing:'修订问答',repair_reviewing:'复核修订',preflight:'入库校验',waiting_review:'待核对入库',insufficient:'证据不足',failed:'处理失败',completed:'已完成',completed_under_min:'已完成·数量不足',demo_complete:'演示已完成',parsed_demo:'演示仅解析'};
const statusClass=s=>s==='failed'?'failed':['waiting_review','insufficient','parsed_demo'].includes(s)?'pending':['parsing','analyzing','generating','reviewing','repairing','repair_reviewing','preflight','queued'].includes(s)?'running':'';
function toast(message){$('#toast').textContent=message;$('#toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').hidden=true,6500)}
async function api(path,options={}){const join=path.includes('?')?'&':'?';const response=await fetch(path+join+'mode='+mode,{...options,headers:{'X-QA-Session':bootstrap.session,...options.headers}});if(!response.ok){let body=await response.json().catch(()=>({}));const error=Error(body.detail||'请求失败，请稍后再试。');error.status=response.status;throw error}return response.json()}
function tag(t){const kind=t.name==='数学'?'math':t.name==='海洋能源'?'energy':'';return `<span class="tag ${kind}">${esc(t.name)}</span>`}
function updateControls(){const demo=mode==='demo';$('#table-footnote').textContent=demo?'演示每份仅展示 3 条 QA，未按正式数量要求补题':'论文目标5–7组双语；独立图片最多2组，按实际内容生成；每组对应中英2条记录';$('#choose-files').innerHTML=svg('plus')+(demo?'体验文件解析':'选择文件');document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode))}
function renderStats(){const active=topics.filter(t=>t.documents>0);const keywords=new Set(documents.flatMap(d=>d.topics.flatMap(t=>t.keywords)));const total=documents.reduce((n,d)=>n+d.qa_count,0);const stats=[['文件总数',documents.length,'份','files'],['正式主题',active.length,'个','topic'],['已入库 QA',total,'条','qa']];$('#stats').innerHTML=stats.map(([name,value,unit,icon])=>`<div class="stat"><div class="stat-label">${name}</div><div class="stat-icon">${svg(icon)}</div><div class="stat-bottom"><span class="stat-value">${value}</span><span class="stat-unit">${unit}</span></div></div>`).join('');$('#nav-count').textContent=documents.length;$('#document-count').textContent=documents.length;const suggestions=[...new Set([...active.map(t=>t.name),...keywords])].slice(0,4);$('#search-suggestions').hidden=!suggestions.length;$('#search-suggestions').innerHTML='<span>试试搜索</span>'+suggestions.map(term=>`<button data-query="${esc(term)}">${esc(term)}</button>`).join('');bindQueries($('#search-suggestions'))}
function documentCommitButton(d){
  if(mode!=='live')return '';
  const busy=committingDocuments.has(d.id);
  const finished=['waiting_review','insufficient','completed_under_min','completed'].includes(d.status);
  const pending=d.pending_commit_count??Math.max(0,d.candidate_count-d.qa_count);
  const enabled=finished&&pending>0&&!busy;
  const title=busy?'正在入库':!finished?'请先完成文件处理和审核':pending>0?'将该文件的待入库 QA 一次提交':'没有待入库的 QA';
  if(busy)return `<button class="button ghost file-commit is-busy" data-commit-document="${d.id}" aria-label="正在入库 ${esc(d.name)}" title="${title}" disabled><span class="commit-spinner" aria-hidden="true"></span>入库中…</button>`;
  if(enabled)return `<button class="button primary file-commit is-ready" data-commit-document="${d.id}" aria-label="一键入库 ${esc(d.name)}" title="${title}">${svg('download')}一键入库</button>`;
  if(finished&&d.qa_count>0)return `<button class="button ghost file-commit is-done" data-commit-document="${d.id}" aria-label="${esc(d.name)} 已全部入库" title="该文件的 QA 已全部入库" disabled>${svg('check')}已入库</button>`;
  const waiting=d.status==='failed'?'失败待重试':pending>0?'一键入库':'无待入库';
  return `<button class="button ghost file-commit" data-commit-document="${d.id}" aria-label="${esc(d.name)}" title="${title}" disabled>${waiting}</button>`;
}
function renderDocuments(){const term=$('#file-filter').value.trim().toLowerCase();const list=documents.filter(d=>d.name.toLowerCase().includes(term));$('#document-rows').innerHTML=list.length?list.map(d=>`<tr data-document="${d.id}" tabindex="0" role="button" aria-label="查看 ${esc(d.name)}"><td class="document-file-cell"><div class="file-cell"><div class="file-icon">${svg('files')}<span>${esc(d.name.split('.').pop().toUpperCase())}</span></div><div><div class="file-name">${esc(d.name)}</div><div class="file-summary">${esc(d.summary||d.error||'正在排队，完成后自动显示梗概与主题。')}</div><div class="file-meta">${d.pages?d.pages+' 页 · ':''}${d.unit_count} 个证据单元${d.seconds?' · '+d.seconds+' 秒':''}</div></div></div>${documentCommitButton(d)}</td><td>${d.topics.map(tag).join('')||'<span style="color:#a5b2a8">待识别</span>'}</td><td><span class="status ${statusClass(d.status)}">${esc(labels[d.status]||d.status)}</span>${d.progress?`<div class="processing-progress">${esc(d.progress)}</div>`:''}</td><td><div class="qa-number">${d.qa_count} 条<small>${d.qa_pair_count} 组双语已入库 · ${d.candidate_count-d.qa_count>0?(d.candidate_count-d.qa_count)+' 条候选':'已入库'}</small></div></td><td><span class="arrow-button">↗</span></td></tr>`).join(''):`<tr><td colspan="5"><div class="empty-state"><strong>${term?'没有匹配的文件':'这里等待你的第一份资料'}</strong>${term?'换个文件名称试试。':'选择上方文件，或直接拖入 PDF 和图片。'}</div></td></tr>`;bindDocs($('#document-rows'));$('#document-rows').querySelectorAll('[data-commit-document]').forEach(b=>b.onclick=e=>{e.stopPropagation();commitDocument(b.dataset.commitDocument)})}
function bindDocs(root){root.querySelectorAll('[data-document]').forEach(el=>{el.onclick=()=>openDocument(el.dataset.document);el.onkeydown=e=>{if(e.key==='Enter'&&e.target===el)openDocument(el.dataset.document)}})}
async function commitDocument(id){
  if(committingDocuments.has(id))return;
  committingDocuments.add(id);renderDocuments();
  try{
    const result=await api('/api/documents/'+id+'/approve',{method:'POST'});
    toast(result.committed_count?`已将该文件的 ${result.committed_count} 条 QA 一键入库。`:'该文件没有待入库的 QA。');
    await refresh();
  }catch(e){toast(e.message)}finally{committingDocuments.delete(id);renderDocuments()}
}
async function refresh(){const expectedMode=mode,seq=++refreshSeq;try{const result=await Promise.all([api('/api/documents?summary_only=true'),api('/api/topics')]);if(expectedMode!==mode||seq!==refreshSeq)return;[documents,topics]=result;renderStats();renderDocuments()}catch(e){if(expectedMode===mode)toast(e.message)}}
function setView(next){
  view=['files','search','qas'].includes(next)?next:'files';
  for(const key of ['files','search','qas'])$('#'+key+'-view').hidden=view!==key;
  document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===view));
  const names={files:'文件处理',search:'主题模糊搜索',qas:'QA汇总'};
  const descriptions={files:'从一份资料开始，生成有依据的中英双语问答。',search:'用较宽主题整理资料，用具体方向精准查找。',qas:'查看和编辑已入库问答，让每一次修改都有依据。'};
  $('#crumb').textContent=$('#page-title').textContent=names[view];
  $('#page-description').textContent=descriptions[view];
  if(view==='search')runSearch();
  if(view==='qas')loadQAs();
  location.hash=view;
}
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>setView(b.dataset.view));
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=async()=>{mode=b.dataset.mode;documents=[];topics=[];++searchSeq;closeDrawer();updateControls();renderStats();renderDocuments();await refresh();if(view==='search')runSearch()});
$('#file-filter').oninput=renderDocuments;
$('#choose-files').onclick=()=>$('#file-input').click();
$('#file-input').onchange=e=>upload(e.target.files);
async function upload(files){if(!files.length)return;const form=new FormData();Array.from(files).forEach(f=>form.append('files',f));$('#choose-files').disabled=true;try{const result=await api('/api/imports',{method:'POST',body:form});const errors=result.filter(r=>r.error);toast(errors.length?errors.map(e=>e.name+'：'+e.error).join('；'):`${result.filter(r=>r.created).length} 份文件已加入队列${result.some(r=>r.created===false)?'，重复内容已复用':''}。`);await refresh()}catch(e){toast(e.message)}finally{$('#choose-files').disabled=false;$('#file-input').value=''}}
const drop=$('#drop-zone');['dragenter','dragover'].forEach(type=>drop.addEventListener(type,e=>{e.preventDefault();drop.classList.add('dragover')}));['dragleave','drop'].forEach(type=>drop.addEventListener(type,e=>{e.preventDefault();drop.classList.remove('dragover')}));drop.addEventListener('drop',e=>upload(e.dataTransfer.files));
function download(query=''){window.open('/api/export?mode='+mode+query,'_blank')}
$('#export-all').onclick=()=>download();
async function runSearch(){const q=$('#theme-query').value.trim(),expectedMode=mode,seq=++searchSeq;clearTimeout(semanticTimer);$('#semantic-section').hidden=!q;$('#semantic-results').innerHTML='';$('#semantic-status').textContent=q?'正在查找相关主题…':'';if(q)loadRecommendations(q,expectedMode,seq);try{const result=await api('/api/search?q='+encodeURIComponent(q));if(expectedMode!==mode||seq!==searchSeq)return;$('#search-summary').textContent=q?`“${q}” · ${result.length} 个直接匹配主题`:'主题与涵盖文件';$('#search-results').innerHTML=result.length?result.map(r=>`<article class="topic-result"><div class="topic-result-head"><div><div class="topic-result-name">${svg('topic')}${esc(r.topic.name)}<small>${esc(r.reason)}</small></div><p class="topic-definition">${esc(r.topic.definition)}</p><div class="keywords">${r.terms.map(term=>`<button class="keyword" data-query="${esc(term)}">${esc(term)}</button>`).join('')}</div></div><div class="topic-result-count">${r.documents.length} 份涵盖文件 <button class="button ghost" data-export-topic="${r.topic.id}">${svg('download')}</button></div></div><div class="result-files">${r.documents.map(d=>`<div class="result-file" data-document="${d.id}" tabindex="0"><div class="file-cell"><div class="file-icon">${svg('files')}<span>${esc(d.name.split('.').pop().toUpperCase())}</span></div><div><div class="file-name">${esc(d.name)}</div><div class="file-summary">${esc(d.summary)}</div></div></div><span class="status ${statusClass(d.status)}">${esc(labels[d.status]||d.status)}</span><span class="arrow-button">↗</span></div>`).join('')||'<div class="empty-state">此主题暂未关联资料</div>'}</div></article>`).join(''):'<div class="table-card empty-state"><strong>没有直接匹配的主题</strong>下方可查看相似主题推荐，或更换关键词。</div>';bindDocs($('#search-results'));bindQueries($('#search-results'));$('#search-results').querySelectorAll('[data-export-topic]').forEach(b=>b.onclick=()=>download('&topic_id='+b.dataset.exportTopic))}catch(e){toast(e.message)}}

async function loadRecommendations(q,expectedMode,seq){
  if(expectedMode!==mode||seq!==searchSeq||view!=='search')return;
  if(document.hidden){semanticTimer=setTimeout(()=>loadRecommendations(q,expectedMode,seq),2500);return}
  try{
    const data=await api('/api/search/semantic?q='+encodeURIComponent(q));
    if(expectedMode!==mode||seq!==searchSeq||view!=='search')return;
    $('#semantic-status').textContent=data.message;
    $('#semantic-results').innerHTML=data.recommendations.map((r,i)=>`<button class="semantic-card" data-query="${esc(r.topic.name)}"><span class="semantic-rank">${i+1}</span><span class="semantic-copy"><strong>${esc(r.topic.name)}</strong><span>${esc(r.topic.definition)}</span><small>${esc(r.reason)} · ${r.topic.documents} 份关联文件</small></span><span class="semantic-arrow">↗</span></button>`).join('');
    bindQueries($('#semantic-results'));
    if(['pending','busy'].includes(data.status))semanticTimer=setTimeout(()=>loadRecommendations(q,expectedMode,seq),2500);
  }catch(e){if(expectedMode===mode&&seq===searchSeq)$('#semantic-status').textContent='语义推荐暂不可用，基础匹配仍可使用。'}
}

function bindQueries(root){root.querySelectorAll('[data-query]').forEach(b=>b.onclick=()=>{$('#theme-query').value=b.dataset.query;runSearch()})}bindQueries(document);
$('#search-button').onclick=runSearch;$('#theme-query').onkeydown=e=>{if(e.key==='Enter')runSearch()};$('#theme-query').oninput=()=>{++searchSeq;clearTimeout(semanticTimer);clearTimeout(searchTimer);searchTimer=setTimeout(runSearch,350)};
function candidateCard(c,i,d){
  const blocked=c.status==='needs_review';
  const modelPassed=['supported','independent','scope_correct','knowledge_value','distinct','topic_correct','language_correct','translation_consistent'].every(k=>c.review[k]===true);
  const reviewLabel=blocked?(modelPassed&&c.review.local_flags?.length?'本地检查未通过（模型审核已通过）':'审核未通过原因'):'核验说明';
  return `<article class="qa-card"><div class="qa-head"><span class="qa-index">${c.row.language==='zh'?'中文 · zh':'English · en'}</span><span class="status ${c.status==='needs_review'?'pending':''}">${c.status==='approved'?'已入库':c.status==='rejected'?'已排除':c.status==='needs_review'?'修订审核中':'待入库'}</span>${tag(d.topics.find(t=>t.id===(c.canonical_topic_id||c.topic_id))||{name:'待归类'})}</div><div class="qa-field question"><label>input</label><div>${esc(c.row.input)}</div></div><div class="qa-field answer"><label>output</label><div>${esc(c.row.output)}</div></div><div class="qa-field muted"><label>thinking</label><div>${esc(c.row.thinking)}</div></div><div class="qa-field path"><label>source</label><div>${esc(c.row.source)}</div></div><div class="qa-field path"><label>image</label><div>${esc(c.row.image||'""（纯文字依据）')}</div></div><div class="qa-field muted"><label>language</label><div>${esc(c.row.language)}</div></div>${c.review.note?`<div class="review-note">${reviewLabel}：${esc(c.review.note)}</div>`:''}${!['approved','rejected'].includes(c.status)&&mode==='live'?`<div class="qa-actions"><div><button class="button ghost" data-rereview="${c.id}" data-hash="${c.row_hash}">重新核验${c.pair_id?'本组':''}</button><button class="button ghost" data-reject="${c.id}" data-hash="${c.row_hash}">不采用</button><button class="button primary" data-approve="${c.id}" data-hash="${c.row_hash}" ${blocked?'disabled':''}>${blocked?'等待修订审核':'确认入库'}</button></div></div>`:'<div class="qa-actions"><span class="qa-approved">'+(mode==='demo'?'合成演示记录':c.status==='rejected'?'已排除，不进入导出':'已入库')+'</span></div>'}</article>`}
function pairedCards(candidates,d){
  const groups=new Map();
  for(const c of candidates){const key=c.pair_id||c.id;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(c)}
  return Array.from(groups.values()).map((versions,i)=>{
    versions.sort((a,b)=>(a.row.language==='zh'?0:1)-(b.row.language==='zh'?0:1));
    const paired=Boolean(versions[0].pair_id);
    const complete=paired&&versions.length===2&&new Set(versions.map(c=>c.row.language)).size===2;
    return `<section class="qa-pair"><div class="pair-heading">${paired?'双语组':'历史单语'} ${String(i+1).padStart(2,'0')}<span>${complete?(versions.some(c=>c.status==='rejected')?'一版已排除，双语未完整入库':'中文 / English · 同一知识点'):paired?'双语版本缺失，需检查':'未自动补译，不计入双语目标'}</span></div><div class="qa-versions">${versions.map(c=>candidateCard(c,i,d)).join('')}</div></section>`;
  }).join('');
}
function evidenceCard(u){
  const url=u.asset_id?`/api/assets/${encodeURIComponent(u.asset_id)}?mode=${encodeURIComponent(mode)}`:'';
  return `<details class="evidence-toggle"><summary>${esc(u.locator)} · ${url?'图文证据':'文字证据'}</summary>${url?`<a href="${esc(url)}" target="_blank" rel="noopener" aria-label="查看完整图片"><img loading="lazy" decoding="async" class="evidence-img" src="${esc(url)}" alt="原始文献完整图片"></a>`:''}<div class="evidence-text">${esc(u.text)}</div></details>`;
}
async function openDocument(id){const expectedMode=mode,seq=++detailSeq;try{const d=await api('/api/documents/'+id);if(expectedMode!==mode||seq!==detailSeq)return;opened=id;$('#drawer-title').textContent=d.name;const candidates=d.candidates;$('#drawer-body').innerHTML=`<div class="detail-summary"><div class="detail-label">简短梗概</div><p>${esc(d.summary||'梗概尚未完成。')}</p>${d.topics.map(tag).join('')}<div class="file-meta">${esc(d.topics.flatMap(t=>t.keywords).join(' / '))}</div></div>${d.error?`<div class="error-box">${esc(d.error)}${d.status==='failed'?'<button class="button ghost" id="retry-document">重试处理</button>':''}</div>`:''}${d.seconds?`<div class="file-meta processing-timings">本次处理 ${d.seconds} 秒${timingText(d)?'<br>'+esc(timingText(d)):''}</div>`:''}<div class="detail-section"><h3>原始证据</h3><a class="button secondary" href="/api/originals/${d.id}?mode=${mode}" target="_blank">${svg('download')}原始文件</a></div>${d.units.map(u=>evidenceCard(u)).join('')||'<div class="empty-state">等待证据准备完成</div>'}<div class="detail-section"><h3>六字段 QA <span class="tag">${d.bilingual_count} 组双语 · ${candidates.length} 条</span></h3><button class="button secondary" id="export-document">${svg('download')}导出已入库</button></div>${d.discarded_pair_count?`<div class="review-note">${d.discarded_pair_count} 组草稿经自动修订仍未通过审核，已保留处理记录，不计入交付或导出。</div>`:''}${mode==='live'&&['waiting_review','insufficient','completed_under_min'].includes(d.status)?'<div class="detail-section"><button class="button secondary" id="optimize-document">优化问答并重新审核</button></div>':''}${mode==='demo'?'<div class="mode-banner">预置合成问答 · 未调用 AI · 每份仅展示 3 条 · 不进入真实工作区</div>':''}${pairedCards(candidates,d)||'<div class="table-card empty-state"><strong>尚无问答候选</strong>生成完成后将在这里显示六个字段。</div>'}`;if($('#optimize-document'))$('#optimize-document').onclick=async()=>{const b=$('#optimize-document');b.disabled=true;try{await api('/api/documents/'+id+'/optimize',{method:'POST'});toast('已加入优化队列，将复用原证据和已通过的问答。');closeDrawer();await refresh()}catch(e){toast(e.message);b.disabled=false}};$('#drawer-overlay').hidden=false;document.body.style.overflow='hidden';$('#export-document').onclick=()=>download('&document_id='+id);$('#drawer-body').querySelectorAll('[data-rereview]').forEach(b=>b.onclick=async()=>{b.disabled=true;try{await api('/api/candidates/'+b.dataset.rereview+'/rereview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reviewed:false,expected_hash:b.dataset.hash})});toast('已排队核验本组；未通过时自动修订并重新审核。');closeDrawer();await refresh()}catch(e){toast(e.message);b.disabled=false}});$('#drawer-body').querySelectorAll('[data-reject]').forEach(b=>b.onclick=async()=>{b.disabled=true;try{await api('/api/candidates/'+b.dataset.reject+'/reject',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reviewed:false,expected_hash:b.dataset.hash})});toast('已排除该候选，记录仍保留。');await refresh();await openDocument(id)}catch(e){toast(e.message);b.disabled=false}});if($('#retry-document'))$('#retry-document').onclick=async()=>{try{await api('/api/documents/'+id+'/retry',{method:'POST'});toast('已重新排队。');closeDrawer();refresh()}catch(e){toast(e.message)}};$('#drawer-body').querySelectorAll('[data-approve]').forEach(b=>b.onclick=async()=>{b.disabled=true;try{await api('/api/candidates/'+b.dataset.approve+'/approve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reviewed:true,expected_hash:b.dataset.hash})});toast('已通过受控提交入库。');await refresh();await openDocument(id)}catch(e){toast(e.message);b.disabled=false}})}catch(e){toast(e.message)}}
function timingText(d){const names={source_validation:'来源检查',parsing:'解析',analyzing:'主题识别',generating:'生成',reviewing:'审核',repairing:'修订',repair_reviewing:'修订审核',preflight:'入库校验'};return Object.entries(d.timings||{}).map(([k,v])=>`${names[k]||k} ${Number(v).toFixed(1)} 秒`).join(' · ')}
function closeDrawer(){++detailSeq;$('#drawer-overlay').hidden=true;document.body.style.overflow='';opened=null}$('#close-drawer').onclick=closeDrawer;$('#drawer-overlay').onclick=e=>{if(e.target===$('#drawer-overlay'))closeDrawer()};document.addEventListener('keydown',e=>{if(e.key==='Escape'){closeDrawer();$('#connection-popover').hidden=true}});
$('#connection').onclick=()=>{const el=$('#connection-popover');el.hidden=!el.hidden;el.innerHTML=`<strong>模型连接配置</strong><p>地址：${esc(bootstrap.base_url)}<br>文本模型：${esc(bootstrap.text_model)}<br>视觉模型：${esc(bootstrap.vision_model||'尚未配置，图片任务会明确停止')}<br>密钥：${bootstrap.key_configured?'本机已配置（不显示内容）':'未配置'}</p>`};
async function init(){try{bootstrap=await fetch('/api/bootstrap').then(r=>r.json());$('#connection-text').textContent=bootstrap.key_configured?'模型就绪':'模型待配置';$('#upload-limit').textContent=`每份最多 ${bootstrap.limits.max_file_mb} MB · ${bootstrap.limits.max_pages?'PDF 最多 '+bootstrap.limits.max_pages+' 页':'PDF 页数不限'}`;updateControls();await refresh();if(['#search','#qas'].includes(location.hash))setView(location.hash.slice(1));setInterval(async()=>{if(document.hidden)return;const working=documents.some(d=>['queued','parsing','analyzing','generating','reviewing','repairing','repair_reviewing','preflight'].includes(d.status));if(working)await refresh()},2500)}catch(e){toast('本地服务尚未连接，请刷新重试。')}}init();

window.addEventListener('hashchange',()=>{const next=['#search','#qas'].includes(location.hash)?location.hash.slice(1):'files';if(next!==view)setView(next)});

let qaPage=1,qaListSeq=0,qaEditorSeq=0,qaData=null,qaEditing=null,qaSubmitting=false;
const revisionLabels={queued:'修改待审核',reviewing:'修改审核中',applied:'修改已保存',rejected:'修改未通过',failed:'修改审核异常'};
async function loadQAs(){
  const seq=++qaListSeq,expectedMode=mode;
  try{
    const data=await api('/api/qas?page='+qaPage);
    if(seq!==qaListSeq||expectedMode!==mode)return;
    qaData=data;qaPage=data.page;
    $('#qa-total').textContent=data.total;
    $('#qas-view .table-scroll').hidden=!data.items.length;$('#qa-empty').hidden=Boolean(data.items.length);
    $('#qa-page-info').textContent=`共 ${data.total} 条 · 第 ${data.page} / ${data.pages} 页`;
    $('#qa-prev').disabled=data.page<=1;$('#qa-next').disabled=data.page>=data.pages;
    $('#qa-rows').innerHTML=data.items.length?data.items.map(item=>`<tr>${['input','thinking','output','source','image','language'].map(k=>`<td><div class="qa-cell-text ${['source','image'].includes(k)?'qa-path':''}">${esc(item.row[k]||(k==='image'?'无图片':''))}</div></td>`).join('')}<td><button class="button ghost qa-edit-button" data-edit-qa="${esc(item.id)}">查看 / 编辑</button>${item.revision?`<span class="qa-row-status">${esc(revisionLabels[item.revision.status]||'')}</span>`:''}</td></tr>`).join(''):'<tr><td colspan="7"><div class="empty-state"><strong>还没有已入库的问答</strong>在文件处理中完成审核并入库后，问答会显示在这里。</div></td></tr>';
    $('#qa-rows').querySelectorAll('[data-edit-qa]').forEach(b=>b.onclick=()=>openQAEditor(b.dataset.editQa));
  }catch(e){if(seq===qaListSeq){$('#qa-page-info').textContent='读取失败，请点击刷新重试。';toast(e.message)}}
}
async function openQAEditor(id){
  const seq=++qaEditorSeq,expectedMode=mode;
  try{
    const detail=await api('/api/qas/'+encodeURIComponent(id));
    if(seq!==qaEditorSeq||expectedMode!==mode)return;
    qaEditing={id,...detail};
    for(const v of detail.versions){const draft=detail.draft_versions.find(d=>d.id===v.id);if(draft)v.row=draft.row}
    const pending=['queued','reviewing'].includes(detail.revision?.status);
    $('#qa-editor-title').textContent=detail.versions.length===2?'编辑中英双语问答':'编辑历史单语问答';
    $('#qa-editor-body').innerHTML=`<div class="detail-summary"><div class="detail-label">${esc(detail.document_name)}</div><p>中英版本一起核验。来源、图片和语言沿用入库记录；修改通过前，表格与导出保留原内容。</p></div>${evidenceCard(detail.evidence)}<div class="qa-edit-versions">${detail.versions.map(v=>`<section class="qa-edit-version" data-qa-version="${esc(v.id)}"><h3>${v.row.language==='zh'?'中文 · zh':'English · en'}</h3>${[['input','问题'],['thinking','依据摘要'],['output','答案']].map(([k,label])=>`<label for="edit-${v.id}-${k}">${k} · ${label}</label><textarea id="edit-${v.id}-${k}" name="${k}" required maxlength="20000" ${pending?'disabled':''}>${esc(v.row[k])}</textarea>`).join('')}${['source','image','language'].map(k=>`<label>${k} · 只读</label><div class="qa-readonly">${esc(v.row[k]||(k==='image'?'无图片':''))}</div>`).join('')}</section>`).join('')}</div>`;
    $('#qa-edit-save').disabled=pending||mode!=='live';
    $('#qa-edit-status').textContent=detail.revision?detail.revision.note:'修改后点击“保存并审核”。';
    if(detail.revision?.status==='failed'){
      const b=document.createElement('button');b.type='button';b.className='button ghost';b.textContent='重试上次修改';
      b.onclick=async()=>{b.disabled=true;try{await api('/api/qas/'+encodeURIComponent(id)+'/retry',{method:'POST'});await openQAEditor(id);await loadQAs()}catch(e){toast(e.message);b.disabled=false}};
      $('#qa-edit-status').append(' ',b);
    }
    if(!$('#qa-editor').open)$('#qa-editor').showModal();
  }catch(e){toast(e.message)}
}
function closeQAEditor(){if(qaSubmitting)return;++qaEditorSeq;$('#qa-editor').close();qaEditing=null}
$('#qa-editor-close').onclick=closeQAEditor;$('#qa-edit-cancel').onclick=closeQAEditor;
$('#qa-editor').addEventListener('cancel',e=>{e.preventDefault();closeQAEditor()});
$('#qa-edit-form').onsubmit=async e=>{
  e.preventDefault();if(!qaEditing||qaSubmitting)return;
  const id=qaEditing.id;
  const versions=qaEditing.versions.map(v=>{
    const root=document.querySelector(`[data-qa-version="${v.id}"]`);
    return {id:v.id,expected_hash:v.row_hash,...Object.fromEntries(['input','thinking','output'].map(k=>[k,root.querySelector(`[name="${k}"]`).value]))};
  });
  qaSubmitting=true;$('#qa-edit-save').disabled=true;
  try{
    const result=await api('/api/qas/'+encodeURIComponent(id)+'/revisions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({versions,expected_edit_version:qaEditing.edit_version})});
    qaSubmitting=false;
    if(result.unchanged){toast('内容没有变化。');$('#qa-edit-save').disabled=false;return}
    toast('修改已保存为待审核版本，通过后自动更新正式内容。');closeQAEditor();await loadQAs();
  }catch(error){$('#qa-edit-status').textContent=error.message;$('#qa-edit-save').disabled=error.status===409;if(error.status===409){$('#qa-edit-status').append(' 当前输入仍保留，请先复制需要保留的内容。');const b=document.createElement('button');b.type='button';b.className='button ghost';b.textContent='刷新最新版本';b.onclick=()=>openQAEditor(id);$('#qa-edit-status').append(' ',b)}}finally{qaSubmitting=false}
};
$('#qa-refresh').onclick=()=>loadQAs();$('#qa-export').onclick=()=>download();
$('#qa-prev').onclick=()=>{if(qaPage>1){qaPage--;loadQAs()}};
$('#qa-next').onclick=()=>{if(qaData&&qaPage<qaData.pages){qaPage++;loadQAs()}};
setInterval(async()=>{
  if(document.hidden)return;
  if(view==='qas'&&qaData?.items.some(i=>['queued','reviewing'].includes(i.revision?.status))){await loadQAs();await refresh()}
  if(qaEditing&&['queued','reviewing'].includes(qaEditing.revision?.status)){
    const id=qaEditing.id,seq=qaEditorSeq;
    try{const data=await api('/api/qas/'+encodeURIComponent(id));if(seq===qaEditorSeq&&qaEditing?.id===id){$('#qa-edit-status').textContent=data.revision?.note||'';if(!['queued','reviewing'].includes(data.revision?.status))await openQAEditor(id)}}catch(e){/* Keep the current editor available during connection failures. */}
  }
},2500);
