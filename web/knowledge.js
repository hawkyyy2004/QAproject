let knowledgePage=1,knowledgeSeq=0;
let knowledgeJobsPage=1,knowledgeJobsSeq=0;
const knowledgeJobLabels={queued:'等待处理',running:'处理中',completed:'处理完成',failed:'处理失败'};
function setProcessingTab(next){
  processingTab=next==='knowledge'?'knowledge':'qa';
  for(const button of document.querySelectorAll('[data-processing-tab]')){
    const active=button.dataset.processingTab===processingTab;
    button.classList.toggle('active',active);button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1;
  }
  $('#qa-records-panel').hidden=processingTab!=='qa';$('#knowledge-records-panel').hidden=processingTab!=='knowledge';
  $('#export-all').hidden=processingTab!=='qa';
  $('#processing-description').textContent=processingTab==='qa'?'点击文件，查看梗概、原始证据和六字段问答。':'点击文件，查看知识条目整理进度、审核结果及原文。';
  if(processingTab==='knowledge')loadKnowledgeJobs();else{renderStats();renderDocuments()}
}
document.querySelectorAll('[data-processing-tab]').forEach(button=>{
  button.onclick=()=>setProcessingTab(button.dataset.processingTab);
  button.onkeydown=e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){
    e.preventDefault();const next=e.key==='Home'?'qa':e.key==='End'?'knowledge':processingTab==='qa'?'knowledge':'qa';
    setProcessingTab(next);document.querySelector(`[data-processing-tab="${next}"]`).focus();
  }};
});
async function loadKnowledgeJobs(){
  const seq=++knowledgeJobsSeq,expectedMode=mode;
  try{
    const data=await api('/api/knowledge/jobs?page='+knowledgeJobsPage+'&q='+encodeURIComponent($('#file-filter').value.trim()));
    if(seq!==knowledgeJobsSeq||expectedMode!==mode||processingTab!=='knowledge')return;
    knowledgeJobsPage=data.page;$('#document-count').textContent=data.total;
    $('#knowledge-job-rows').innerHTML=data.items.map(j=>`<tr data-knowledge-job="${esc(j.document_id)}" tabindex="0" role="button" aria-label="查看 ${esc(j.name)} 的知识条目处理记录"><td><div class="file-name">${esc(j.name)}</div><div class="file-meta">${esc(j.error||j.progress||'等待处理')}</div></td><td><span class="status ${j.status==='failed'?'failed':j.status==='running'?'running':''}">${esc(knowledgeJobLabels[j.status]||j.status)}</span></td><td>${j.approved_count} 个实体<small class="file-meta">${j.rejected_count?`未通过 ${j.rejected_count} 条，仅留内部审核记录`:''}</small></td><td><button class="button ghost" data-knowledge-job="${esc(j.document_id)}">查看记录</button>${j.status==='failed'?`<button class="button ghost" data-knowledge-doc="${esc(j.document_id)}">重试</button>`:''}</td></tr>`).join('')||'<tr><td colspan="4"><div class="empty-state"><strong>暂无知识条目处理记录</strong>导入时选择“整理知识条目”（可与其他方式同时勾选），也可在已有文件详情中发起整理。</div></td></tr>';
    $('#knowledge-job-rows').querySelectorAll('tr[data-knowledge-job]').forEach(row=>row.onkeydown=e=>{if(e.key==='Enter'&&e.target===row)openKnowledgeJob(row.dataset.knowledgeJob)});
    $('#knowledge-jobs-page-info').textContent=`共 ${data.total} 份 · 第 ${data.page} / ${data.pages} 页`;
    $('#knowledge-jobs-prev').disabled=data.page<=1;$('#knowledge-jobs-next').disabled=data.page>=data.pages;
  }catch(e){if(seq===knowledgeJobsSeq){$('#knowledge-jobs-page-info').textContent='读取失败，请重试。';toast(e.message)}}
}
$('#knowledge-jobs-prev').onclick=()=>{knowledgeJobsPage--;loadKnowledgeJobs()};
$('#knowledge-jobs-next').onclick=()=>{knowledgeJobsPage++;loadKnowledgeJobs()};
async function openKnowledgeJob(docId,page=1,rejectedPage=1){
  const seq=++detailSeq,expectedMode=mode;
  try{
    const [job,entries,rejected]=await Promise.all([api('/api/knowledge/jobs/'+encodeURIComponent(docId)),api('/api/knowledge?document_id='+encodeURIComponent(docId)+'&page='+page),api('/api/knowledge/jobs/'+encodeURIComponent(docId)+'/rejected?page='+rejectedPage)]);
    if(seq!==detailSeq||expectedMode!==mode||view!=='files'||processingTab!=='knowledge')return;
    $('#drawer-title').textContent=job.name;
    $('#drawer-body').innerHTML=`<div class="detail-summary"><strong>${esc(knowledgeJobLabels[job.status]||job.status)}</strong><p>${esc(job.progress||'等待处理')}</p><p>汇总 ${job.approved_count} 个实体 · ${job.fact_count} 条已审核事实 · 历史未通过 ${job.rejected_count} 条</p>${job.error?`<div class="error-box">${esc(job.error)}</div>`:''}${job.status==='failed'?`<button class="button secondary" data-knowledge-doc="${esc(docId)}">重试整理</button>`:''}</div><div class="detail-section"><h3>未通过记录（含已修复） · ${rejected.total} 条</h3>${rejected.items.map(r=>`<article class="qa-card"><strong>${esc(r.entity_name)}</strong><p>${esc(r.description)}</p><p class="review-note">${esc(r.reason)} · ${esc(({queued:'等待修复',running:'修复中',retryable:'修复中断，可续跑',approved:'已修复',failed:'修复未通过',rejected:'待处理'})[r.status]||r.status)}</p>${r.history.map(h=>h.error?`<p class="review-note">${esc(h.error)}</p>`:'').join('')}${r.repairable&&r.status!=='approved'&&(!['queued','running'].includes(r.status))&&(r.attempts<2||r.status==='retryable')?`<button class="button ghost" data-knowledge-repair="${esc(r.id)}" data-document="${esc(docId)}" data-expected="${esc(r.expected_hash)}">${r.status==='retryable'?'继续修复':'单独修复并重审'}</button>`:''}</article>`).join('')||'<p>本页无未通过条目。</p>'}<div class="table-footer"><span>第 ${rejected.page} 页</span><button class="button ghost" id="rejected-prev" ${rejectedPage<=1?'disabled':''}>上一页</button><button class="button ghost" id="rejected-next" ${rejectedPage*20>=rejected.total?'disabled':''}>下一页</button></div></div><div class="detail-section"><h3>知识条目</h3><a class="button secondary" href="/api/originals/${encodeURIComponent(docId)}?mode=${encodeURIComponent(mode)}" target="_blank" rel="noopener">查看原始文件</a></div>${entries.items.map(item=>`<article class="qa-card"><div class="qa-head"><strong>${esc(item.entity_name)}</strong><span class="tag">${esc(knowledgeTypes[item.entity_type])}</span><span class="tag">${esc(item.topic)}</span></div><p>${esc(item.entity_description)}</p><small class="file-meta">积累 ${item.fact_count||1} 条已审核事实</small><button class="button ghost" data-knowledge-entry="${esc(item.id)}">查看原文及审核说明</button></article>`).join('')||'<div class="empty-state">暂无审核通过的条目；处理完成后会显示在这里。</div>'}<div class="table-footer"><span>共 ${entries.total} 条 · 第 ${entries.page} / ${entries.pages} 页</span><div><button class="button ghost" id="job-entries-prev" ${entries.page<=1?'disabled':''}>上一页</button><button class="button ghost" id="job-entries-next" ${entries.page>=entries.pages?'disabled':''}>下一页</button></div></div>`;
    $('#rejected-prev').onclick=()=>openKnowledgeJob(docId,page,rejectedPage-1);$('#rejected-next').onclick=()=>openKnowledgeJob(docId,page,rejectedPage+1);
    $('#job-entries-prev').onclick=()=>openKnowledgeJob(docId,entries.page-1,rejectedPage);$('#job-entries-next').onclick=()=>openKnowledgeJob(docId,entries.page+1,rejectedPage);
    $('#drawer-overlay').hidden=false;document.body.style.overflow='hidden';$('#drawer-body').scrollTop=0;
  }catch(e){toast(e.message)}
}
const knowledgeTypes={Location:'地点',Instrument:'仪器',Organism:'生物',Process:'过程',Phenomenon:'现象',Substance:'物质',Property:'属性',Theory:'理论'};
for(const [value,label] of Object.entries(knowledgeTypes)){
  const option=document.createElement('option');option.value=value;option.textContent=label;$('#knowledge-type').append(option);
}
function knowledgeQuery(){
  return new URLSearchParams({q:$('#knowledge-query').value.trim(),entity_type:$('#knowledge-type').value});
}
async function loadKnowledge(){
  const seq=++knowledgeSeq,expectedMode=mode;
  $('#knowledge-page-info').textContent='正在读取知识条目…';
  try{
    const data=await api('/api/knowledge?'+knowledgeQuery()+'&page='+knowledgePage);
    if(seq!==knowledgeSeq||expectedMode!==mode)return;
    knowledgePage=data.page;$('#knowledge-total').textContent=data.total;
    const fields=['entity_name','entity_type','entity_description','page_urls','page_titles','full_text','full_text_length'];
    $('#knowledge-rows').innerHTML=data.items.map(item=>`<tr>${fields.map(key=>{
      const value=key==='full_text'?item.full_text_preview:key==='entity_type'?`${item.entity_type} · ${knowledgeTypes[item.entity_type]||''}`:item[key];
      return `<td><div class="knowledge-cell-text ${['page_urls','page_titles'].includes(key)?'knowledge-source-cell':''}">${esc(value??'')||'—'}</div></td>`;
    }).join('')}<td><button class="button ghost" data-knowledge-entry="${esc(item.id)}">查看原文${item.fact_count>1?` · ${item.fact_count} 条事实`:""}</button></td></tr>`).join('')||'<tr><td colspan="8"><div class="empty-state"><strong>暂无匹配的知识条目</strong>在文件导入时选择“整理知识条目”，或在已有文件详情中发起整理。</div></td></tr>';
    $('#knowledge-page-info').textContent=`共 ${data.total} 条 · 第 ${data.page} / ${data.pages} 页`;
    $('#knowledge-prev').disabled=data.page<=1;$('#knowledge-next').disabled=data.page>=data.pages;

  }catch(e){if(seq===knowledgeSeq){$('#knowledge-page-info').textContent='读取失败，请刷新重试。';toast(e.message)}}
}
async function openKnowledge(id){
  const seq=++detailSeq,expectedMode=mode;
  try{
    const entry=await api('/api/knowledge/'+encodeURIComponent(id));
    if(seq!==detailSeq||expectedMode!==mode||!['knowledge','files'].includes(view))return;
    $('#drawer-title').textContent=entry.row.entity_name;
    const labels={entity_name:'实体名称',entity_type:'实体类型',entity_description:'解释',page_urls:'来源网址',page_titles:'来源标题',full_text:'支持原文（按证据单元汇集）',full_text_length:'原文字符数'};
    const evidence=(entry.evidence||[]).map((fact,index)=>`<article class="qa-card"><strong>事实 ${index+1} · ${esc(fact.locator)}</strong><p>${esc(fact.description)}</p><label>审核引用的原文摘录</label><blockquote>${esc(fact.quote)}</blockquote><div class="review-note">${esc(fact.review.note||'审核通过')}</div></article>`).join('');
    $('#drawer-body').innerHTML=`${view==='files'?`<button class="button ghost" data-knowledge-job="${esc(entry.document_id)}">← 返回处理记录</button>`:''}<div class="detail-summary"><span class="tag">${esc(knowledgeTypes[entry.row.entity_type])}</span><span class="tag">${esc((entry.topics||[entry.topic]).join('、'))}</span><p>${esc(({web:'网页知识',paper:'论文/文档事实',image:'图片资料',data:'数据资料'})[entry.source_kind]||'资料知识')} · ${esc(entry.document_name)} · ${esc(entry.locator)}</p><p>积累 ${entry.fact_count||1} 条已审核事实${entry.fact_count>1?"；解释按原有表述汇集，条件和不同结论分别保留。":""}</p>${entry.source_relocated?'<p class="review-note">资料目录位置已变化，当前原件已通过内容校验；历史登记来源保留。</p>':''}${entry.source_valid?'':'<p class="error-box">原始文件已变更或失效，当前展示历史冻结证据，不能导出。</p>'}</div><p class="review-note">${esc(entry.export_warning||entry.full_text_semantics||'')}</p>${Object.entries(labels).map(([key,label])=>`<div class="qa-field ${key==='full_text'?'knowledge-full-text':''}"><label>${label} · ${key}</label><div>${esc((entry.export_row||entry.row)[key]||'未提供')}</div></div>`).join('')}<div class="detail-section"><h3>逐项事实与证据</h3>${evidence}</div><div class="review-note">${entry.fact_count>1?'事实逐条审核通过':entry.status==='approved'?'审核通过':'审核未通过'}：${esc(entry.review.note||'无补充说明')}</div><div class="qa-field path"><label>原件位置</label><div>${esc(entry.source_path)} | ${esc(entry.locator)}${entry.source_relocated?`<br>历史登记位置：${esc(entry.registered_source_path)}`:''}</div></div><a class="button secondary" href="/api/originals/${encodeURIComponent(entry.document_id)}?mode=${encodeURIComponent(mode)}" target="_blank" rel="noopener">查看原始文件</a>`;
    $('#drawer-overlay').hidden=false;document.body.style.overflow='hidden';$('#drawer-body').scrollTop=0;
  }catch(e){toast(e.message)}
}
document.addEventListener('click',async e=>{
  const job=e.target.closest('[data-knowledge-doc]')?null:e.target.closest('[data-knowledge-job]');
  if(job){openKnowledgeJob(job.dataset.knowledgeJob);return}
  const qa=e.target.closest('[data-start-qa]');
  if(qa){
    qa.disabled=true;
    try{await api('/api/documents/'+encodeURIComponent(qa.dataset.startQa)+'/qa',{method:'POST'});closeDrawer();setView('files');setProcessingTab('qa');await refresh();toast('已加入原有双语 QA 处理队列。')}
    catch(error){toast(error.message)}finally{qa.disabled=false}
    return;
  }
  const repair=e.target.closest('[data-knowledge-repair]');
  if(repair){
    repair.disabled=true;
    try{await api('/api/knowledge/jobs/'+encodeURIComponent(repair.dataset.document)+'/rejected/'+encodeURIComponent(repair.dataset.knowledgeRepair)+'/retry',{method:'POST',body:JSON.stringify({expected_hash:repair.dataset.expected})});toast('已加入单条修复队列，完成后刷新记录。');openKnowledgeJob(repair.dataset.document)}
    catch(error){toast(error.message)}finally{repair.disabled=false}
    return;
  }
  const entry=e.target.closest('[data-knowledge-entry]');
  if(entry){openKnowledge(entry.dataset.knowledgeEntry);return}
  const button=e.target.closest('[data-knowledge-doc]');
  if(!button)return;
  button.disabled=true;
  try{
    await api('/api/documents/'+encodeURIComponent(button.dataset.knowledgeDoc)+'/knowledge',{method:'POST'});
    closeDrawer();setView('files');setProcessingTab('knowledge');toast('已登记知识条目任务；已完成的结果会复用。');
  }catch(error){toast(error.message)}finally{button.disabled=false}
});
$('#knowledge-filters').onsubmit=e=>{e.preventDefault();knowledgePage=1;loadKnowledge()};
for(const id of ['knowledge-type'])$('#'+id).onchange=()=>{knowledgePage=1;loadKnowledge()};
$('#knowledge-refresh').onclick=loadKnowledge;
$('#knowledge-prev').onclick=()=>{knowledgePage--;loadKnowledge()};$('#knowledge-next').onclick=()=>{knowledgePage++;loadKnowledge()};
$('#knowledge-export').onclick=async()=>{
  const button=$('#knowledge-export');button.disabled=true;
  try{
    const query=knowledgeQuery();query.set('mode',mode);
    const response=await fetch('/api/knowledge/export?'+query);
    if(!response.ok){const error=await response.json();throw Error(error.detail||'导出失败')}
    const url=URL.createObjectURL(await response.blob());const a=document.createElement('a');a.href=url;a.download='knowledge.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }catch(e){toast(e.message)}finally{button.disabled=false}
};
setInterval(()=>{if(document.hidden)return;if(view==='knowledge')loadKnowledge();if(view==='files'&&processingTab==='knowledge')loadKnowledgeJobs()},5000);
