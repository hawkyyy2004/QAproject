// The optional catalogue is loaded only when the picker is open.
let libraryPage=1,libraryData=null,librarySeq=0,libraryTimer=null,librarySubmitting=false;
const librarySelection=new Map();
function librarySelectionChanged(){const b=$('#library-import');b.textContent=`开始处理（${librarySelection.size}）`;b.disabled=librarySubmitting||!librarySelection.size}
async function loadLibrary(){
  const seq=++librarySeq;
  try{
    const data=await api('/api/library?q='+encodeURIComponent($('#library-query').value.trim())+'&page='+libraryPage);
    if(seq!==librarySeq||!$('#library-picker').open)return;
    libraryData=data;libraryPage=data.page;
    $('#library-status').textContent=!data.enabled?'尚未配置服务器资料目录，本机选择和拖拽上传仍可正常使用。':data.error||(data.scanning||data.pending?'目录将在文件任务空闲时更新；已有资料仍可选择。':'选择原件后直接处理，不在项目内复制原件。');
    $('#library-query').disabled=!data.enabled;$('#library-search').disabled=!data.enabled;$('#library-refresh').disabled=!data.enabled||librarySubmitting;
    $('#library-page-info').textContent=`共 ${data.total} 份 · 第 ${data.page} / ${data.pages} 页`;
    $('#library-prev').disabled=librarySubmitting||data.page<=1;$('#library-next').disabled=librarySubmitting||data.page>=data.pages;
    $('#library-rows').innerHTML=data.items.length?data.items.map(item=>`<tr><td><input type="checkbox" data-library-id="${esc(item.id)}" aria-label="选择 ${esc(item.name)}" ${librarySelection.has(item.id)?'checked':''} ${librarySubmitting?'disabled':''}></td><td><strong>${esc(item.name)}</strong><div class="file-meta">${esc(item.relative_path)}</div></td><td>${(item.size/1024/1024).toFixed(2)} MB</td><td>${esc(item.status==='available'?'待处理':item.status==='updated'?'原件有更新':labels[item.status]||'已登记')}</td></tr>`).join(''):'<tr><td colspan="4"><div class="empty-state">'+(!data.enabled?'配置目录后可在此选择服务器资料。':'暂无匹配资料，可更新目录或更换搜索词。')+'</div></td></tr>';
    $('#library-rows').querySelectorAll('[data-library-id]').forEach(box=>box.onchange=()=>{
      const item=data.items.find(x=>x.id===box.dataset.libraryId);
      if(box.checked){if(librarySelection.size>=50){box.checked=false;toast('每次最多选择50份资料。');return}librarySelection.set(item.id,{id:item.id,version:item.version})}else librarySelection.delete(item.id);
      librarySelectionChanged();
    });
    clearTimeout(libraryTimer);if(data.scanning||data.pending)libraryTimer=setTimeout(loadLibrary,3000);
  }catch(e){if(seq===librarySeq)$('#library-status').textContent=e.message}
}
$('#choose-library').onclick=()=>{libraryPage=1;librarySelection.clear();librarySelectionChanged();$('#library-picker').showModal();loadLibrary()};
function closeLibrary(){++librarySeq;clearTimeout(libraryTimer);$('#library-picker').close()}
$('#library-close').onclick=closeLibrary;$('#library-picker').addEventListener('cancel',e=>{e.preventDefault();closeLibrary()});
$('#library-search').onclick=()=>{libraryPage=1;loadLibrary()};$('#library-query').addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();libraryPage=1;loadLibrary()}});
$('#library-prev').onclick=()=>{if(libraryPage>1){libraryPage--;loadLibrary()}};$('#library-next').onclick=()=>{if(libraryData&&libraryPage<libraryData.pages){libraryPage++;loadLibrary()}};
$('#library-refresh').onclick=async()=>{try{await api('/api/library/refresh',{method:'POST'});loadLibrary()}catch(e){toast(e.message)}};
$('#library-import').onclick=async()=>{
  if(librarySubmitting||!librarySelection.size)return;
  librarySubmitting=true;librarySelectionChanged();
  try{
    const result=await api('/api/library/imports',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({files:[...librarySelection.values()]})});
    const errors=result.filter(x=>x.error);
    toast(errors.length?`${result.length-errors.length} 份已登记；${errors.length} 份未导入，请刷新检查原件。`:`${result.length} 份资料已登记，后台将继续处理，关闭网页不影响任务。`);
    librarySelection.clear();await refresh();await refreshQueue();
  }catch(e){toast(e.message)}finally{librarySubmitting=false;librarySelectionChanged();if($('#library-picker').open)loadLibrary()}
};
async function refreshQueue(){try{const q=await api('/api/queue');$('#queue-status').textContent=q.running||q.queued?`正在处理 ${q.running} 份 · 等待 ${q.queued} 份 · 关闭网页后后台继续处理`:''}catch(e){/* Keep existing file actions available when status refresh fails. */}}
setInterval(()=>{if(!document.hidden&&view==='files')refreshQueue()},5000);
