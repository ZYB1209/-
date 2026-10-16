'use strict';
(function(root){
function parseCSV(text){
 const lines=text.replace(/^\uFEFF/,'').trim().split(/\r?\n/);if(lines.length<2)throw Error('CSV没有数据');
 const split=line=>{let out=[],s='',quoted=false;for(let i=0;i<line.length;i++){const c=line[i];if(c==='"'){if(quoted&&line[i+1]==='"'){s+='"';i++;}else quoted=!quoted;}else if(c===','&&!quoted){out.push(s.trim());s='';}else s+=c;}if(quoted)throw Error('不支持跨行引号字段');out.push(s.trim());return out;};
 const head=split(lines[0]).map(v=>v.toLowerCase());if(new Set(head).size!==head.length)throw Error('CSV列名重复');
 const indices=['x','y','z'].map(axis=>Array.from({length:32},(_,c)=>{const n='mag_'+(c+1)+axis,i=head.indexOf(n);if(i<0)throw Error('缺少列：'+n);return i;}));
 const n=lines.length-1;if(n<2||n>100000)throw Error('支持2至100000行数据');
 const axes=indices.map(()=>new Float32Array(n*32));
 for(let r=0;r<n;r++){const row=split(lines[r+1]);if(row.length!==head.length)throw Error('第'+(r+2)+'行列数不一致');for(let a=0;a<3;a++)for(let c=0;c<32;c++){const cell=row[indices[a][c]];const v=Number(cell);if(cell===''||!Number.isFinite(v)||Math.abs(v)>3.4e38)throw Error('第'+(r+2)+'行含无效信号');axes[a][r*32+c]=v;}}
 return{n,axes};
}
function validate(items,n){if(!Array.isArray(items)||items.length>10000)throw Error('标注列表无效');for(const a of items){if(!['X','Y','Z','ALL'].includes(a.axis)||!Number.isInteger(a.start)||!Number.isInteger(a.end)||a.start<0||a.end<a.start||a.end>=n||!Number.isInteger(a.channel_start)||!Number.isInteger(a.channel_end)||a.channel_start<1||a.channel_end<a.channel_start||a.channel_end>32)throw Error('标注坐标无效，请检查采样点和通道范围');if(typeof a.kind!=='string'||typeof a.note!=='string'||a.kind.length>200||a.note.length>5000)throw Error('标注文本格式无效');}return true;}
const api={parseCSV,validate};if(typeof module!=='undefined')module.exports=api;else root.MFLSignal=api;
})(globalThis);
