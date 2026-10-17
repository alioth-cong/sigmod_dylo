import json,csv,html,time
from pathlib import Path
R=Path(__file__).resolve().parent;O=R.parents[1]/'outputs/table5-std4-20261016'
def table(rows):
 if not rows:return '<p>尚无已完成的实际生成结果。CPU计划不作为实测耗时或token。</p>'
 fields=list(rows[0]);return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in fields)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in fields)+'</tr>' for r in rows)+'</table>'
p=json.loads((O/'protocol.json').read_text());s=json.loads((O/'schedule.json').read_text()) if (O/'schedule.json').exists() else {'status':'CPU_validated_waiting_for_GPU_slot'}
rows=json.loads((O/'results.json').read_text())['rows'] if (O/'results.json').exists() else []
content='<html><meta charset="utf-8"><title>Table5原std4复现</title><style>body{font:15px/1.7 sans-serif;margin:30px}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ccc}</style><h1>Table5原std4复现</h1>'
content+='<p>状态：'+html.escape(s['status'])+'。归档304→327更新图由DyLo、Batch、Rebuild共同使用。原std4二进制和分区复现一致；DyLo及Batch采用成员集合相同且旧报告存在即继承，不追加输入变化刷新；Rebuild全部重写。</p>'
content+='<p>DyLo394=26个delta摘要+368个合并图任务，其中50个是旧索引缺失报告补修。归档协议总成本与剔除初始欠账的净更新成本同时报告；失败调用单列、重试时间保留。沿用原GraphRAG无内部边社区过滤行为，不宣称相同社区覆盖。</p>'
content+='<h2>CPU任务计划</h2>'+table([{'method':k,**v} for k,v in p['checks'].items()])
content+='<h2>已完成的实测结果</h2>'+table(rows)
content+='<p>旧表2513.684k为cl100k逻辑输入计账；本轮原生输入、输出和失败usage分开保存，不能直接视为相同token口径。</p><p>新索引sensemaking：同一60题，先15题预实验；四维独立本地gpt-oss120b裁判，先做顺序/长度/相同答案校验，校验失败不采纳质量分数。质量开销不计维护。<a href="quality/report.html">质量进度</a></p>'
content+='<p>更新于 '+time.strftime('%Y-%m-%d %H:%M:%S%z')+'</p></html>'
(O/'report.html').write_text(content)
